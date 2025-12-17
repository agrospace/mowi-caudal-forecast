import os
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------
# CONFIG
# -----------------------
TARGET_COL = 'pp'
TIMESTAMP_COL = 'timestamp'
EXCLUDE_REGRESSORS = ['station', TIMESTAMP_COL, TARGET_COL]
future_regressor_strategy = 'linear'
FORECAST_HORIZON_DAYS = 15
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MIN_DATE = pd.Timestamp("2022-02-01")
# -----------------------

# -----------------------
# Data load (assumes concatenated files exist)
# -----------------------
df_x = pd.read_csv('df_x_concatenated.csv', parse_dates=['timestamp'])
df_y = pd.read_csv('df_y_concatenated.csv', parse_dates=['Fecha'])

# Ensure timestamp is datetime
df_x[TIMESTAMP_COL] = pd.to_datetime(df_x[TIMESTAMP_COL])

# Build full hourly range and reindex to fill missing hours
full_range = pd.date_range(start=df_x[TIMESTAMP_COL].min(), end=df_x[TIMESTAMP_COL].max(), freq="H")
df_x = df_x[df_x['timestamp'] >= MIN_DATE]
print(df_x['timestamp'].min())
df_x = (
    df_x.set_index(TIMESTAMP_COL)
        .reindex(full_range)
        .rename_axis(TIMESTAMP_COL)
        .reset_index()
)

# Merge PFA measurements
df_data = df_x.merge(df_y, left_on=TIMESTAMP_COL, right_on="Fecha", how='left')
# drop columns as original
drop_cols = ["Fecha", "Altura (cm)", "station"]
df_data = df_data.drop(columns=[c for c in drop_cols if c in df_data.columns])
df_data.sort_values(TIMESTAMP_COL, inplace=True)
df_data = df_data[df_data['timestamp'] >= MIN_DATE]


# --- Aggregation to daily ---
def aggregate_to_daily(df):
    """
    Aggregates hourly/finer data to daily:
    - Target ('Caudal (L/s)'): mean per day (consistent with your function)
    - Precipitation ('pp'): sum per day
    - Other numeric regressors: mean per day
    """
    df = df.copy()
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.set_index(TIMESTAMP_COL)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET_COL in numeric_cols:
        numeric_cols.remove(TARGET_COL)
    if 'pp' in numeric_cols:
        numeric_cols.remove('pp')

    agg_dict = {TARGET_COL: 'sum', 'pp': 'sum'}
    for col in numeric_cols:
        agg_dict[col] = 'sum'

    df_daily = df.resample('D').agg(agg_dict).reset_index()
    return df_daily


# --- Prepare for Prophet ---
def prepare_df_for_prophet(df):
    df = df.copy()
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    df_prophet = df.rename(columns={TIMESTAMP_COL: 'ds', TARGET_COL: 'y'})

    # Identify regressors
    numeric_cols = df_prophet.select_dtypes(include=[np.number]).columns.tolist()
    regressors = [c for c in numeric_cols if c not in ['y'] and c not in EXCLUDE_REGRESSORS]

    # Drop rows with missing y (we cannot fit on them)
    df_prophet = df_prophet.dropna(subset=['y']).reset_index(drop=True)

    return df_prophet, regressors


# --- Build future regressors ---
def build_future_regressors(history_df, regressors, future_len, strategy='last'):
    """
    history_df: DataFrame with columns ['ds'] + regressors (sorted by ds)
    returns DataFrame with 'ds' starting at last history ds + step for future_len rows,
    and regressor columns filled according to strategy.
    """
    history_df = history_df.sort_values('ds').reset_index(drop=True)
    if history_df.shape[0] >= 2:
        step = history_df['ds'].iloc[-1] - history_df['ds'].iloc[-2]
    else:
        step = pd.Timedelta(days=1)
    last_ds = history_df['ds'].iloc[-1] if history_df.shape[0] >= 1 else pd.Timestamp.today()
    future_dates = [last_ds + (i+1) * step for i in range(future_len)]
    future = pd.DataFrame({'ds': future_dates})

    if len(history_df) == 0:
        for r in regressors:
            future[r] = 0.0
        return future

    if strategy == 'last':
        last_vals = history_df[regressors].iloc[-1]
        for r in regressors:
            future[r] = float(last_vals[r])
    elif strategy == 'median':
        med_vals = history_df[regressors].median()
        for r in regressors:
            future[r] = float(med_vals[r])
    elif strategy == 'linear':
        for r in regressors:
            if len(history_df) >= 2:
                y = history_df[r].values[-2:].astype(float)
                slope = y[1] - y[0]
                future[r] = [float(y[-1] + slope * (i + 1)) for i in range(future_len)]
            else:
                future[r] = float(history_df[r].iloc[-1])
    else:
        raise ValueError("Unknown future regressor strategy")
    return future


def compute_metrics_overall(forecast, df_prophet, regressors=None, require_regressors=False):
    """
    Compute metrics comparing the full forecast to the real measurements (in-sample comparison).
    """
    left = df_prophet[['ds', 'y']].copy()
    merged = left.merge(forecast[['ds', 'yhat']], on='ds', how='inner')

    if require_regressors and regressors:
        for r in regressors:
            if r not in df_prophet.columns:
                raise KeyError(f"Regressor '{r}' not found in df_prophet")
        merged = df_prophet[['ds'] + regressors + ['y']].merge(merged[['ds', 'yhat']], on='ds', how='inner')
        mask = merged['y'].notna() & (~merged[regressors].isnull().any(axis=1))
    else:
        mask = merged['y'].notna()

    used_rows = int(mask.sum())
    total_rows = int(left.shape[0])

    if used_rows == 0:
        return {'rmse': np.nan, 'mae': np.nan, 'mape_%': np.nan, 'r2': np.nan,
                'used_rows': used_rows, 'total_rows': total_rows}

    y_true = merged.loc[mask, 'y'].values
    y_pred = merged.loc[mask, 'yhat'].values

    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-8, y_true))) * 100
    r2 = r2_score(y_true, y_pred)

    return {'rmse': rmse, 'mae': mae, 'mape_%': mape, 'r2': r2,
            'used_rows': used_rows, 'total_rows': total_rows}


# --- Fit on all data and forecast horizon days ---
def fit_and_forecast_full(df_prophet, regressors, horizon_days=15, future_regressor_strategy='last'):
    """
    Fit the model on all available df_prophet and produce a forecast for horizon_days after last ds.
    """

    #df_prophet = df_prophet.sort_values('ds').reset_index(drop=True)
    #df_prophet['y_7d'] = df_prophet['y'].rolling(7).mean()
    #regressors.append('y_7d')

    # --- Fill regressors for fitting (ffill then median)
    if regressors:
        train_for_fit = df_prophet.copy()
        train_for_fit[regressors] = train_for_fit[regressors].ffill().fillna(train_for_fit[regressors].median())
    else:
        train_for_fit = df_prophet.copy()

    # Fit Prophet
    m = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.5,   # try 0.2–0.5
        seasonality_prior_scale=10.0
    )
    for r in regressors:
        m.add_regressor(r)
    m.fit(train_for_fit[['ds', 'y'] + regressors])

    # infer frequency (fallback to daily)
    freq = pd.infer_freq(df_prophet['ds'])
    if freq is None:
        freq = 'D'

    # step length (robust): use median diff
    diffs = df_prophet['ds'].diff().dropna()
    step = diffs.median() if not diffs.empty else pd.Timedelta(days=1)

    last_ds = df_prophet['ds'].max()
    # Build future that includes history and the forecast horizon so components plotting works
    future_end = last_ds + step * horizon_days
    future_dates = pd.date_range(start=df_prophet['ds'].min(), end=future_end, freq=freq)
    future = pd.DataFrame({'ds': future_dates})

    # Merge known regressors from historical df_prophet (may be NaN in gap areas)
    if regressors:
        future = future.merge(df_prophet[['ds'] + regressors], on='ds', how='left')
        # ffill known regressor values
        future[regressors] = future[regressors].ffill()

        # For any remaining NaNs (usually the horizon after last_ds), fill using strategy
        nan_mask = future['ds'] > last_ds
        if nan_mask.any():
            n_future_rows = nan_mask.sum()
            history_for_strategy = df_prophet[['ds'] + regressors].drop_duplicates(subset=['ds']).sort_values('ds')
            fut_reg_df = build_future_regressors(history_for_strategy, regressors, n_future_rows, strategy=future_regressor_strategy)
            # assign built values to rows where ds > last_ds in chronological order
            future.loc[nan_mask, regressors] = fut_reg_df[regressors].values

        # any remaining NaNs -> fill by median
        future[regressors] = future[regressors].fillna(future[regressors].median())

    # Predict
    forecast = m.predict(future)

    # compute in-sample metrics
    metrics = compute_metrics_overall(forecast, df_prophet, regressors=regressors, require_regressors=False)
    print("In-sample metrics (model vs historic y):", metrics)

    return m, forecast, df_prophet, future, metrics


# --- Plotting (save to files) ---
def plot_forecast_vs_history_save(forecast, df_prophet, last_ds, output_path, metrics=None, title_suffix=''):
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df_prophet['ds'], df_prophet['y'], label='observed (y)', linewidth=1)
    ax.plot(forecast['ds'], forecast['yhat'], label='forecast (yhat)', linewidth=1, alpha=0.9)
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2, label='confianza')

    # mark forecast horizon
    ax.axvline(last_ds, color='black', linestyle='--', linewidth=1, label='last observed')

    # metrics box
    if metrics:
        metrics_text = f"RMSE: {metrics['rmse']:.2f}\nMAE: {metrics['mae']:.2f}\nR²: {metrics['r2']:.3f}"
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    ax.set_xlabel('ds')
    ax.set_ylabel(TARGET_COL)
    ax.set_title(f'Forecast {TARGET_COL}' + title_suffix)
    ax.grid(True)
    ax.set_ylim(0, None)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_forecast_closeup_save(forecast, df_prophet, last_ds, output_path, days_back=90, title_suffix=''):
    start_plot = last_ds - pd.Timedelta(days=days_back)
    df_prophet_plot = df_prophet[df_prophet['ds'] >= start_plot]
    forecast_plot = forecast[forecast['ds'] >= start_plot]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_prophet_plot['ds'], df_prophet_plot['y'], label='observed (y)', linewidth=1)
    ax.plot(forecast_plot['ds'], forecast_plot['yhat'], label='forecast (yhat)', linewidth=1, alpha=0.9)
    ax.fill_between(forecast_plot['ds'], forecast_plot['yhat_lower'], forecast_plot['yhat_upper'], alpha=0.2, label='confianza')

    ax.axvline(last_ds, color='black', linestyle='--', linewidth=1, label='last observed')

    ax.set_xlabel('ds')
    ax.set_ylabel(TARGET_COL)
    ax.set_title(f'Forecast {TARGET_COL} (closeup)' + title_suffix)
    ax.grid(True)
    ax.set_ylim(0, None)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_prophet_components_save(model, forecast, output_path):
    fig = model.plot_components(forecast)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# --- Main pipeline ---
def run_prophet_pipeline(df_data):
    df_daily = aggregate_to_daily(df_data)
    df_prophet, regressors = prepare_df_for_prophet(df_daily)
    df_prophet['cap'] = 100
    df_prophet['floor'] = 0

    model, forecast, df_prophet_used, future_df, metrics = fit_and_forecast_full(
        df_prophet,
        regressors,
        horizon_days=FORECAST_HORIZON_DAYS,
        future_regressor_strategy=future_regressor_strategy
    )

    last_ds = df_prophet_used['ds'].max()

    # Save forecast CSV (only the horizon portion)
    horizon_mask = future_df['ds'] > last_ds
    forecast_horizon = forecast[forecast['ds'].isin(future_df.loc[horizon_mask, 'ds'])]
    forecast_horizon[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(os.path.join(OUTPUT_DIR, 'caudal_forecast_daily.csv'), index=False)

    # Save plots
    plot_forecast_vs_history_save(forecast, df_prophet_used, last_ds, os.path.join(OUTPUT_DIR, 'forecast_vs_history.png'), metrics=metrics, title_suffix=' (daily)')
    plot_prophet_components_save(model, forecast, os.path.join(OUTPUT_DIR, 'prophet_components.png'))
    plot_forecast_closeup_save(forecast, df_prophet_used, last_ds, os.path.join(OUTPUT_DIR, 'forecast_closeup_90d.png'), days_back=90)

    print("Saved outputs to:", OUTPUT_DIR)
    return {
        'model': model,
        'forecast': forecast,
        'df_prophet': df_prophet_used,
        'future': future_df,
        'metrics': metrics,
        'regressors': regressors
    }


# -----------------------
# Run
# -----------------------
result = run_prophet_pipeline(df_data)
