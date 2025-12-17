"""
Script descartable para preparar los archivos concatenados.
Ejecutar una vez para generar df_x_concatenated.csv y df_y_concatenated.csv
"""
import pandas as pd

print("Creating df_x_concatenated.csv...")
df_x = pd.read_excel('data-meteo-pfa_filtrado.xlsx', sheet_name='data-meteo-pfa')

df_patm = pd.read_csv('measurement_1h_detail_rows.csv', parse_dates=['timestamp'])
df_patm.rename(columns={'value': 'patm'}, inplace=True)
df_x = df_x.merge(df_patm[['timestamp', 'patm']], on='timestamp', how='left')

df_x.to_csv('df_x_concatenated.csv', index=False)
print(f"✓ Saved df_x_concatenated.csv ({len(df_x)} rows)")

print("\nCreating df_y_concatenated.csv...")
df_y = pd.concat(
    [pd.read_excel(f'PFA {year}.xlsx', sheet_name='1830-H1 - Caudal Ecológico PFA') for year in ['2023', '2024', '2025']],
    axis=0
)

df_y.to_csv('df_y_concatenated.csv', index=False)
print(f"✓ Saved df_y_concatenated.csv ({len(df_y)} rows)")

print("\n✓ Done! You can now run prophet_caudal_pipeline.py")

