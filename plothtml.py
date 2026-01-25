# # # # # # # # # # # import asyncio
# # # # # # # # # # # import asyncpg
# # # # # # # # # # # import json
# # # # # # # # # # # import plotly.graph_objects as go

# # # # # # # # # # # async def get_data():
# # # # # # # # # # #     conn = await asyncpg.connect("postgresql://memphis:Memphis%401234%24@192.168.18.175/obanext_8")
# # # # # # # # # # #     row = await conn.fetchrow("""
# # # # # # # # # # #         SELECT ae_response FROM "ModelStatusMaster" 
# # # # # # # # # # #         WHERE fk_vessel = 8 AND equipment_hierarchy = 'ME_cyl'
# # # # # # # # # # #     """)
# # # # # # # # # # #     await conn.close()
# # # # # # # # # # #     return json.loads(row['ae_response'])

# # # # # # # # # # # data = asyncio.run(get_data())

# # # # # # # # # # # # Extract data
# # # # # # # # # # # timestamps = data['anomaly_detection_overall']['timestamps']
# # # # # # # # # # # errors = data['anomaly_detection_overall']['reconstruction_errors']
# # # # # # # # # # # threshold_upper = data['anomaly_detection_overall']['threshold_upper']
# # # # # # # # # # # threshold_lower = data['anomaly_detection_overall']['threshold_lower']
# # # # # # # # # # # anomaly_flags = data['anomaly_detection_overall']['anomaly_flags']

# # # # # # # # # # # # Count anomalies
# # # # # # # # # # # num_anomalies = sum(anomaly_flags)
# # # # # # # # # # # print(f"Total points: {len(errors)}")
# # # # # # # # # # # print(f"Anomalies: {num_anomalies}")
# # # # # # # # # # # print(f"Threshold upper: {threshold_upper}")
# # # # # # # # # # # print(f"Max error: {max(errors)}")
# # # # # # # # # # # print(f"Errors above threshold: {sum(1 for e in errors if e > threshold_upper)}")

# # # # # # # # # # # # Plot
# # # # # # # # # # # fig = go.Figure()

# # # # # # # # # # # # Reconstruction errors
# # # # # # # # # # # fig.add_trace(go.Scatter(
# # # # # # # # # # #     x=timestamps,
# # # # # # # # # # #     y=errors,
# # # # # # # # # # #     mode='markers',
# # # # # # # # # # #     name='Reconstruction Error',
# # # # # # # # # # #     marker=dict(color='blue', size=5)
# # # # # # # # # # # ))

# # # # # # # # # # # # Upper threshold line
# # # # # # # # # # # fig.add_hline(y=threshold_upper, line_dash="dash", line_color="red", 
# # # # # # # # # # #               annotation_text=f"Upper ({threshold_upper})")

# # # # # # # # # # # # Lower threshold line
# # # # # # # # # # # fig.add_hline(y=threshold_lower, line_dash="dash", line_color="green",
# # # # # # # # # # #               annotation_text=f"Lower ({threshold_lower})")

# # # # # # # # # # # fig.update_layout(
# # # # # # # # # # #     title="Autoencoder Anomaly Detection - OVERALL",
# # # # # # # # # # #     xaxis_title="Timestamp",
# # # # # # # # # # #     yaxis_title="Reconstruction Error",
# # # # # # # # # # #     height=600
# # # # # # # # # # # )

# # # # # # # # # # # fig.write_html("overall_plot.html")
# # # # # # # # # # # print("Saved to overall_plot.html - open in browser")

# # # # # # # # # # import asyncio
# # # # # # # # # # import asyncpg
# # # # # # # # # # import json
# # # # # # # # # # import plotly.graph_objects as go

# # # # # # # # # # async def get_data():
# # # # # # # # # #     conn = await asyncpg.connect("postgresql://memphis:Memphis%401234%24@192.168.18.175/obanext_8")
# # # # # # # # # #     row = await conn.fetchrow("""
# # # # # # # # # #         SELECT ae_response FROM "ModelStatusMaster" 
# # # # # # # # # #         WHERE fk_vessel = 8 AND equipment_hierarchy = 'ME>>cyl'
# # # # # # # # # #     """)
# # # # # # # # # #     await conn.close()
# # # # # # # # # #     return json.loads(row['ae_response'])

# # # # # # # # # # data = asyncio.run(get_data())

# # # # # # # # # # # Extract data
# # # # # # # # # # timestamps = data['anomaly_detection_overall']['timestamps']
# # # # # # # # # # errors = data['anomaly_detection_overall']['reconstruction_errors']
# # # # # # # # # # threshold_upper = data['anomaly_detection_overall']['threshold_upper']
# # # # # # # # # # threshold_lower = data['anomaly_detection_overall']['threshold_lower']
# # # # # # # # # # anomaly_flags = data['anomaly_detection_overall']['anomaly_flags']

# # # # # # # # # # # Count anomalies
# # # # # # # # # # num_anomalies = sum(anomaly_flags)
# # # # # # # # # # print(f"Total points: {len(errors)}")
# # # # # # # # # # print(f"Anomalies: {num_anomalies}")
# # # # # # # # # # print(f"Threshold upper: {threshold_upper}")
# # # # # # # # # # print(f"Max error: {max(errors)}")
# # # # # # # # # # print(f"Errors above threshold: {sum(1 for e in errors if e > threshold_upper)}")

# # # # # # # # # # # Plot
# # # # # # # # # # fig = go.Figure()

# # # # # # # # # # # Reconstruction errors
# # # # # # # # # # fig.add_trace(go.Scatter(
# # # # # # # # # #     x=timestamps,
# # # # # # # # # #     y=errors,
# # # # # # # # # #     mode='markers',
# # # # # # # # # #     name='Reconstruction Error',
# # # # # # # # # #     marker=dict(color='blue', size=5)
# # # # # # # # # # ))

# # # # # # # # # # # Upper threshold line
# # # # # # # # # # fig.add_hline(y=threshold_upper, line_dash="dash", line_color="red", 
# # # # # # # # # #               annotation_text=f"Upper ({threshold_upper})")

# # # # # # # # # # # Lower threshold line
# # # # # # # # # # fig.add_hline(y=threshold_lower, line_dash="dash", line_color="green",
# # # # # # # # # #               annotation_text=f"Lower ({threshold_lower})")

# # # # # # # # # # fig.update_layout(
# # # # # # # # # #     title="Autoencoder Anomaly Detection - OVERALL",
# # # # # # # # # #     xaxis_title="Timestamp",
# # # # # # # # # #     yaxis_title="Reconstruction Error",
# # # # # # # # # #     height=600
# # # # # # # # # # )

# # # # # # # # # # fig.write_html("overall_plot.html")
# # # # # # # # # # print("Saved to overall_plot.html - open in browser")
# # # # # # # # # # import pickle

# # # # # # # # # # with open("models/9665657_ME_cyl/autoencoder/ae_input_scaler.pkl", "rb") as f:
# # # # # # # # # #     scaler = pickle.load(f)

# # # # # # # # # # print("mean:", scaler.mean_[:3])
# # # # # # # # # # print("scale:", scaler.scale_[:3])
# # # # # # # # # import pandas as pd

# # # # # # # # # # Load data
# # # # # # # # # df = pd.read_csv(r"C:\Users\User\Desktop\siemens\freya_schulte\combined_vessels_15min.csv")
# # # # # # # # # df['Local_time'] = pd.to_datetime(df['Local_time'], format='mixed', dayfirst=True)

# # # # # # # # # # Filter Aug 7-8
# # # # # # # # # df_aug7 = df[(df['Local_time'] >= '2025-08-07') & (df['Local_time'] <= '2025-08-08')]

# # # # # # # # # print(f"Rows: {len(df_aug7)}")
# # # # # # # # # print()

# # # # # # # # # # Check the source columns
# # # # # # # # # cols = ['M_E_NO1_CYL_EXH_GAS_OUT_TEMP', 'M_E_NO2_CYL_EXH_GAS_OUT_TEMP', 
# # # # # # # # #         'M_E_NO3_CYL_EXH_GAS_OUT_TEMP', 'M_E_NO4_CYL_EXH_GAS_OUT_TEMP',
# # # # # # # # #         'M_E_NO5_CYL_EXH_GAS_OUT_TEMP', 'M_E_NO6_CYL_EXH_GAS_OUT_TEMP']

# # # # # # # # # for col in cols:
# # # # # # # # #     if col in df_aug7.columns:
# # # # # # # # #         print(f"{col}: min={df_aug7[col].min():.2f}, max={df_aug7[col].max():.2f}, mean={df_aug7[col].mean():.2f}")
# # # # # # # # #     else:
# # # # # # # # #         print(f"{col}: NOT FOUND")

# # # # # # # # # print()

# # # # # # # # # # Calculate ME_CYL_EXHAUST_GAS_TEMPERATURE manually
# # # # # # # # # existing_cols = [c for c in cols if c in df_aug7.columns]
# # # # # # # # # if existing_cols:
# # # # # # # # #     df_aug7['ME_CYL_EXHAUST_GAS_TEMPERATURE'] = df_aug7[existing_cols].mean(axis=1)
# # # # # # # # #     print(f"ME_CYL_EXHAUST_GAS_TEMPERATURE: min={df_aug7['ME_CYL_EXHAUST_GAS_TEMPERATURE'].min():.2f}, max={df_aug7['ME_CYL_EXHAUST_GAS_TEMPERATURE'].max():.2f}")
# # # # # # # # import pandas as pd

# # # # # # # # df = pd.read_csv(r"C:\Users\User\Desktop\siemens\freya_schulte\combined_vessels_15min.csv")
# # # # # # # # df['Local_time'] = pd.to_datetime(df['Local_time'], format='mixed', dayfirst=True)

# # # # # # # # print(f"Date range: {df['Local_time'].min()} to {df['Local_time'].max()}")
# # # # # # # # print(f"Total rows: {len(df)}")

# # # # # # # # # Check 80% split point
# # # # # # # # split_idx = int(0.8 * len(df))
# # # # # # # # print(f"Train: rows 0-{split_idx}, Val: rows {split_idx}-{len(df)}")
# # # # # # # # print(f"Train ends at: {df['Local_time'].iloc[split_idx]}")

# # # # # # # # import asyncio
# # # # # # # # import asyncpg
# # # # # # # # import json

# # # # # # # # async def check():
# # # # # # # #     conn = await asyncpg.connect("postgresql://memphis:Memphis%401234%24@192.168.18.175/obanext_8")
# # # # # # # #     row = await conn.fetchrow("""
# # # # # # # #         SELECT ae_response FROM "ModelStatusMaster" 
# # # # # # # #         WHERE fk_vessel = 8 AND equipment_hierarchy = 'ME>>cyl'
# # # # # # # #     """)
# # # # # # # #     await conn.close()
    
# # # # # # # #     data = json.loads(row['ae_response'])
# # # # # # # #     timestamps = data['anomaly_detection_overall']['timestamps']
# # # # # # # #     errors = data['anomaly_detection_overall']['reconstruction_errors']
    
# # # # # # # #     # Find Aug 7 entries
# # # # # # # #     for ts, err in zip(timestamps, errors):
# # # # # # # #         if '2025-08-07' in ts:
# # # # # # # #             print(f"{ts}: {err}")

# # # # # # # # asyncio.run(check())

# # # # # # # # import pandas as pd

# # # # # # # # df = pd.read_csv(r"C:\Users\User\Desktop\siemens\freya_schulte\combined_vessels_15min.csv")
# # # # # # # # df['Local_time'] = pd.to_datetime(df['Local_time'], format='mixed', dayfirst=True)

# # # # # # # # # Filter Aug 7-8
# # # # # # # # df_aug7 = df[(df['Local_time'] >= '2025-08-07') & (df['Local_time'] <= '2025-08-08')]

# # # # # # # # print(f"Rows: {len(df_aug7)}")
# # # # # # # # print()
# # # # # # # # print("Timestamps around 21:30:")
# # # # # # # # for ts in df_aug7['Local_time']:
# # # # # # # #     if '21:' in str(ts):
# # # # # # # #         print(ts)

# # # # # # # import pandas as pd

# # # # # # # df = pd.read_csv(r"C:\Users\User\Desktop\siemens\freya_schulte\combined_vessels_15min.csv")
# # # # # # # df['Local_time'] = pd.to_datetime(df['Local_time'], format='mixed', dayfirst=True)

# # # # # # # # Get exact 21:30 row
# # # # # # # row_2130 = df[df['Local_time'] == '2025-08-07 21:30:00']

# # # # # # # print(f"Rows found: {len(row_2130)}")
# # # # # # # print()

# # # # # # # # Check key features
# # # # # # # features = ['ME_Load@AVG', 'ME_SCAV_AIR_PRESS', 'SHAFT_POWER', 'SHAFT_TORQUE', 'ME_RPM']
# # # # # # # for f in features:
# # # # # # #     if f in row_2130.columns:
# # # # # # #         print(f"{f}: {row_2130[f].values}")

# # # # # # # import pandas as pd

# # # # # # # df = pd.read_csv(r"C:\Users\User\Desktop\siemens\freya_schulte\combined_vessels_15min.csv")
# # # # # # # df['Local_time'] = pd.to_datetime(df['Local_time'], format='mixed', dayfirst=True)

# # # # # # # # Get exact 21:30 rows
# # # # # # # rows = df[df['Local_time'] == '2025-08-07 21:30:00']

# # # # # # # print(f"Rows found: {len(rows)}")
# # # # # # # print()
# # # # # # # print(rows[['vessel_id', 'Local_time', 'ME_Load@AVG', 'SHAFT_POWER']].to_string())
# # # # # # # import pandas as pd

# # # # # # # df = pd.read_csv(r"C:\Users\User\Desktop\siemens\freya_schulte\combined_vessels_15min.csv")
# # # # # # # print(df[['vessel_id']].drop_duplicates())
# # # # # # # import asyncio
# # # # # # # import asyncpg

# # # # # # # async def check():
# # # # # # #     conn = await asyncpg.connect("postgresql://memphis:Memphis%401234%24@192.168.18.175/obanext_8")
# # # # # # #     rows = await conn.fetch('SELECT id, imo_number FROM "VesselMaster"')
# # # # # # #     for r in rows:
# # # # # # #         print(f"id={r['id']}, imo={r['imo_number']}")
# # # # # # #     await conn.close()

# # # # # # # asyncio.run(check())
# # # # # # import pandas as pd

# # # # # # df = pd.read_csv(r"C:\Users\User\Desktop\siemens\freya_schulte\combined_vessels_15min.csv")

# # # # # # # Check each vessel_id's data characteristics
# # # # # # for vid in [0, 1, 2]:
# # # # # #     subset = df[df['vessel_id'] == vid]
# # # # # #     print(f"vessel_id={vid}: rows={len(subset)}, date_range={subset['Local_time'].min()} to {subset['Local_time'].max()}")
# # # # # # import pandas as pd

# # # # # # df = pd.read_csv(r"C:\Users\User\Desktop\siemens\freya_schulte\combined_vessels_15min.csv")
# # # # # # df['Local_time'] = pd.to_datetime(df['Local_time'], format='mixed', dayfirst=True)

# # # # # # for vid in [0, 1, 2]:
# # # # # #     subset = df[df['vessel_id'] == vid]
# # # # # #     print(f"vessel_id={vid}: {subset['Local_time'].min()} to {subset['Local_time'].max()}")
# # # # # import pandas as pd

# # # # # df = pd.read_csv(r"C:\Users\User\Desktop\siemens\freya_schulte\combined_vessels_15min.csv")
# # # # # df['Local_time'] = pd.to_datetime(df['Local_time'], format='mixed', dayfirst=True)

# # # # # # Filter to vessel_id=2 only
# # # # # df = df[df['vessel_id'] == 2]

# # # # # # Get Aug 7 21:30
# # # # # row = df[df['Local_time'] == '2025-08-07 21:30:00']
# # # # # print(f"Rows: {len(row)}")
# # # # # print(row[['Local_time', 'ME_Load@AVG', 'SHAFT_POWER', 'ME_RPM']].to_string())
# # # # import asyncio
# # # # import asyncpg
# # # # import json

# # # # async def check():
# # # #     conn = await asyncpg.connect("postgresql://memphis:Memphis%401234%24@192.168.18.175/obanext_8")
# # # #     row = await conn.fetchrow("""
# # # #         SELECT ae_response FROM "ModelStatusMaster" 
# # # #         WHERE fk_vessel = 8 AND equipment_hierarchy = 'ME>>cyl'
# # # #     """)
# # # #     await conn.close()
    
# # # #     data = json.loads(row['ae_response'])
# # # #     timestamps = data['anomaly_detection_overall']['timestamps']
    
# # # #     print(f"First timestamp: {timestamps[0]}")
# # # #     print(f"Last timestamp: {timestamps[-1]}")

# # # # # asyncio.run(check())
# # # # import pandas as pd

# # # # df = pd.read_csv(r"C:\Users\User\Desktop\siemens\freya_schulte\combined_vessels_15min.csv")
# # # # df['Local_time'] = pd.to_datetime(df['Local_time'], format='mixed', dayfirst=True)

# # # # # Check ME_Tc_rpm source tags for Aug 7-8
# # # # aug7 = df[(df['Local_time'] >= '2025-08-07') & (df['Local_time'] <= '2025-08-08')]

# # # # print("ME_NO_1_T_C_RPM for Aug 7-8:")
# # # # if 'ME_NO_1_T_C_RPM' in aug7.columns:
# # # #     print(f"  min={aug7['ME_NO_1_T_C_RPM'].min()}, max={aug7['ME_NO_1_T_C_RPM'].max()}")

# # # # print("\nME_NO_2_T_C_RPM for Aug 7-8:")  
# # # # if 'ME_NO_2_T_C_RPM' in aug7.columns:
# # # #     print(f"  min={aug7['ME_NO_2_T_C_RPM'].min()}, max={aug7['ME_NO_2_T_C_RPM'].max()}")
# # # # import pandas as pd

# # # # df = pd.read_csv(r"C:\Users\User\Desktop\siemens\freya_schulte\combined_vessels_15min.csv")
# # # # df['Local_time'] = pd.to_datetime(df['Local_time'], format='mixed', dayfirst=True)

# # # # # Check ME_Tc_rpm source tags for Aug 7-8
# # # # aug7 = df[(df['Local_time'] >= '2025-08-07') & (df['Local_time'] <= '2025-08-08')]

# # # # print("ME_NO_1_TC_RPM for Aug 7-8:")
# # # # print(f"  min={aug7['ME_NO_1_TC_RPM'].min()}, max={aug7['ME_NO_1_TC_RPM'].max()}")

# # # # print("\nME_NO_2_TC_RPM for Aug 7-8:")  
# # # # print(f"  min={aug7['ME_NO_2_TC_RPM'].min()}, max={aug7['ME_NO_2_TC_RPM'].max()}")

# # # # # Also check Sep data (where values look good in overall)
# # # # sep = df[(df['Local_time'] >= '2025-09-15') & (df['Local_time'] <= '2025-09-16')]

# # # # print("\n\nME_NO_1_TC_RPM for Sep 15-16:")
# # # # print(f"  min={sep['ME_NO_1_TC_RPM'].min()}, max={sep['ME_NO_1_TC_RPM'].max()}")

# # # # print("\nME_NO_2_TC_RPM for Sep 15-16:")  
# # # # print(f"  min={sep['ME_NO_2_TC_RPM'].min()}, max={sep['ME_NO_2_TC_RPM'].max()}")

# # # import pandas as pd

# # # df = pd.read_csv(r"C:\Users\User\Desktop\siemens\freya_schulte\combined_vessels_15min.csv")
# # # df['Local_time'] = pd.to_datetime(df['Local_time'], format='mixed', dayfirst=True)

# # # # Check Aug 7 21:30 for all vessels
# # # rows = df[df['Local_time'] == '2025-08-07 21:30:00']

# # # print("Aug 7 21:30 - all vessels:")
# # # for _, row in rows.iterrows():
# # #     print(f"  vessel_id={row['vessel_id']}, ME_NO_1_TC_RPM={row['ME_NO_1_TC_RPM']}, ME_NO_2_TC_RPM={row['ME_NO_2_TC_RPM']}")

# # import pandas as pd

# # df = pd.read_csv(r"C:\Users\User\Desktop\siemens\freya_schulte\combined_vessels_15min.csv")
# # df['Local_time'] = pd.to_datetime(df['Local_time'], format='mixed', dayfirst=True)

# # # Check Aug 7 21:30 for vessel_id=1 (the one that had ME_Load=22.4)
# # row = df[(df['Local_time'] == '2025-08-07 21:30:00') & (df['vessel_id'] == 1)]

# # # Check ALL features used in model
# # features = ['ME_Load@AVG', 'ME_SCAV_AIR_PRESS', 'M_E_NO_1_T_C_EXH_GAS_IN_TEMP', 
# #             'ME_NO_2_TC_LO_OUT_TEMP', 'M_E_NO5_CYL_PCO_OUT_TEMP', 
# #             'BOILER_EXH_GAS_OUT_M_E_TEMP_H', 'SHAFT_TORQUE', 'ME_RPM',
# #             'ME_NO1_CYL_LINER_PP_SIDE_TEMP', 'ME_NO2_CYL_LINER_EXH_SIDE_TEMP', 'SHAFT_POWER']

# # print("Aug 7 21:30 vessel_id=1 features:")
# # for f in features:
# #     if f in row.columns:
# #         print(f"  {f}: {row[f].values[0]}")
# # import pandas as pd

# # df = pd.read_csv(r"C:\Users\User\Desktop\siemens\freya_schulte\combined_vessels_15min.csv")
# # df['Local_time'] = pd.to_datetime(df['Local_time'], format='mixed', dayfirst=True)

# # # Find the exact 22.41975933 row
# # row = df[(df['ME_Load@AVG'] > 22.419) & (df['ME_Load@AVG'] < 22.420)]
# # print(f"Rows found: {len(row)}")
# # print(row[['vessel_id', 'Local_time', 'ME_Load@AVG', 'SHAFT_POWER', 'ME_RPM']].to_string())

# import torch
# import json
# import pickle
# import numpy as np

# model_dir = r"C:\Users\User\Desktop\siemens\freya_schulte\models\9665657_ME_cyl\autoencoder"

# # Load metadata
# with open(f"{model_dir}/autoencoder_metadata.json", 'r') as f:
#     metadata = json.load(f)

# print(f"input_dim: {metadata['input_dim']}")
# print(f"encoder_dims: {metadata['encoder_dims']}")
# print(f"latent_dim: {metadata['latent_dim']}")
# print(f"dropout: {metadata['best_params']['dropout']}")

import torch
import json
import pickle
import numpy as np
from autoencoder2 import DenseAutoencoder

model_dir = r"C:\Users\User\Desktop\siemens\freya_schulte\models\9665657_ME_cyl\autoencoder"

# Load metadata
with open(f"{model_dir}/autoencoder_metadata.json", 'r') as f:
    metadata = json.load(f)

# Load scaler
with open(f"{model_dir}/ae_input_scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

# Create model
model = DenseAutoencoder(
    input_dim=metadata['input_dim'],
    encoder_dims=metadata['encoder_dims'],
    latent_dim=metadata['latent_dim'],
    dropout_rate=metadata['best_params']['dropout']
)

# Load weights
model.load_state_dict(torch.load(f"{model_dir}/autoencoder_model.pth", map_location='cpu'))
model.eval()

# Create the exact row: ME_Load=22.4197, rest zeros (with 0.001 for calculated cols)
test_row = np.array([[22.4197593, 0.001, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# Scale
test_scaled = scaler.transform(test_row)

# Reconstruct
with torch.no_grad():
    recon_scaled = model(torch.FloatTensor(test_scaled)).numpy()

# Inverse transform
test_original = scaler.inverse_transform(test_scaled)
recon_original = scaler.inverse_transform(recon_scaled)

print("Original:", test_original[0])
print("Reconstructed:", recon_original[0])

# Calculate error
error = np.sqrt(np.sum((test_original - recon_original) ** 2))
print(f"\nReconstruction error: {error}")