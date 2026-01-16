import torch
import torch.nn as nn
import pymongo
import pandas as pd
import numpy as np
import pickle
import json
import math
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================================
# MONGODB CONNECTION
# ============================================================================

MONGO_URI = "mongodb://admin:secret@192.168.17.21:27017/?authSource=admin"
client = pymongo.MongoClient(MONGO_URI)
db = client["bsm"]
collection = db["imo_9665669"]  # Clemens

print("🔌 Connected to MongoDB!")

# ============================================================================
# MODEL ARCHITECTURE (Copy from your training script)
# ============================================================================

def generate_square_subsequent_mask(sz, device):
    mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class LSTMTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(LSTMTransformerEncoderLayer, self).__init__()
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout_rate, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dff, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
    
    def forward(self, x, mask=None):
        lstm_out, (h, c) = self.lstm(x)
        attn_out, _ = self.mha(lstm_out, lstm_out, lstm_out, attn_mask=mask)
        attn_out = self.dropout1(attn_out)
        out1 = self.layernorm1(lstm_out + attn_out)
        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out)
        out2 = self.layernorm2(out1 + ffn_out)
        return out2, h, c

class LSTMTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(LSTMTransformerDecoderLayer, self).__init__()
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.mha1 = nn.MultiheadAttention(d_model, num_heads, dropout=dropout_rate, batch_first=True)
        self.mha2 = nn.MultiheadAttention(d_model, num_heads, dropout=dropout_rate, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dff, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
    
    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        lstm_out, (h, c) = self.lstm(x)
        attn1, _ = self.mha1(lstm_out, lstm_out, lstm_out, attn_mask=tgt_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(lstm_out + attn1)
        attn2, _ = self.mha2(out1, enc_output, enc_output, attn_mask=memory_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(out1 + attn2)
        ffn_out = self.ffn(out2)
        ffn_out = self.dropout3(ffn_out)
        out3 = self.layernorm3(out2 + ffn_out)
        return out3

class LSTMTransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, input_dim, dropout_rate=0.1):
        super(LSTMTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            LSTMTransformerEncoderLayer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, mask=None):
        x = self.input_projection(x)
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x, h, c = layer(x, mask)
        return x

class LSTMTransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, output_dim, dropout_rate=0.1):
        super(LSTMTransformerDecoder, self).__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(output_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            LSTMTransformerDecoderLayer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.output_projection = nn.Linear(d_model, output_dim)
    
    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        x = self.input_projection(x)
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, memory_mask)
        x = self.output_projection(x)
        return x

class LSTMTransformerSeq2Seq(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, input_dim, output_dim, dropout_rate=0.1):
        super(LSTMTransformerSeq2Seq, self).__init__()
        self.encoder = LSTMTransformerEncoder(num_layers, d_model, num_heads, dff, input_dim, dropout_rate)
        self.decoder = LSTMTransformerDecoder(num_layers, d_model, num_heads, dff, output_dim, dropout_rate)
    
    def forward(self, src, tgt, tgt_mask=None):
        enc_output = self.encoder(src)
        dec_output = self.decoder(tgt, enc_output, tgt_mask=tgt_mask)
        return dec_output

# ============================================================================
# AUTOREGRESSIVE PREDICTION
# ============================================================================

def autoregressive_predict(model, X_input, output_dim, horizon, device):
    model.eval()
    batch_size = X_input.size(0)
    
    with torch.no_grad():
        enc_output = model.encoder(X_input)
        tgt = torch.zeros(batch_size, horizon, output_dim).to(device)
        tgt[:, 0:1, :] = X_input[:, -1:, :output_dim]
        
        for t in range(1, horizon):
            mask = generate_square_subsequent_mask(t, device)
            dec_output = model.decoder(tgt[:, :t, :], enc_output, tgt_mask=mask)
            tgt[:, t:t+1, :] = dec_output[:, -1:, :]
        
        mask = generate_square_subsequent_mask(horizon, device)
        output = model.decoder(tgt, enc_output, tgt_mask=mask)
    
    return output.cpu().numpy()

# ============================================================================
# FETCH DATA FROM MONGODB
# ============================================================================

def fetch_data_from_mongo(collection, end_time, hours_back):
    """
    Fetch per-minute data from MongoDB
    
    Args:
        collection: MongoDB collection
        end_time: datetime object (e.g., 6 hours ago)
        hours_back: how many hours of history to fetch
    """
    start_time = end_time - timedelta(hours=hours_back)
    
    print(f"📥 Fetching data from {start_time} to {end_time}")
    
    # Query MongoDB
    # query = {
    #     "Local_time": {
    #         "$gte": start_time,
    #         "$lte": end_time
    #     }
    # }
    query = {
    "Local_time": {
        "$gte": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "$lte": end_time.strftime("%Y-%m-%d %H:%M:%S")
    }
}
    
    # Fetch documents
    cursor = collection.find(query).sort("Local_time", 1)
    
    # Convert to DataFrame
    df = pd.DataFrame(list(cursor))
    
    if len(df) == 0:
        raise ValueError("No data found in MongoDB for the specified time range!")
    
    print(f"✅ Fetched {len(df)} per-minute records")
    
    return df

# ============================================================================
# MAIN TESTING PIPELINE
# ============================================================================

def main():
    print("\n" + "="*70)
    print("🧪 REAL-TIME MODEL TESTING WITH HISTORICAL DATA")
    print("="*70)
    
    # ========================================================================
    # 1. LOAD MODEL AND METADATA
    # ========================================================================
    
    model_dir = "models/combined_vessels"
    
    with open(f"{model_dir}/metadata.json", 'r') as f:
        metadata = json.load(f)
    
    with open(f"{model_dir}/input_scaler.pkl", 'rb') as f:
        input_scaler = pickle.load(f)
    
    with open(f"{model_dir}/output_scaler.pkl", 'rb') as f:
        output_scaler = pickle.load(f)
    
    print("✅ Model metadata and scalers loaded")

    #########################################################################################
    print("\n📥 Fetching recent data from MongoDB...")
    # cursor = collection.find().sort("Local_time", -1).limit(12000)  # ~8 days at 1-min
    cursor = collection.find().sort([("Local_time", -1)]).allow_disk_use(True).limit(12000)
    df_raw = pd.DataFrame(list(cursor))

    if len(df_raw) == 0:
        raise ValueError("Collection is empty!")
    
    # Use 'time' column (epoch timestamp)
    if 'time' not in df_raw.columns:
        raise ValueError("'time' column not found!")
    df_raw['Local_time'] = pd.to_datetime(df_raw['time'], unit='s')
    df_raw = df_raw.sort_values('Local_time').reset_index(drop=True)
    print(f"✅ Fetched {len(df_raw)} per-minute records")
    print(f"   Date range: {df_raw['Local_time'].min()} to {df_raw['Local_time'].max()}")
    print("\n🔧 Creating calculated features...")

    exh_temp_cols = [
    'M_E_NO1_CYL_EXH_GAS_OUT_TEMP',
    'M_E_NO2_CYL_EXH_GAS_OUT_TEMP',
    'M_E_NO3_CYL_EXH_GAS_OUT_TEMP',
    'M_E_NO4_CYL_EXH_GAS_OUT_TEMP',
    'M_E_NO5_CYL_EXH_GAS_OUT_TEMP',
    'M_E_NO6_CYL_EXH_GAS_OUT_TEMP'
    ]
    if all(col in df_raw.columns for col in exh_temp_cols):
        df_raw['M_E_CYL_EXH_GAS_OUT_TEMP'] = df_raw[exh_temp_cols].mean(axis=1)
        print("   ✅ M_E_CYL_EXH_GAS_OUT_TEMP created")

    # 2. Average TC RPM
    tc_rpm_cols = ['ME_NO_1_TC_RPM', 'ME_NO_2_TC_RPM']
    if all(col in df_raw.columns for col in tc_rpm_cols):
        df_raw['ME_TC_RPM'] = df_raw[tc_rpm_cols].mean(axis=1)
        print("   ✅ ME_TC_RPM created")

    # 3. TC RPM / Scav Air Pressure ratio
    if 'ME_SCAV_AIR_PRESS' in df_raw.columns and 'ME_TC_RPM' in df_raw.columns:
        df_raw['ME_SCAV_AIR_PRESS'] = df_raw['ME_SCAV_AIR_PRESS'].replace(0, 0.75)
        df_raw['ME_TC_RPM_SCAV_RATIO'] = df_raw['ME_TC_RPM'] / df_raw['ME_SCAV_AIR_PRESS']
        df_raw['ME_TC_RPM_SCAV_RATIO'] = df_raw['ME_TC_RPM_SCAV_RATIO'].interpolate(method='linear')
        df_raw['ME_TC_RPM_SCAV_RATIO'] = df_raw['ME_TC_RPM_SCAV_RATIO'].bfill().ffill()
        print("   ✅ ME_TC_RPM_SCAV_RATIO created")

    print("✅ Feature engineering complete")

    # print(f"MongoDB columns: {df_raw.columns.tolist()}")
    # print(f"\nFirst row:\n{df_raw.iloc[0]}")
    # if len(df_raw) == 0:
    #     raise ValueError("Collection is empty!")
    # print(f"Sample Local_time values: {df_raw['Local_time'].head()}")
    # print(f"Local_time dtype: {df_raw['Local_time'].dtype}")
    
    # df_raw['Local_time'] = pd.to_datetime(df_raw['Local_time'], errors='coerce')
    # nat_count = df_raw['Local_time'].isna().sum()
    # if nat_count > 0:
    #     print(f"⚠️ Warning: {nat_count} timestamps couldn't be parsed!")
    #     df_raw = df_raw.dropna(subset=['Local_time'])
    # df_raw = df_raw.sort_values('Local_time').reset_index(drop=True)
    # print(f"✅ Fetched {len(df_raw)} per-minute records")
    # print(f"   Date range: {df_raw['Local_time'].min()} to {df_raw['Local_time'].max()}")
    six_hours_minutes = 6 * 60  # 360 minutes
    
    if len(df_raw) < (7*24*60 + six_hours_minutes):  # Need 7 days + 6 hours
        raise ValueError(f"Not enough data! Need ~10,440 minutes, have {len(df_raw)}")
    
    # Split point: 6 hours from the end
    split_point = len(df_raw) - six_hours_minutes
    
    df_input_raw = df_raw.iloc[:split_point].copy()
    df_actual_raw = df_raw.iloc[split_point:].copy()
    
    forecast_start_time = df_actual_raw['Local_time'].iloc[0]
    forecast_end_time = df_actual_raw['Local_time'].iloc[-1]
    
    print(f"\n📊 Data split:")
    print(f"   Input (per-minute): {len(df_input_raw)} rows")
    print(f"   Actual (per-minute): {len(df_actual_raw)} rows")
    print(f"   Forecast window: {forecast_start_time} → {forecast_end_time}")
    # ========================================================================
    # 2. DEFINE TIME WINDOW
    # ========================================================================
    
    # Simulate "current time" as 6 hours ago
    # So we can compare predictions with actual data
##############################################################################    
    # now = datetime.now()
    # forecast_end_time = now - timedelta(hours=6)  # "Now" is 6 hours ago
    # forecast_start_time = forecast_end_time - timedelta(hours=6)  # Predict this 6-hour window
    
    # # Need 7 days of history BEFORE forecast_start_time for input
    # input_start_time = forecast_start_time - timedelta(days=7)
    
    # print(f"\n📅 Time Windows:")
    # print(f"   Input period:    {input_start_time} → {forecast_start_time}")
    # print(f"   Forecast period: {forecast_start_time} → {forecast_end_time}")
    # print(f"   (We'll compare predictions with actual data in this 6-hour window)")
    
    # # ========================================================================
    # # 3. FETCH PER-MINUTE DATA FROM MONGODB
    # # ========================================================================
    
    # # Fetch 7 days + 6 hours of data
    # total_hours = 7 * 24 + 6
    # df_raw = fetch_data_from_mongo(collection, forecast_end_time, total_hours)
    
    # # Ensure Local_time is datetime
    # if 'Local_time' not in df_raw.columns:
    #     raise ValueError("'Local_time' column not found in MongoDB data!")
    
    # df_raw['Local_time'] = pd.to_datetime(df_raw['Local_time'])
    # df_raw = df_raw.sort_values('Local_time').reset_index(drop=True)
    
    # print(f"✅ Raw per-minute data: {len(df_raw)} rows")
    
    # # ========================================================================
    # # 4. SPLIT: INPUT DATA vs ACTUAL DATA (for comparison)
    # # ========================================================================
    
    # # Input data: everything before forecast_start_time
    # df_input_raw = df_raw[df_raw['Local_time'] < forecast_start_time].copy()
    
    # # Actual data: the 6 hours we want to compare against
    # df_actual_raw = df_raw[
    #     (df_raw['Local_time'] >= forecast_start_time) & 
    #     (df_raw['Local_time'] <= forecast_end_time)
    # ].copy()
    
    # print(f"\n📊 Data split:")
    # print(f"   Input (per-minute): {len(df_input_raw)} rows")
    # print(f"   Actual (per-minute): {len(df_actual_raw)} rows")
    
    # ========================================================================
    # 5. AGGREGATE INPUT DATA TO 15-MINUTE
    # ========================================================================
    
    print("\n🔄 Aggregating input data to 15-minute intervals...")
    
    df_input_raw.set_index('Local_time', inplace=True)
    # df_input_15min = df_input_raw.resample('15min').mean().reset_index()
    numeric_cols = df_input_raw.select_dtypes(include=[np.number]).columns.tolist()
    df_input_15min = df_input_raw[numeric_cols].resample('15min').mean().reset_index()
    
    # Create calculated column if needed
    if 'ME_TC_RPM_SCAV_RATIO' not in df_input_15min.columns:
        df_input_15min['ME_SCAV_AIR_PRESS'] = df_input_15min['ME_SCAV_AIR_PRESS'].replace(0, 0.75)
        df_input_15min['ME_TC_RPM_SCAV_RATIO'] = df_input_15min['ME_TC_RPM'] / df_input_15min['ME_SCAV_AIR_PRESS']
        df_input_15min['ME_TC_RPM_SCAV_RATIO'] = df_input_15min['ME_TC_RPM_SCAV_RATIO'].interpolate(method='linear')
        df_input_15min['ME_TC_RPM_SCAV_RATIO'] = df_input_15min['ME_TC_RPM_SCAV_RATIO'].bfill().ffill()
    
    print(f"✅ Aggregated to 15-min: {len(df_input_15min)} rows")
    
    # ========================================================================
    # 6. PREPARE MODEL INPUT (LAST 672 TIMESTEPS)
    # ========================================================================
    
    input_cols = metadata['input_features']
    output_cols = metadata['output_features']
    
    # Check if we have enough data
    if len(df_input_15min) < metadata['lookback']:
        raise ValueError(f"Not enough data! Need {metadata['lookback']} timesteps, have {len(df_input_15min)}")
    
    # Take last 672 timesteps
    df_model_input = df_input_15min.tail(metadata['lookback']).copy()
    
    # Extract features
    X_input = df_model_input[input_cols].values
    
    # Handle NaN
    if np.isnan(X_input).any():
        print("⚠️  Warning: NaN found in input, filling with forward fill...")
        X_input = pd.DataFrame(X_input, columns=input_cols).ffill().bfill().values
    
    # Scale
    X_input_scaled = input_scaler.transform(X_input)
    
    # Reshape for model: (1, 672, 25)
    X_input_tensor = torch.FloatTensor(X_input_scaled).unsqueeze(0)
    
    print(f"✅ Model input prepared: shape {X_input_tensor.shape}")
    
    # ========================================================================
    # 7. LOAD MODEL AND PREDICT
    # ========================================================================
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = LSTMTransformerSeq2Seq(
        num_layers=metadata['best_params']['num_layers'],
        d_model=metadata['best_params']['d_model'],
        num_heads=metadata['best_params']['num_heads'],
        dff=metadata['best_params']['dff'],
        input_dim=metadata['input_dim'],
        output_dim=metadata['output_dim'],
        dropout_rate=metadata['best_params']['dropout']
    ).to(device)
    
    model.load_state_dict(torch.load(f"{model_dir}/model.pth", map_location=device))
    model.eval()
    
    print(f"✅ Model loaded on {device}")
    
    # Predict
    print("\n🔮 Generating predictions...")
    X_input_tensor = X_input_tensor.to(device)
    
    y_pred_scaled = autoregressive_predict(
        model, 
        X_input_tensor, 
        metadata['output_dim'], 
        metadata['horizon'],  # 24 timesteps (6 hours at 15-min)
        device
    )
    
    # Denormalize
    y_pred_scaled_flat = y_pred_scaled.reshape(-1, metadata['output_dim'])
    y_pred_15min = output_scaler.inverse_transform(y_pred_scaled_flat)
    
    print(f"✅ Predictions generated: {y_pred_15min.shape}")
    
    # ========================================================================
    # 8. CREATE 15-MIN PREDICTION DATAFRAME
    # ========================================================================
    
    # Generate timestamps for predictions (24 timesteps at 15-min intervals)
    pred_timestamps = pd.date_range(
        start=forecast_start_time,
        periods=metadata['horizon'],
        freq='15min'
    )
    
    df_pred_15min = pd.DataFrame(
        y_pred_15min,
        columns=output_cols
    )
    df_pred_15min['Local_time'] = pred_timestamps
    
    print(f"✅ 15-min predictions DataFrame created")
    
    # ========================================================================
    # 9. DISAGGREGATE TO PER-MINUTE (LINEAR INTERPOLATION)
    # ========================================================================
    
    print("\n🔄 Disaggregating predictions to per-minute...")
    
    df_pred_15min.set_index('Local_time', inplace=True)
    
    # Resample to 1-minute with linear interpolation
    df_pred_1min = df_pred_15min.resample('1min').interpolate(method='linear').reset_index()
    
    # Trim to exactly the forecast window
    df_pred_1min = df_pred_1min[
        (df_pred_1min['Local_time'] >= forecast_start_time) &
        (df_pred_1min['Local_time'] <= forecast_end_time)
    ]
    
    print(f"✅ Per-minute predictions: {len(df_pred_1min)} rows")
    
    # ========================================================================
    # 10. PREPARE ACTUAL DATA FOR COMPARISON
    # ========================================================================
    
    # Ensure actual data also has calculated column
    if 'ME_TC_RPM_SCAV_RATIO' not in df_actual_raw.columns:
        df_actual_raw['ME_SCAV_AIR_PRESS'] = df_actual_raw['ME_SCAV_AIR_PRESS'].replace(0, 0.75)
        df_actual_raw['ME_TC_RPM_SCAV_RATIO'] = df_actual_raw['ME_TC_RPM'] / df_actual_raw['ME_SCAV_AIR_PRESS']
        df_actual_raw['ME_TC_RPM_SCAV_RATIO'] = df_actual_raw['ME_TC_RPM_SCAV_RATIO'].interpolate(method='linear')
        df_actual_raw['ME_TC_RPM_SCAV_RATIO'] = df_actual_raw['ME_TC_RPM_SCAV_RATIO'].bfill().ffill()
    
    # ========================================================================
    # 11. ALIGN TIMESTAMPS AND CALCULATE METRICS
    # ========================================================================
    
    print("\n📊 Calculating metrics...")
    
    # Merge on timestamp
    # df_comparison = pd.merge(
    #     df_actual_raw[['Local_time'] + output_cols],
    #     df_pred_1min,
    #     on='Local_time',
    #     suffixes=('_actual', '_pred')
    # )
    df_actual_raw['Local_time_rounded'] = df_actual_raw['Local_time'].dt.round('min')
    df_pred_1min['Local_time_rounded'] = df_pred_1min['Local_time'].dt.round('min')
    actual_cols = ['Local_time_rounded'] + output_cols
    pred_cols = ['Local_time_rounded'] + [col for col in output_cols if col in df_pred_1min.columns]

    # Merge on rounded timestamp
    df_comparison = pd.merge(
        df_actual_raw[['Local_time_rounded'] + output_cols],
        df_pred_1min[['Local_time_rounded'] + [col for col in df_pred_1min.columns if col != 'Local_time']],
        on='Local_time_rounded',
        suffixes=('_actual', '_pred')
    )

    # Use rounded time for plotting
    df_comparison.rename(columns={'Local_time_rounded': 'Local_time'}, inplace=True)
    
    print(f"✅ Aligned data: {len(df_comparison)} matching timestamps")
    print("\n🔍 DEBUG INFO:")
    print(f"Actual data date range: {df_actual_raw['Local_time'].min()} to {df_actual_raw['Local_time'].max()}")
    print(f"Predicted data date range: {df_pred_1min['Local_time'].min()} to {df_pred_1min['Local_time'].max()}")
    print(f"\nActual timestamps sample (first 10):")
    print(df_actual_raw['Local_time'].head(10))
    print(f"\nPredicted timestamps sample (first 10):")
    print(df_pred_1min['Local_time'].head(10))
    print(f"\nActual data shape: {df_actual_raw.shape}")
    print(f"Predicted data shape: {df_pred_1min.shape}")
    
    # Calculate metrics per feature
    print("\n" + "="*70)
    print("PREDICTION METRICS (PER-MINUTE COMPARISON)")
    print("="*70)
    
    for feature in output_cols:
        actual_col = f"{feature}_actual"
        pred_col = f"{feature}_pred"
        
        if actual_col not in df_comparison.columns or pred_col not in df_comparison.columns:
            print(f"\n⚠️  Skipping {feature} - columns not found")
            continue
        
        actual = df_comparison[actual_col].values
        pred = df_comparison[pred_col].values
        
        # Remove NaN
        mask = ~(np.isnan(actual) | np.isnan(pred))
        actual = actual[mask]
        pred = pred[mask]
        
        if len(actual) == 0:
            print(f"\n⚠️  {feature}: No valid data after removing NaN")
            continue
        
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        
        mean_actual = np.mean(np.abs(actual))
        rmse_percent = (rmse / mean_actual * 100) if mean_actual > 0 else 0
        
        print(f"\n📈 {feature}:")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE:  {mae:.4f}")
        print(f"   R²:   {r2:.4f}")
        print(f"   RMSE %: {rmse_percent:.2f}%")
        print(f"   Valid points: {len(actual)}")
    
    # ========================================================================
    # 12. VISUALIZATION
    # ========================================================================
    
    print("\n📊 Creating visualizations...")
    
    n_features = len(output_cols)
    fig, axes = plt.subplots(n_features, 1, figsize=(16, 4*n_features))
    
    if n_features == 1:
        axes = [axes]
    
    for i, feature in enumerate(output_cols):
        actual_col = f"{feature}_actual"
        pred_col = f"{feature}_pred"
        
        if actual_col not in df_comparison.columns:
            continue
        
        axes[i].plot(df_comparison['Local_time'], df_comparison[actual_col], 
                     label='Actual', alpha=0.8, linewidth=1.5, color='blue')
        axes[i].plot(df_comparison['Local_time'], df_comparison[pred_col], 
                     label='Predicted', alpha=0.8, linewidth=1.5, color='red', linestyle='--')
        
        axes[i].set_title(f'{feature} - Actual vs Predicted (Per-Minute)', 
                         fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Time', fontsize=12)
        axes[i].set_ylabel('Value', fontsize=12)
        axes[i].legend(fontsize=11)
        axes[i].grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('real_time_test_predictions.png', dpi=150, bbox_inches='tight')
    print("✅ Visualization saved to 'real_time_test_predictions.png'")
    
    print("\n" + "="*70)
    print("🎉 TESTING COMPLETE!")
    print("="*70)
    print("\n💡 Review the plot to see how well predictions match actual data!")
    print("   Blue = Actual per-minute data from DB")
    print("   Red = Model predictions (disaggregated to per-minute)")
    
    return df_comparison

# ============================================================================
# RUN IT!
# ============================================================================

if __name__ == "__main__":
    try:
        df_results = main()
        print("\n✅ Results saved in 'df_results' DataFrame")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()