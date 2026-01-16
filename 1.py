import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import optuna
import logging
import json
import pickle
from datetime import datetime
import math
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# POSITIONAL ENCODING
# ============================================================================

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

# ============================================================================
# LSTM-TRANSFORMER ENCODER LAYER
# ============================================================================

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

# ============================================================================
# LSTM-TRANSFORMER DECODER LAYER
# ============================================================================

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

# ============================================================================
# FULL ENCODER
# ============================================================================

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

# ============================================================================
# FULL DECODER
# ============================================================================

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

# ============================================================================
# COMPLETE MODEL
# ============================================================================

class LSTMTransformerSeq2Seq(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, input_dim, output_dim, dropout_rate=0.1):
        super(LSTMTransformerSeq2Seq, self).__init__()
        self.encoder = LSTMTransformerEncoder(num_layers, d_model, num_heads, dff, input_dim, dropout_rate)
        self.decoder = LSTMTransformerDecoder(num_layers, d_model, num_heads, dff, output_dim, dropout_rate)
    
    def forward(self, src, tgt):
        enc_output = self.encoder(src)
        dec_output = self.decoder(tgt, enc_output)
        return dec_output

# ============================================================================
# DATASET
# ============================================================================

class TimeSeriesDataset(Dataset):
    def __init__(self, input_data, output_data, timestamps):
        self.input_data = torch.FloatTensor(input_data)
        self.output_data = torch.FloatTensor(output_data)
        self.timestamps = [str(ts) for ts in timestamps]
    
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        return self.input_data[idx], self.output_data[idx], idx  # Return index instead of timestamp

# ============================================================================
# SEQUENCE CREATION WITH VESSEL BOUNDARY CHECKING
# ============================================================================

def create_sequences(df, input_cols, output_cols, timestamp_col, vessel_boundaries, lookback=672, horizon=672, step=96):
    """Create sliding window sequences with vessel boundary checking"""
    input_sequences = []
    output_sequences = []
    timestamps = []
    
    # Create boundary list: [0, 4678, 10550, 17265]
    boundaries = [0] + vessel_boundaries + [len(df)]
    
    logger.info(f"Creating sequences with boundaries: {boundaries}")
    
    skipped_count = 0
    for i in range(0, len(df) - lookback - horizon + 1, step):
        sequence_start = i
        sequence_end = i + lookback + horizon
        
        # Check if sequence crosses ANY vessel boundary
        crosses_boundary = False
        for boundary in boundaries[1:-1]:  # Skip first (0) and last (total length)
            if sequence_start < boundary < sequence_end:
                crosses_boundary = True
                skipped_count += 1
                break
        
        if crosses_boundary:
            continue  # Skip sequences that cross vessel boundaries
        
        # Input: lookback timesteps
        input_seq = df[input_cols].iloc[i:i+lookback].values
        
        # Output: next horizon timesteps
        output_seq = df[output_cols].iloc[i+lookback:i+lookback+horizon].values
        
        # Timestamp of the last input point
        timestamp = df[timestamp_col].iloc[i+lookback-1]
        
        input_sequences.append(input_seq)
        output_sequences.append(output_seq)
        timestamps.append(timestamp)
    
    logger.info(f"Skipped {skipped_count} sequences due to vessel boundary crossing")
    
    return np.array(input_sequences), np.array(output_sequences), timestamps

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, patience, device):
    """Training loop with early stopping"""
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for src, tgt, _ in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            
            # Teacher forcing: use actual target as decoder input (shifted)
            tgt_input = torch.zeros_like(tgt)
            tgt_input[:, 1:, :] = tgt[:, :-1, :]
            
            optimizer.zero_grad()
            output = model(src, tgt_input)
            loss = criterion(output, tgt)
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss detected, skipping batch")
                continue
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src, tgt, _ in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_input = torch.zeros_like(tgt)
                tgt_input[:, 1:, :] = tgt[:, :-1, :]
                
                output = model(src, tgt_input)
                loss = criterion(output, tgt)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses, best_val_loss

# ============================================================================
# OPTUNA OBJECTIVE
# ============================================================================

# def objective(trial, train_data, val_data, input_dim, output_dim, device):
#     """Optuna objective function for hyperparameter tuning"""
    
#     # Hyperparameters to tune
#     d_model = trial.suggest_categorical('d_model', [64, 128, 256])
#     num_heads = trial.suggest_categorical('num_heads', [4, 8])
#     num_layers = trial.suggest_int('num_layers', 2, 4)
#     dff = trial.suggest_categorical('dff', [128, 256, 512])
#     dropout = trial.suggest_float('dropout', 0.3, 0.5)
#     # lr = trial.suggest_loguniform('lr', 1e-4, 1e-3)
#     lr = trial.suggest_float('lr', 1e-5, 1e-4, log=True)  # Much lower learning rate
#     batch_size = trial.suggest_categorical('batch_size', [8, 16])
def objective(trial, train_data, val_data, input_dim, output_dim, device):
    d_model = trial.suggest_categorical('d_model', [8, 16])  # Your idea - very tiny!
    num_heads = trial.suggest_categorical('num_heads', [2])  # Keep at 2 (minimum for MHA)
    num_layers = trial.suggest_int('num_layers', 1, 2)  # Allow 1 or 2 layers
    dff = trial.suggest_categorical('dff', [16, 32])  # Your idea - very small!
    dropout = trial.suggest_float('dropout', 0.6, 0.7)  # High dropout
    lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16])  # Fixed
    
    # Create model
    model = LSTMTransformerSeq2Seq(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_dim=input_dim,
        output_dim=output_dim,
        dropout_rate=dropout
    ).to(device)
    
    # Data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    # Loss and optimizer
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Train
    _, _, _, best_val_loss = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        epochs=30,  # Reduced for tuning
        patience=8,
        device=device
    )
    
    return best_val_loss

# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    logger.info("🚀 Starting Time Series Model Training...")
    
    # ========================================================================
    # 1. LOAD COMBINED VESSEL DATA
    # ========================================================================
    csv_path = r"C:\Users\User\Desktop\siemens\freya_schulte\training_data_averaged.csv"
    logger.info(f"📊 Reading data from {csv_path}")
    
    df = pd.read_csv(csv_path)
    logger.info(f"✅ Data loaded: {df.shape}")
    df = df.sort_values(['vessel_id', 'Local_time']).reset_index(drop=True)
    logger.info(f"✅ Data sorted by vessel_id and timestamp")
    
    # Check for required columns
    if 'vessel_id' not in df.columns:
        raise ValueError("Missing 'vessel_id' column in data")
    
    if 'Local_time' not in df.columns:
        raise ValueError("Missing 'Local_time' column in data")
    
    # Convert timestamp
    df['Local_time'] = pd.to_datetime(df['Local_time'], format='mixed', dayfirst=True)
    # df = df.sort_values('Local_time').reset_index(drop=True)
    
    # ========================================================================
    # 2. GET VESSEL BOUNDARIES
    # ========================================================================
    vessel_groups = df.groupby('vessel_id').groups
    vessel_boundaries = []
    for vessel_id in sorted(vessel_groups.keys())[1:]:  # Skip first vessel
        vessel_boundaries.append(vessel_groups[vessel_id][0])
    vessel_boundaries = [b for b in vessel_boundaries if b > 0]
    vessel_boundaries = sorted(vessel_boundaries)  # Ensure sorted order    
    
    logger.info(f"🚢 Vessel boundaries: {vessel_boundaries}")
    
    # ========================================================================
    # 3. DEFINE INPUT AND OUTPUT FEATURES
    # ========================================================================
    load_feature = 'ME_Load@AVG'
    
    efd_features = [
        'M_E_CYL_EXH_GAS_OUT_TEMP',
        'ME_SCAV_AIR_PRESS',
        'ME_TC_RPM',
        'ME_TC_RPM_SCAV_RATIO'
    ]
    
    important_features = [
        'SHAFT_POWER', 'SHAFT_TORQUE', 'ME_RPM', 'PROPULSION_EFFICIENCY',
        'ME_NO_1_TC_LO_OUT_TEMP', 'ME_NO_2_TC_LO_OUT_TEMP',
        'ME_NO1_CYL_LINER_PP_SIDE_TEMP', 'ME_NO2_CYL_LINER_EXH_SIDE_TEMP',
        'ME_NO2_CYL_LINER_PP_SIDE_TEMP', 'ME_NO3_CYL_LINER_PP_SIDE_TEMP',
        'ME_NO5_CYL_LINER_EXH_SIDE_TEMP', 'ME_NO6_CYL_LINER_PP_SIDE_TEMP',
        'M_E_NO1_CYL_PCO_OUT_TEMP', 'M_E_NO3_CYL_PCO_OUT_TEMP',
        'M_E_NO5_CYL_PCO_OUT_TEMP', 'M_E_THRUST_BEARING_TEMP',
        'ME_JACKET_CW_INLET_TEMP', 'SA_POW_act_kW@AVG',
        'SA_TQU_act_kNm@AVG', 'ME_HFO_FMS_act_kgPh@AVG'
    ]
    
    # Combine
    input_cols = [load_feature] + efd_features + important_features
    output_cols = [load_feature] + efd_features
    
    logger.info(f"📋 Input features: {len(input_cols)} total")
    logger.info(f"📋 Output features: {len(output_cols)} total")
    if 'ME_TC_RPM_SCAV_RATIO' not in df.columns:
        if 'ME_TC_RPM' in df.columns and 'ME_SCAV_AIR_PRESS' in df.columns:
            logger.info("📊 Creating calculated column: ME_TC_RPM_SCAV_RATIO = ME_TC_RPM / ME_SCAV_AIR_PRESS")
            # df['ME_SCAV_AIR_PRESS'] = df['ME_SCAV_AIR_PRESS'].replace(0, np.nan)
            df['ME_SCAV_AIR_PRESS'] = df['ME_SCAV_AIR_PRESS'].replace(0, 0.75)
            df['ME_TC_RPM_SCAV_RATIO'] = df['ME_TC_RPM'] / df['ME_SCAV_AIR_PRESS']
            df['ME_TC_RPM_SCAV_RATIO'] = df['ME_TC_RPM_SCAV_RATIO'].interpolate(method='linear')
            df['ME_TC_RPM_SCAV_RATIO'] = df['ME_TC_RPM_SCAV_RATIO'].fillna(method='bfill').fillna(method='ffill')
            nan_count = df['ME_TC_RPM_SCAV_RATIO'].isna().sum()
            logger.info(f"✅ Column created successfully (filled {nan_count} NaN values)")
        else:
            raise ValueError("Cannot create ME_TC_RPM_SCAV_RATIO: Missing ME_TC_RPM or ME_SCAV_AIR_PRESS")
    
    # Check if all columns exist
    missing_cols = [col for col in input_cols + output_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    # ========================================================================
    # 4. CREATE SEQUENCES WITH VESSEL BOUNDARY CHECKING
    # ========================================================================
    timestamp_col = 'Local_time'
    
    logger.info("🔄 Creating sequences with vessel boundary checking...")
    X, y, timestamps = create_sequences(
        df, input_cols, output_cols, timestamp_col,
        vessel_boundaries=vessel_boundaries,
        lookback=672,   # 7 days * 24 hours * 4 (15-min intervals)
        horizon=288,    # 7 days forecast         #672
        step=96         # 1 day step (24 hours * 4)
    )
    logger.info(f"✅ Created {len(X)} sequences (after boundary filtering)")
    # 5. CHECK AND CLEAN DATA BEFORE SEQUENCES
# ========================================================================
    logger.info("🔍 Checking for NaN in raw data...")

    # Check which columns have NaN
    nan_cols = []
    for col in input_cols + output_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            nan_cols.append((col, nan_count))
            logger.warning(f"Column '{col}' has {nan_count} NaN values ({nan_count/len(df)*100:.2f}%)")

    if nan_cols:
        logger.warning(f"⚠️  Found NaN in {len(nan_cols)} columns")
        
        # Drop rows with NaN in ANY of the required columns
        df_before = len(df)
        df = df.dropna(subset=input_cols + output_cols)
        df_after = len(df)
        
        logger.warning(f"⚠️  Dropped {df_before - df_after} rows with NaN")
        logger.info(f"✅ Clean data: {df_after} rows remaining")
        
        # Recreate sequences with clean data
        logger.info("🔄 Recreating sequences with clean data...")
        X, y, timestamps = create_sequences(
            df, input_cols, output_cols, timestamp_col,
            vessel_boundaries=vessel_boundaries,
            lookback=672,
            horizon=288,
            step=96
        )
        logger.info(f"✅ Created {len(X)} sequences (after cleaning)")
    else:
        logger.info("✅ No NaN found in data")

    if len(X) < 10:
        raise ValueError(f"Insufficient sequences for training: {len(X)} < 10")
    
    # ========================================================================
    # 5. TRAIN-VAL SPLIT (80-20)
    # ========================================================================
    logger.info("🔀 Shuffling sequences to ensure vessel mix in train/val...")

    
    indices = np.arange(len(X))
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    timestamps = [timestamps[i] for i in indices]

    logger.info("✅ Sequences shuffled with seed=42")

    # Now do the 80-20 split
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    timestamps_train = timestamps[:split_idx]
    timestamps_val = timestamps[split_idx:]
    
    logger.info(f"✅ Train sequences: {len(X_train)}, Validation sequences: {len(X_val)}")
    
    # ========================================================================
    # 6. SCALE DATA (FIT ON TRAIN ONLY)
    # ========================================================================
    logger.info("📊 Scaling data (fit on train only)...")
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()
    
    # Reshape for scaling
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
    y_train_reshaped = y_train.reshape(-1, y_train.shape[-1])
    y_val_reshaped = y_val.reshape(-1, y_val.shape[-1])
    
    # FIT on train, TRANSFORM both
    X_train_scaled = input_scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_val_scaled = input_scaler.transform(X_val_reshaped).reshape(X_val.shape)
    
    y_train_scaled = output_scaler.fit_transform(y_train_reshaped).reshape(y_train.shape)
    y_val_scaled = output_scaler.transform(y_val_reshaped).reshape(y_val.shape)
    
    logger.info("✅ Scaling complete (no data leakage)")
    if np.isnan(X_train_scaled).any() or np.isinf(X_train_scaled).any():
        logger.error("❌ NaN/Inf found in scaled training input data!")
        nan_cols = np.where(np.isnan(X_train_scaled.reshape(-1, X_train_scaled.shape[-1])).any(axis=0))[0]
        logger.error(f"Problematic input columns (indices): {nan_cols}")
        raise ValueError("Data contains NaN/Inf after scaling")
    
    if np.isnan(y_train_scaled).any() or np.isinf(y_train_scaled).any():
        logger.error("❌ NaN/Inf found in scaled training output data!")
        nan_cols = np.where(np.isnan(y_train_scaled.reshape(-1, y_train_scaled.shape[-1])).any(axis=0))[0]
        logger.error(f"Problematic output columns (indices): {nan_cols}")
        raise ValueError("Data contains NaN/Inf after scaling")

    logger.info("✅ No NaN/Inf in scaled data")
    
    # ========================================================================
    # 7. CREATE DATASETS
    # ========================================================================
    train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled, timestamps_train)
    val_dataset = TimeSeriesDataset(X_val_scaled, y_val_scaled, timestamps_val)
    
    # ========================================================================
    # 8. DEVICE SETUP
    # ========================================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️  Using device: {device}")
    
    # ========================================================================
    # 9. HYPERPARAMETER TUNING WITH OPTUNA
    # ========================================================================
    logger.info("🔍 Starting hyperparameter tuning (30 trials)...")
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(trial, train_dataset, val_dataset, X_train.shape[-1], y_train.shape[-1], device),
        n_trials=30,
        show_progress_bar=True
    )
    
    best_params = study.best_params
    logger.info(f"✅ Best hyperparameters: {best_params}")
    
    # ========================================================================
    # 10. TRAIN FINAL MODEL WITH BEST PARAMS
    # ========================================================================
    logger.info("🏋️ Training final model with best hyperparameters...")
    final_model = LSTMTransformerSeq2Seq(
        num_layers=best_params['num_layers'],
        d_model=best_params['d_model'],
        num_heads=best_params['num_heads'],
        dff=best_params['dff'],
        input_dim=X_train.shape[-1],
        output_dim=y_train.shape[-1],
        dropout_rate=best_params['dropout']
    ).to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)
    
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(final_model.parameters(), lr=best_params['lr'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    final_model, train_losses, val_losses, best_val_loss = train_model(
        final_model, train_loader, val_loader, criterion, optimizer, scheduler,
        epochs=200,
        patience=15,
        device=device
    )
    
    # ========================================================================
    # 11. SAVE MODEL AND SCALERS
    # ========================================================================
    model_dir = "models/combined_vessels"
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = f"{model_dir}/model.pth"
    input_scaler_path = f"{model_dir}/input_scaler.pkl"
    output_scaler_path = f"{model_dir}/output_scaler.pkl"
    metadata_path = f"{model_dir}/metadata.json"
    
    torch.save(final_model.state_dict(), model_path)
    
    with open(input_scaler_path, 'wb') as f:
        pickle.dump(input_scaler, f)
    
    with open(output_scaler_path, 'wb') as f:
        pickle.dump(output_scaler, f)
    
    metadata = {
        "vessels": "Combined: Clemens (9665671), Charlotte (9665657), Christa (9665669)",
        "data_split_method": "random_shuffle",  # ADD THIS LINE
        "random_seed": 42,                      # ADD THIS LINE
        "temporal_ordering": "shuffled (not chronological)",  # ADD THIS LINE
        "input_features": input_cols,
        "output_features": output_cols,
        "input_dim": int(X_train.shape[-1]),
        "output_dim": int(y_train.shape[-1]),
        "lookback": 672,
        "horizon": 288,       #672
        "step": 96,
        "best_params": best_params,
        "best_val_loss": float(best_val_loss),
        "final_train_loss": float(train_losses[-1]),
        "final_val_loss": float(val_losses[-1]),
        "num_epochs": len(train_losses),
        "training_date": str(datetime.now()),
        "num_sequences": int(len(X)),
        "train_sequences": int(len(X_train)),
        "val_sequences": int(len(X_val))
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✅ Model saved to {model_path}")
    logger.info(f"✅ Input scaler saved to {input_scaler_path}")
    logger.info(f"✅ Output scaler saved to {output_scaler_path}")
    logger.info(f"✅ Metadata saved to {metadata_path}")
    logger.info("🎉 Training completed successfully!")
    
    # ========================================================================
    # 12. FINAL REPORT
    # ========================================================================
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Total sequences created:       {len(X)}")
    print(f"Training sequences:            {len(X_train)}")
    print(f"Validation sequences:          {len(X_val)}")
    print(f"Input features:                {len(input_cols)}")
    print(f"Output features:               {len(output_cols)}")
    print(f"Best validation loss:          {best_val_loss:.6f}")
    print(f"Final training loss:           {train_losses[-1]:.6f}")
    print(f"Final validation loss:         {val_losses[-1]:.6f}")
    print(f"Number of epochs:              {len(train_losses)}")
    print(f"Model saved to:                {model_path}")
    print("="*70)

if __name__ == "__main__":
    main()