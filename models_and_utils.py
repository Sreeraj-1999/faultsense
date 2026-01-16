import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import logging
from torch.utils.data import Dataset,DataLoader
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# MASK GENERATION FUNCTION
# ============================================================================

def generate_square_subsequent_mask(sz, device):
    """
    Generate causal mask to prevent decoder from seeing future positions.
    Returns a mask where future positions are -inf (will become 0 after softmax).
    """
    mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

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
    
    def forward(self, src, tgt,tgt_mask=None):           #added tgt_mask=None
        enc_output = self.encoder(src)
        dec_output = self.decoder(tgt, enc_output,tgt_mask=tgt_mask)
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

def create_sequences(df, input_cols, output_cols, timestamp_col, vessel_boundaries, lookback, horizon, step):
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
            # tgt_input = torch.zeros_like(tgt)
            # tgt_input[:, 1:, :] = tgt[:, :-1, :]  commented
            batch_size = src.size(0)
            horizon = tgt.size(1)
            output_dim = tgt.size(2)
            
            optimizer.zero_grad()
            enc_output = model.encoder(src)
            tgt_input = torch.zeros(batch_size, horizon, output_dim).to(device)
            tgt_input[:, 0:1, :] = src[:, -1:, :output_dim]
            for t in range(1, horizon):
                # Create causal mask for current length
                mask = generate_square_subsequent_mask(t, device)
                dec_output = model.decoder(tgt_input[:, :t, :], enc_output, tgt_mask=mask)
                # Use prediction for next step (detach to save memory)
                tgt_input[:, t:t+1, :] = dec_output[:, -1:, :].detach()
            mask = generate_square_subsequent_mask(horizon, device)
            output = model.decoder(tgt_input, enc_output, tgt_mask=mask)    
            # output = model(src, tgt_input)
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
                # tgt_input = torch.zeros_like(tgt)
                # tgt_input[:, 1:, :] = tgt[:, :-1, :]
                batch_size = src.size(0)
                horizon = tgt.size(1)
                output_dim = tgt.size(2)
                enc_output = model.encoder(src)
                
                # output = model(src, tgt_input)
                tgt_input = torch.zeros(batch_size, horizon, output_dim).to(device)
                tgt_input[:, 0:1, :] = src[:, -1:, :output_dim]

                for t in range(1, horizon):
                    mask = generate_square_subsequent_mask(t, device)
                    dec_output = model.decoder(tgt_input[:, :t, :], enc_output, tgt_mask=mask)
                    tgt_input[:, t:t+1, :] = dec_output[:, -1:, :]
                mask = generate_square_subsequent_mask(horizon, device)
                output = model.decoder(tgt_input, enc_output, tgt_mask=mask)    

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


def objective(trial, train_data, val_data, input_dim, output_dim, device):
    d_model = trial.suggest_categorical('d_model', [8, 16,32])  # Your idea - very tiny!
    
    num_heads = trial.suggest_categorical('num_heads', [2,4, 8])    

    num_layers = trial.suggest_int('num_layers', 1, 3)  # Allow 1 or 2 layers
    dff = trial.suggest_categorical('dff', [16, 32,64,128])  # Your idea - very small!
    dropout = trial.suggest_float('dropout', 0.3, 0.7)  # High dropout
    lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8,16,32])  # Fixed
    
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
        epochs=10,  # Reduced for tuning
        patience=15,
        device=device
    )
    
    return best_val_loss