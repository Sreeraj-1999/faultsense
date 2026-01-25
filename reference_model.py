import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import pickle
import json
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# SIMPLE FFN MODEL
# ============================================================================

class SimpleFeedForwardNetwork(nn.Module):
    """Simple Feed-Forward Network for EFD prediction"""
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.3):
        super(SimpleFeedForwardNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# ============================================================================
# DATASET
# ============================================================================

class CleanDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_ffn_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                   epochs, patience, device):
    """Training loop with early stopping"""
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    train_losses = []
    val_losses = []
    
    logger.info(f"🏋️  Starting FFN training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"⚠️  NaN/Inf loss detected at epoch {epoch+1}, skipping batch")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                output = model(batch_X)
                loss = criterion(output, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        logger.info(f"   Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"   ⏹️  Early stopping at epoch {epoch+1}")
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses, best_val_loss

# ============================================================================
# OPTUNA OBJECTIVE
# ============================================================================

def ffn_objective(trial, train_data, val_data, input_dim, output_dim, device):
    """Optuna objective for FFN hyperparameter tuning"""
    num_layers = trial.suggest_int('num_layers', 1, 3)
    
    hidden_dims = []
    for i in range(num_layers):
        hidden_dim = trial.suggest_categorical(f'hidden_dim_{i}', [32, 64, 128, 256])
        hidden_dims.append(hidden_dim)
    
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    model = SimpleFeedForwardNetwork(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        dropout_rate=dropout
    ).to(device)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    _, _, _, best_val_loss = train_ffn_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        epochs=10, patience=10, device=device
    )
    
    return best_val_loss

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def train_clean_data_ffn(all_actual_data, anomaly_flags, input_feature_names, 
                        output_feature_names, all_feature_names, timestamps_clean, save_dir, device):
    """
    Train FFN on clean data
    
    Args:
        all_actual_data: numpy array (num_sequences, num_features) - UNSCALED, original values
        anomaly_flags: list of bool (num_sequences,) - True = anomaly, False = clean
        input_feature_names: list of str - names of input features (Load + Important)
        output_feature_names: list of str - names of output features (EFDs)
        save_dir: str - directory to save model
        device: torch device
    """
    logger.info("=" * 80)
    logger.info("🧹 CLEAN DATA FFN TRAINING")
    logger.info("=" * 80)
    
    # Step 1: Filter clean data
    logger.info("📋 Step 1: Filtering clean data...")
    clean_mask = ~np.array(anomaly_flags)
    clean_data = all_actual_data[clean_mask]
    
    num_total = len(anomaly_flags)
    num_clean = len(clean_data)
    num_anomalies = num_total - num_clean
    
    logger.info(f"   Total sequences: {num_total}")
    logger.info(f"   Clean sequences: {num_clean} ({num_clean/num_total*100:.1f}%)")
    logger.info(f"   Anomalies removed: {num_anomalies} ({num_anomalies/num_total*100:.1f}%)")
    
    if num_clean < 10:
        raise ValueError(f"Insufficient clean data: {num_clean} sequences")
    
    # Step 2: Split into input and output
    logger.info("📊 Step 2: Preparing input/output data...")
    
    # Get column indices
    input_indices = [all_feature_names.index(name) for name in input_feature_names]
    output_indices = [all_feature_names.index(name) for name in output_feature_names]
    
    X_clean = clean_data[:, input_indices]
    y_clean = clean_data[:, output_indices]
    
    logger.info(f"   Input features: {len(input_feature_names)} - {input_feature_names}")
    logger.info(f"   Output features: {len(output_feature_names)} - {output_feature_names}")
    logger.info(f"   X shape: {X_clean.shape}")
    logger.info(f"   y shape: {y_clean.shape}")
    
    # Step 3: Save to Excel (UNSCALED data for verification)
    logger.info("💾 Step 3: Saving clean data to Excel...")
    
    excel_data = pd.DataFrame(
        clean_data,
        columns=all_feature_names
    )
    
    excel_path = os.path.join(save_dir, "clean_data_unscaled.xlsx")
    excel_data.to_excel(excel_path, index=False)
    
    logger.info(f"   ✅ Excel saved: {excel_path}")
    logger.info(f"   Rows: {len(excel_data)}, Columns: {len(all_feature_names)}")
    
    # Step 4: Train/Val split
    logger.info("🔀 Step 4: Splitting train/validation (80/20)...")
    
    split_idx = int(0.8 * len(X_clean))
    X_train, X_val = X_clean[:split_idx], X_clean[split_idx:]
    y_train, y_val = y_clean[:split_idx], y_clean[split_idx:]
    
    logger.info(f"   Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Step 5: Scale data
    logger.info("📊 Step 5: Scaling data...")
    
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()
    
    X_train_scaled = input_scaler.fit_transform(X_train)
    X_val_scaled = input_scaler.transform(X_val)
    y_train_scaled = output_scaler.fit_transform(y_train)
    y_val_scaled = output_scaler.transform(y_val)
    
    logger.info("   ✅ Scaling complete")
    
    # Step 6: Create datasets
    train_dataset = CleanDataset(X_train_scaled, y_train_scaled)
    val_dataset = CleanDataset(X_val_scaled, y_val_scaled)
    
    # Step 7: Optuna tuning
    logger.info("🔍 Step 6: Optuna hyperparameter tuning (10 trials)...")
    
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: ffn_objective(trial, train_dataset, val_dataset, 
                                    X_train.shape[1], y_train.shape[1], device),
        n_trials=10,
        show_progress_bar=True
    )
    
    best_params = study.best_params
    logger.info(f"✅ Best hyperparameters:")
    for key, value in best_params.items():
        logger.info(f"   {key}: {value}")
    
    # Step 8: Train final model
    logger.info("🏋️  Step 7: Training final FFN model...")
    
    num_layers = best_params['num_layers']
    hidden_dims = [best_params[f'hidden_dim_{i}'] for i in range(num_layers)]
    
    final_model = SimpleFeedForwardNetwork(
        input_dim=X_train.shape[1],
        hidden_dims=hidden_dims,
        output_dim=y_train.shape[1],
        dropout_rate=best_params['dropout']
    ).to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(final_model.parameters(), lr=best_params['lr'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    final_model, train_losses, val_losses, best_val_loss = train_ffn_model(
        final_model, train_loader, val_loader, criterion, optimizer, scheduler,
        epochs=50, patience=15, device=device
    )
    
    logger.info(f"✅ Training complete! Best val loss: {best_val_loss:.6f}")
    
    # Step 9: Calculate metrics and generate plot_data
    logger.info("📊 Step 8: Calculating validation metrics and generating plot_data...")
    
    final_model.eval()
    with torch.no_grad():
        y_val_pred_scaled = final_model(torch.FloatTensor(X_val_scaled).to(device)).cpu().numpy()
    
    y_val_pred = output_scaler.inverse_transform(y_val_pred_scaled)
    
    
    # Split timestamps same as data split (80/20)
    split_idx = int(0.8 * len(timestamps_clean))
    timestamps_train_ffn = timestamps_clean[:split_idx]
    timestamps_val_ffn = timestamps_clean[split_idx:]
    
    logger.info(f"   Timestamps split: Train {len(timestamps_train_ffn)}, Val {len(timestamps_val_ffn)}")
    
    results = []
    plot_data = []
    
    for i, feature_name in enumerate(output_feature_names):
        actual = y_val[:, i]
        predicted = y_val_pred[:, i]
        
        # rmse = np.sqrt(mean_squared_error(actual, predicted))
        # mae = mean_absolute_error(actual, predicted)
        # r2 = r2_score(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)

        # Convert to percentages
        mean_actual = np.mean(np.abs(actual))
        mae_pct = (mae / mean_actual) * 100 if mean_actual != 0 else 0
        rmse_pct = (rmse / mean_actual) * 100 if mean_actual != 0 else 0
        r2_pct = r2 * 100
        
        logger.info(f"   {feature_name}:")
        logger.info(f"      RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        # results.append({
        #     "feature_name": feature_name,
        #     "rmse": float(rmse),
        #     "mae": float(mae),
        #     "r2": float(r2)
        # })
        results.append({
            "feature_name": feature_name,
            "rmse": round(rmse_pct, 1),
            "mae": round(mae_pct, 1),
            "r2": round(r2_pct, 1)
        })
        
        # Create plot_data entry (same structure as ts_response)
        plot_data.append({
            "feature_name": feature_name,
            "actual": [round(float(x), 2) for x in actual],
            "predicted": [round(float(x), 2) for x in predicted],
            "timestamps": [str(ts) for ts in timestamps_val_ffn],  # Real timestamps
            # "metrics": {
            #     "rmse": round(float(rmse), 2),
            #     "mae": round(float(mae), 2),
            #     "r2": round(float(r2), 4)
            # }
            "metrics": {
            "rmse": round(rmse_pct, 1),
            "mae": round(mae_pct, 1),
            "r2": round(r2_pct, 1)
}
        })
    
    # Step 10: Save model
    logger.info("💾 Step 9: Saving FFN model...")
    
    ffn_dir = os.path.join(save_dir, "ffn_clean")
    os.makedirs(ffn_dir, exist_ok=True)
    
    model_path = os.path.join(ffn_dir, "ffn_model.pth")
    input_scaler_path = os.path.join(ffn_dir, "input_scaler.pkl")
    output_scaler_path = os.path.join(ffn_dir, "output_scaler.pkl")
    metadata_path = os.path.join(ffn_dir, "metadata.json")
    
    torch.save(final_model.state_dict(), model_path)
    
    with open(input_scaler_path, 'wb') as f:
        pickle.dump(input_scaler, f)
    
    with open(output_scaler_path, 'wb') as f:
        pickle.dump(output_scaler, f)
    
    metadata = {
        "input_features": input_feature_names,
        "output_features": output_feature_names,
        "input_dim": int(X_train.shape[1]),
        "output_dim": int(y_train.shape[1]),
        "best_params": best_params,
        "best_val_loss": float(best_val_loss),
        "num_clean_sequences": int(num_clean),
        "num_anomalies_removed": int(num_anomalies),
        "train_sequences": int(len(X_train)),
        "val_sequences": int(len(X_val)),
        "training_date": str(datetime.now()),
        "metrics": results
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"   ✅ Model saved: {model_path}")
    logger.info(f"   ✅ Scalers saved")
    logger.info(f"   ✅ Metadata saved: {metadata_path}")
    
    logger.info("=" * 80)
    logger.info("🎉 CLEAN DATA FFN TRAINING COMPLETED!")
    logger.info("=" * 80)
    
    return {
        "model_path": model_path,
        "excel_path": excel_path,
        "best_val_loss": float(best_val_loss),
        "metrics": results,
        "num_clean_sequences": int(num_clean),
        "plot_data": plot_data  # ← ADD THIS
    }