import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import pickle
import os
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# DENSE AUTOENCODER MODEL
# ============================================================================

class DenseAutoencoder(nn.Module):
    """
    Dense (Fully Connected) Autoencoder for anomaly detection.
    Flattens sequence data and reconstructs it.
    """
    def __init__(self, input_dim, encoder_dims, latent_dim, dropout_rate=0.2):
        """
        Args:
            input_dim: Flattened input size (lookback * num_features)
            encoder_dims: List of hidden layer sizes for encoder [2048, 512]
            latent_dim: Bottleneck dimension (compressed representation)
            dropout_rate: Dropout probability
        """
        super(DenseAutoencoder, self).__init__()
        
        # Build Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in encoder_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Bottleneck
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build Decoder (mirror of encoder)
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(encoder_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer (no activation - regression task)
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        """
        Forward pass: encode then decode
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """
        Get latent representation only
        """
        return self.encoder(x)

# ============================================================================
# DATASET FOR AUTOENCODER
# ============================================================================

class AutoencoderDataset(Dataset):
    """
    Dataset for autoencoder - input and target are the same (reconstruction)
    """
    def __init__(self, data):
        """
        Args:
            data: numpy array of shape (num_sequences, flattened_dim)
        """
        self.data = torch.FloatTensor(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # For autoencoder, input = target (reconstruction task)
        return self.data[idx], self.data[idx]

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_autoencoder_model(model, train_loader, val_loader, criterion, optimizer, 
                           scheduler, epochs, patience, device):
    """
    Training loop with early stopping for autoencoder
    
    Returns:
        model, train_losses, val_losses, best_val_loss
    """
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    train_losses = []
    val_losses = []
    
    logger.info(f"🏋️  Starting autoencoder training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_input, batch_target in train_loader:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed = model(batch_input)
            loss = criterion(reconstructed, batch_target)
            
            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"⚠️  NaN/Inf loss detected at epoch {epoch+1}, skipping batch")
                continue
            
            # Backward pass
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
            for batch_input, batch_target in val_loader:
                batch_input = batch_input.to(device)
                batch_target = batch_target.to(device)
                
                reconstructed = model(batch_input)
                loss = criterion(reconstructed, batch_target)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
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
                logger.info(f"   ⏹️  Early stopping triggered at epoch {epoch+1}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses, best_val_loss

# ============================================================================
# OPTUNA OBJECTIVE FOR HYPERPARAMETER TUNING
# ============================================================================

def autoencoder_objective(trial, train_data, val_data, input_dim, device):
    """
    Optuna objective function for autoencoder hyperparameter tuning
    """
    # Hyperparameters to tune
    num_encoder_layers = trial.suggest_int('num_encoder_layers', 1, 3)
    
    # Encoder dimensions (decreasing towards bottleneck)
    encoder_dims = []
    prev_dim = input_dim
    
    for i in range(num_encoder_layers):
        # Each layer reduces dimension
        hidden_dim = trial.suggest_categorical(f'encoder_dim_{i}', [256, 512, 1024, 2048])
        encoder_dims.append(hidden_dim)
        prev_dim = hidden_dim
    
    latent_dim = trial.suggest_categorical('latent_dim', [64, 128, 256, 512])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    # Create model
    model = DenseAutoencoder(
        input_dim=input_dim,
        encoder_dims=encoder_dims,
        latent_dim=latent_dim,
        dropout_rate=dropout
    ).to(device)
    
    # Data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    # Loss and optimizer
    criterion = nn.MSELoss()  # MSE for reconstruction
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Train
    _, _, _, best_val_loss = train_autoencoder_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        epochs=30,  # More epochs for autoencoder
        patience=10,
        device=device
    )
    
    return best_val_loss

# ============================================================================
# MAIN TRAINING FUNCTION (CALLED FROM app.py)
# ============================================================================

def train_autoencoder(X_train_scaled, X_val_scaled, device, save_dir, feature_names, input_scaler,timestamps_train, timestamps_val):
    """
    Main function to train autoencoder with Optuna tuning
    
    Args:
        X_train_scaled: Scaled training data, shape (num_sequences, lookback, num_features)
        X_val_scaled: Scaled validation data, shape (num_sequences, lookback, num_features)
        device: torch device (cuda or cpu)
        save_dir: Directory to save model artifacts
    
    Returns:
        dict with keys: 'r2', 'rmse', 'mae', 'model_path', 
                       'reconstruction_errors_train', 'reconstruction_errors_val'
    """
    logger.info("=" * 80)
    logger.info("🤖 AUTOENCODER TRAINING STARTED")
    logger.info("=" * 80)
    
    # Step 1: Flatten sequences
    logger.info("📦 Step 1: Flattening sequences for dense autoencoder...")
    
    num_train_sequences, lookback, num_features = X_train_scaled.shape
    num_val_sequences = X_val_scaled.shape[0]
    
    logger.info(f"   Original train shape: {X_train_scaled.shape}")
    logger.info(f"   Original val shape: {X_val_scaled.shape}")
    
    # Flatten: (num_sequences, lookback, num_features) -> (num_sequences, lookback * num_features)
    X_train_flat = X_train_scaled.reshape(num_train_sequences, -1)
    X_val_flat = X_val_scaled.reshape(num_val_sequences, -1)
    
    input_dim = X_train_flat.shape[1]  # lookback * num_features
    
    logger.info(f"   Flattened train shape: {X_train_flat.shape}")
    logger.info(f"   Flattened val shape: {X_val_flat.shape}")
    logger.info(f"   Input dimension: {input_dim}")
    
    # Step 2: Create datasets
    logger.info("📊 Step 2: Creating autoencoder datasets...")
    train_dataset = AutoencoderDataset(X_train_flat)
    val_dataset = AutoencoderDataset(X_val_flat)
    
    logger.info(f"   Train dataset size: {len(train_dataset)}")
    logger.info(f"   Val dataset size: {len(val_dataset)}")
    
    # Step 3: Optuna hyperparameter tuning
    logger.info("🔍 Step 3: Running Optuna hyperparameter tuning (20 trials)...")
    
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: autoencoder_objective(trial, train_dataset, val_dataset, input_dim, device),
        n_trials=20,
        show_progress_bar=True
    )
    
    best_params = study.best_params
    logger.info(f"✅ Best hyperparameters found:")
    for key, value in best_params.items():
        logger.info(f"   {key}: {value}")
    
    # Step 4: Train final model with best parameters
    logger.info("🏋️  Step 4: Training final autoencoder model...")
    
    # Extract encoder dimensions from best params
    num_encoder_layers = best_params['num_encoder_layers']
    encoder_dims = [best_params[f'encoder_dim_{i}'] for i in range(num_encoder_layers)]
    
    final_model = DenseAutoencoder(
        input_dim=input_dim,
        encoder_dims=encoder_dims,
        latent_dim=best_params['latent_dim'],
        dropout_rate=best_params['dropout']
    ).to(device)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(final_model.parameters(), lr=best_params['lr'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Train
    final_model, train_losses, val_losses, best_val_loss = train_autoencoder_model(
        final_model, train_loader, val_loader, criterion, optimizer, scheduler,
        epochs=100,  # More epochs for final training
        patience=20,
        device=device
    )
    
    logger.info(f"✅ Training complete! Best validation loss: {best_val_loss:.6f}")
    
    # Step 5: Calculate reconstruction errors and metrics
    logger.info("📊 Step 5: Calculating reconstruction errors and metrics...")
    
    final_model.eval()
    
    # Get reconstructions for train set
    train_reconstructions = []
    train_originals = []
    
    with torch.no_grad():
        for batch_input, batch_target in train_loader:
            batch_input = batch_input.to(device)
            reconstructed = final_model(batch_input)
            train_reconstructions.append(reconstructed.cpu().numpy())
            train_originals.append(batch_target.cpu().numpy())
    
    train_reconstructions = np.concatenate(train_reconstructions, axis=0)
    train_originals = np.concatenate(train_originals, axis=0)
    
    # Get reconstructions for val set
    val_reconstructions = []
    val_originals = []
    
    with torch.no_grad():
        for batch_input, batch_target in val_loader:
            batch_input = batch_input.to(device)
            reconstructed = final_model(batch_input)
            val_reconstructions.append(reconstructed.cpu().numpy())
            val_originals.append(batch_target.cpu().numpy())
    
    val_reconstructions = np.concatenate(val_reconstructions, axis=0)
    val_originals = np.concatenate(val_originals, axis=0)
    
    # Calculate reconstruction errors (MSE per sequence)
    reconstruction_errors_train = np.mean((train_originals - train_reconstructions) ** 2, axis=1)
    reconstruction_errors_val = np.mean((val_originals - val_reconstructions) ** 2, axis=1)
    
    logger.info(f"   Train reconstruction errors - Mean: {np.mean(reconstruction_errors_train):.6f}, "
                f"Std: {np.std(reconstruction_errors_train):.6f}")
    logger.info(f"   Val reconstruction errors - Mean: {np.mean(reconstruction_errors_val):.6f}, "
                f"Std: {np.std(reconstruction_errors_val):.6f}")
    
    # Calculate overall metrics (flatten everything)
    train_flat = train_originals.flatten()
    train_recon_flat = train_reconstructions.flatten()
    val_flat = val_originals.flatten()
    val_recon_flat = val_reconstructions.flatten()
    
    # Training metrics
    train_rmse = np.sqrt(mean_squared_error(train_flat, train_recon_flat))
    train_mae = mean_absolute_error(train_flat, train_recon_flat)
    train_r2 = r2_score(train_flat, train_recon_flat)
    
    # Validation metrics
    val_rmse = np.sqrt(mean_squared_error(val_flat, val_recon_flat))
    val_mae = mean_absolute_error(val_flat, val_recon_flat)
    val_r2 = r2_score(val_flat, val_recon_flat)
    
    logger.info("=" * 80)
    logger.info("📈 TRAINING SET METRICS:")
    logger.info(f"   RMSE: {train_rmse:.6f}")
    logger.info(f"   MAE:  {train_mae:.6f}")
    logger.info(f"   R²:   {train_r2:.6f}")
    logger.info("=" * 80)
    logger.info("📈 VALIDATION SET METRICS:")
    logger.info(f"   RMSE: {val_rmse:.6f}")
    logger.info(f"   MAE:  {val_mae:.6f}")
    logger.info(f"   R²:   {val_r2:.6f}")
    logger.info("=" * 80)

    logger.info("📊 Step 5.1: Creating per-feature plot points...")

    logger.info("🔄 Inverse transforming to original scale...")

    # val_originals_original = input_scaler.inverse_transform(val_originals)
    # val_reconstructions_original = input_scaler.inverse_transform(val_reconstructions)
    
    # val_originals_unflattened = val_originals_original.reshape(num_val_sequences, lookback, num_features)
    # val_reconstructions_unflattened = val_reconstructions_original.reshape(num_val_sequences, lookback, num_features)
    val_originals_unflattened = val_originals.reshape(num_val_sequences, lookback, num_features)
    val_reconstructions_unflattened = val_reconstructions.reshape(num_val_sequences, lookback, num_features)

    # Now inverse transform: reshape to (sequences * lookback, features), transform, reshape back
    val_originals_reshaped = val_originals_unflattened.reshape(-1, num_features)  # (250*672, 12)
    val_reconstructions_reshaped = val_reconstructions_unflattened.reshape(-1, num_features)  # (250*672, 12)

    # Inverse transform (now shapes match!)
    val_originals_original = input_scaler.inverse_transform(val_originals_reshaped)
    val_reconstructions_original = input_scaler.inverse_transform(val_reconstructions_reshaped)

    # Reshape back to (sequences, lookback, features)
    val_originals_unflattened = val_originals_original.reshape(num_val_sequences, lookback, num_features)
    val_reconstructions_unflattened = val_reconstructions_original.reshape(num_val_sequences, lookback, num_features)

    # Also inverse transform TRAIN data (needed for anomaly detection)
    train_originals_unflattened_temp = train_originals.reshape(num_train_sequences, lookback, num_features)
    train_reconstructions_unflattened_temp = train_reconstructions.reshape(num_train_sequences, lookback, num_features)

    train_originals_reshaped = train_originals_unflattened_temp.reshape(-1, num_features)
    train_reconstructions_reshaped = train_reconstructions_unflattened_temp.reshape(-1, num_features)

    train_originals_original = input_scaler.inverse_transform(train_originals_reshaped)
    train_reconstructions_original = input_scaler.inverse_transform(train_reconstructions_reshaped)

    train_originals_unflattened = train_originals_original.reshape(num_train_sequences, lookback, num_features)
    train_reconstructions_unflattened = train_reconstructions_original.reshape(num_train_sequences, lookback, num_features)

    logger.info("✅ Data inverse transformed to original scale")

    # We'll use the LAST timestep of each sequence for plotting (like TS model does)
    # Or flatten all timesteps - your choice. Let's flatten all for more data points.

    plot_data = []

    # Loop through each feature
    for feature_idx in range(num_features):
        feature_name = feature_names[feature_idx]
        # Extract this feature across all sequences and all timesteps
       
        actual_feature = val_originals_unflattened[:, -1, feature_idx]
        # reconstructed_feature = val_reconstructions_unflattened[:, :, feature_idx].flatten()
        reconstructed_feature = val_reconstructions_unflattened[:, -1, feature_idx] 
        
        # Calculate per-feature metrics
        feature_rmse = np.sqrt(mean_squared_error(actual_feature, reconstructed_feature))
        feature_mae = mean_absolute_error(actual_feature, reconstructed_feature)
        feature_r2 = r2_score(actual_feature, reconstructed_feature)
        
        
        all_actual_feature = np.concatenate([
        train_originals_unflattened[:, -1, feature_idx],
        val_originals_unflattened[:, -1, feature_idx]
        ])
    
        all_reconstructed_feature = np.concatenate([
            train_reconstructions_unflattened[:, -1, feature_idx],
            val_reconstructions_unflattened[:, -1, feature_idx]
        ])
        
        # Calculate reconstruction errors for this feature
        feature_errors = np.abs(all_actual_feature - all_reconstructed_feature)
        
        # Calculate error thresholds
        mean_error_feat = np.mean(feature_errors)
        std_error_feat = np.std(feature_errors)
        error_threshold_upper = mean_error_feat + 3 * std_error_feat
        
        # Find normal values (where error <= threshold)
        normal_mask_feat = feature_errors <= error_threshold_upper
        normal_values_feat = all_actual_feature[normal_mask_feat]
        
        # Calculate VALUE thresholds
        if len(normal_values_feat) == 0:
            logger.warning(f"      ⚠️  No normal values for {feature_name}, using all values")
            value_threshold_lower = float(np.min(all_actual_feature))
            value_threshold_upper = float(np.max(all_actual_feature))
        else:
            value_threshold_lower = float(np.min(normal_values_feat))
            value_threshold_upper = float(np.max(normal_values_feat))
        
        # ========================================================================
        
        # Create timestamps
        # feature_timestamps = [f"seq_{seq_idx}" for seq_idx in range(num_val_sequences)]
        # Use actual timestamps from validation set
        feature_timestamps = [str(ts) for ts in timestamps_val]
        
        logger.info(f"   📈 {feature_name}:")
        logger.info(f"      RMSE={feature_rmse:.4f}, MAE={feature_mae:.4f}, R²={feature_r2:.4f}")
        logger.info(f"      Value thresholds: [{value_threshold_lower:.2f}, {value_threshold_upper:.2f}]")
        
        plot_data.append({
        "feature_name": feature_name,
        "actual": [round(float(x), 2) for x in actual_feature],           # ← ROUND
        "reconstructed": [round(float(x), 2) for x in reconstructed_feature],  # ← ROUND
        "threshold_lower": round(value_threshold_lower, 2),               # ← ROUND
        "threshold_upper": round(value_threshold_upper, 2),               # ← ROUND
        "timestamps": feature_timestamps,
        "metrics": {
            "rmse": round(float(feature_rmse), 2),                        # ← ROUND
            "mae": round(float(feature_mae), 2),                          # ← ROUND
            "r2": round(float(feature_r2), 4)                             # ← Keep R² at 4 decimals
        }
        })

    logger.info(f"✅ Created plot points for {len(plot_data)} features")
    logger.info("📅 Sorting plot_data by timestamps...")

    for feature_data in plot_data:
        # Get timestamps, actual, reconstructed
        timestamps = feature_data['timestamps']
        actual = feature_data['actual']
        reconstructed = feature_data['reconstructed']
        
        # Create list of (timestamp, actual, reconstructed) tuples
        combined = list(zip(timestamps, actual, reconstructed))
        
        # Sort by timestamp
        combined_sorted = sorted(combined, key=lambda x: x[0])
        
        # Unpack back into separate lists
        timestamps_sorted, actual_sorted, reconstructed_sorted = zip(*combined_sorted)
        
        # Update the feature data
        feature_data['timestamps'] = list(timestamps_sorted)
        feature_data['actual'] = list(actual_sorted)
        feature_data['reconstructed'] = list(reconstructed_sorted)

    logger.info("✅ Plot data sorted chronologically")

    

    

    # ========================================================================
    # OVERALL (ROW-WISE) ANOMALY DETECTION
    # ========================================================================

    logger.info("🚨 Step 5.3: Calculating overall row-wise anomaly thresholds...")

    # Combine train + val data (last timestep only) - ALL FEATURES
    all_actual_overall = np.concatenate([
        train_originals_unflattened[:, -1, :],  # (num_train, num_features)
        val_originals_unflattened[:, -1, :]     # (num_val, num_features)
    ], axis=0)

    all_reconstructed_overall = np.concatenate([
        train_reconstructions_unflattened[:, -1, :],
        val_reconstructions_unflattened[:, -1, :]
    ], axis=0)

    # Calculate ROW-WISE reconstruction error (across all features per sequence)
    reconstruction_errors_overall = np.sqrt(np.sum((all_actual_overall - all_reconstructed_overall) ** 2, axis=1))

    # Calculate mean and std
    # Calculate mean and std (for logging)
    mean_error_overall = np.mean(reconstruction_errors_overall)
    std_error_overall = np.std(reconstruction_errors_overall)

    # Use percentile thresholds (more robust to outliers)
    # This captures 99.8% of data as "normal" (0.1% on each tail)
    threshold_lower_overall = float(np.percentile(reconstruction_errors_overall, 1))
    threshold_upper_overall = float(np.percentile(reconstruction_errors_overall, 99))

    # # Flag anomalies (outside EITHER threshold)
    # anomaly_flags_overall = ((reconstruction_errors_overall < threshold_lower_overall) | 
    #                         (reconstruction_errors_overall > threshold_upper_overall)).tolist()
    # Flag anomalies (ONLY above upper threshold)
    anomaly_flags_overall = (reconstruction_errors_overall > threshold_upper_overall).tolist()

    # Count anomalies
    num_anomalies_overall = sum(anomaly_flags_overall)
    total_points_overall = len(reconstruction_errors_overall)
    clean_percentage_overall = ((total_points_overall - num_anomalies_overall) / total_points_overall) * 100

    logger.info(f"   Total sequences: {total_points_overall}")
    logger.info(f"   Mean error: {mean_error_overall:.4f}")
    logger.info(f"   Std error: {std_error_overall:.4f}")
    logger.info(f"   Percentile thresholds (1%, 99%):")
    logger.info(f"   Lower threshold: {threshold_lower_overall:.4f}")
    logger.info(f"   Upper threshold: {threshold_upper_overall:.4f}")
    logger.info(f"   Anomalies: {num_anomalies_overall}/{total_points_overall} ({100-clean_percentage_overall:.1f}%)")

    all_timestamps = timestamps_train + timestamps_val

    combined_overall = list(zip(all_timestamps, reconstruction_errors_overall, anomaly_flags_overall))
    combined_overall_sorted = sorted(combined_overall, key=lambda x: x[0])
    timestamps_sorted, errors_sorted, flags_sorted = zip(*combined_overall_sorted)
    overall_rmse = float(np.sqrt(np.mean(np.array(reconstruction_errors_overall) ** 2)))
    overall_mae = float(np.mean(np.abs(reconstruction_errors_overall)))
    overall_std = float(np.std(reconstruction_errors_overall))

    logger.info(f"   Overall RMSE: {overall_rmse:.4f}")
    logger.info(f"   Overall MAE: {overall_mae:.4f}")
    logger.info(f"   Overall Std: {overall_std:.4f}")
    overall_metrics_to_save = {
    "rmse": round(overall_rmse, 2),
    "mae": round(overall_mae, 2),
    "std": round(overall_std, 2)
}

    # Create overall anomaly detection data
    anomaly_detection_overall = {
    "reconstruction_errors": [round(float(e), 2) for e in errors_sorted],  # ← ROUND
    "threshold_lower": round(float(threshold_lower_overall), 2),            # ← ROUND
    "threshold_upper": round(float(threshold_upper_overall), 2),            # ← ROUND
    "timestamps": [str(ts) for ts in timestamps_sorted],
    "anomaly_flags": list(flags_sorted), 
    "metrics": {
        "rmse": round(overall_rmse, 2),                                     # ← ROUND
        "mae": round(overall_mae, 2),                                       # ← ROUND
        "std": round(overall_std, 2)                                        # ← ROUND
    }
}

    logger.info("✅ Overall anomaly data sorted chronologically")
    logger.info(f"✅ Overall row-wise anomaly detection complete")

    # ========================================================================
    # STEP 5.4: Save thresholds for future inference
    # ========================================================================
    
    logger.info("💾 Step 5.4: Preparing thresholds for inference...")
    
    thresholds_to_save = {
        "overall_threshold_lower": round(float(threshold_lower_overall), 2),
        "overall_threshold_upper": round(float(threshold_upper_overall), 2),
        "overall_metrics": overall_metrics_to_save,  # ← ADD THIS LINE!
        "per_feature_thresholds": {}
    }
    
    # Add per-feature thresholds from plot_data
    for feature_data in plot_data:
        feature_name = feature_data['feature_name']
        thresholds_to_save["per_feature_thresholds"][feature_name] = {
        "threshold_lower": feature_data['threshold_lower'],
        "threshold_upper": feature_data['threshold_upper'],
        "metrics": feature_data['metrics']  # ← ADD THIS LINE!
    }
    
    logger.info(f"   Overall thresholds: [{thresholds_to_save['overall_threshold_lower']}, {thresholds_to_save['overall_threshold_upper']}]")
    logger.info(f"   Per-feature thresholds: {len(thresholds_to_save['per_feature_thresholds'])} features")



    
    # Step 6: Save model artifacts
    logger.info("💾 Step 6: Saving autoencoder model artifacts...")
    
    autoencoder_dir = os.path.join(save_dir, "autoencoder")
    os.makedirs(autoencoder_dir, exist_ok=True)
    
    model_path = os.path.join(autoencoder_dir, "autoencoder_model.pth")
    metadata_path = os.path.join(autoencoder_dir, "autoencoder_metadata.json")
    errors_path = os.path.join(autoencoder_dir, "reconstruction_errors.pkl")
    thresholds_path = os.path.join(autoencoder_dir, "autoencoder_thresholds.json")
    
    # Save model state dict
    torch.save(final_model.state_dict(), model_path)
    logger.info(f"   ✅ Model saved to: {model_path}")
    
    # Save metadata
    metadata = {
        "input_dim": int(input_dim),
        "lookback": int(lookback),
        "num_features": int(num_features),
        "encoder_dims": encoder_dims,
        "latent_dim": best_params['latent_dim'],
        "best_params": best_params,
        "best_val_loss": float(best_val_loss),
        "final_train_loss": float(train_losses[-1]),
        "final_val_loss": float(val_losses[-1]),
        "num_epochs": len(train_losses),
        "training_date": str(datetime.now()),
        "train_metrics": {
            "rmse": float(train_rmse),
            "mae": float(train_mae),
            "r2": float(train_r2)
        },
        "val_metrics": {
            "rmse": float(val_rmse),
            "mae": float(val_mae),
            "r2": float(val_r2)
        },
        "reconstruction_errors_stats": {
            "train_mean": float(np.mean(reconstruction_errors_train)),
            "train_std": float(np.std(reconstruction_errors_train)),
            "val_mean": float(np.mean(reconstruction_errors_val)),
            "val_std": float(np.std(reconstruction_errors_val))
        }
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"   ✅ Metadata saved to: {metadata_path}")
    
    # Save reconstruction errors for future anomaly detection
    errors_data = {
        'train_errors': reconstruction_errors_train.tolist(),
        'val_errors': reconstruction_errors_val.tolist()
    }
    
    with open(errors_path, 'wb') as f:
        pickle.dump(errors_data, f)
    logger.info(f"   ✅ Reconstruction errors saved to: {errors_path}")

    with open(thresholds_path, 'w') as f:
        json.dump(thresholds_to_save, f, indent=2)
    logger.info(f"   ✅ Thresholds saved to: {thresholds_path}")
    
    logger.info("=" * 80)
    logger.info("🎉 AUTOENCODER TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    
    # Return results
    # Return results
    # return {
    #     'r2': float(val_r2),
    #     'rmse': float(val_rmse),
    #     'mae': float(val_mae),
    #     'train_r2': float(train_r2),
    #     'train_rmse': float(train_rmse),
    #     'train_mae': float(train_mae),
    #     'model_path': model_path,
    #     'reconstruction_errors_train': reconstruction_errors_train.tolist(),
    #     'reconstruction_errors_val': reconstruction_errors_val.tolist(),
    #     'best_params': best_params,
    #     'metadata_path': metadata_path,
    #     'plot_data': plot_data  # ← ADD THIS LINE
    # }
    return {
    'r2': float(val_r2),
    'rmse': float(val_rmse),
    'mae': float(val_mae),
    'model_path': model_path,
    'plot_data': plot_data,
    'anomaly_detection_overall': anomaly_detection_overall
}