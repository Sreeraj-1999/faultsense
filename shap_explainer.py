import torch
import numpy as np
import pandas as pd
import pickle
import json
import logging
from datetime import datetime
import shap
from sklearn.preprocessing import StandardScaler
from autoencoder import DenseAutoencoder
from models_and_utils import create_sequences
import config
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# SHAP EXPLAINER FOR AUTOENCODER ANOMALIES
# ============================================================================

def explain_autoencoder_anomalies(start_date, end_date, vessel_imo, equipment, device):
    """
    Use SHAP to explain which features contribute most to anomalies in date range
    
    Args:
        start_date: str, format "YYYY-MM-DD"
        end_date: str, format "YYYY-MM-DD"
        vessel_imo: str
        equipment: str (e.g., "ME_CYL")
        device: torch device
    
    Returns:
        dict with feature importance, anomalies, and plot data
    """
    logger.info("=" * 80)
    logger.info("🔍 SHAP ANOMALY EXPLANATION")
    logger.info("=" * 80)
    logger.info(f"📅 Date Range: {start_date} to {end_date}")
    logger.info(f"🚢 Vessel: {vessel_imo}, Equipment: {equipment}")
    
    # ========================================================================
    # STEP 1: Load saved autoencoder model and metadata
    # ========================================================================
    
    logger.info("📦 Step 1: Loading saved autoencoder model...")
    
    equipment_file = equipment.replace('>>', '_')
    model_dir = f"models/{vessel_imo}_{equipment_file}"
    autoencoder_dir = f"{model_dir}/autoencoder"
    
    # Load metadata
    metadata_path = f"{autoencoder_dir}/autoencoder_metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    input_dim = metadata['input_dim']
    encoder_dims = metadata['encoder_dims']
    latent_dim = metadata['latent_dim']
    lookback = metadata['lookback']
    num_features = metadata['num_features']
    dropout = metadata['best_params']['dropout']
    
    logger.info(f"   Input dim: {input_dim}, Latent dim: {latent_dim}")
    
    # Load model
    model = DenseAutoencoder(
        input_dim=input_dim,
        encoder_dims=encoder_dims,
        latent_dim=latent_dim,
        dropout_rate=dropout
    ).to(device)
    
    model_path = f"{autoencoder_dir}/autoencoder_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    logger.info("   ✅ Model loaded successfully")
    
    # Load input scaler
    input_scaler_path = f"{model_dir}/input_scaler.pkl"
    with open(input_scaler_path, 'rb') as f:
        input_scaler = pickle.load(f)
    
    logger.info("   ✅ Scaler loaded successfully")

    # ========================================================================
    # STEP 1.5: Load saved thresholds from training
    # ========================================================================
    
    logger.info("📊 Step 1.5: Loading saved thresholds from training...")
    
    thresholds_path = f"{autoencoder_dir}/autoencoder_thresholds.json"
    
    if not os.path.exists(thresholds_path):
        raise FileNotFoundError(
            f"Thresholds file not found: {thresholds_path}\n"
            f"This model may have been trained before thresholds saving was implemented. "
            f"Please retrain the model."
        )
    
    with open(thresholds_path, 'r') as f:
        saved_thresholds = json.load(f)
    
    threshold_lower_from_training = saved_thresholds['overall_threshold_lower']
    threshold_upper_from_training = saved_thresholds['overall_threshold_upper']
    per_feature_thresholds_from_training = saved_thresholds['per_feature_thresholds']
    training_overall_metrics = saved_thresholds['overall_metrics']
    
    logger.info(f"   Overall thresholds: [{threshold_lower_from_training}, {threshold_upper_from_training}]")
    logger.info(f"   Per-feature thresholds loaded: {len(per_feature_thresholds_from_training)} features")
    
    # ========================================================================
    # STEP 2: Load and filter data by date range
    # ========================================================================
    
    logger.info("📊 Step 2: Loading and filtering data by date range...")
    
    csv_path = config.CSV_PATH_COMBINED
    df = pd.read_csv(csv_path)
    df = df.sort_values(['vessel_id', 'Local_time']).reset_index(drop=True)
    df['Local_time'] = pd.to_datetime(df['Local_time'], format='mixed', dayfirst=True)
    
    # Filter by date range
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    df_filtered = df[(df['Local_time'] >= start_dt) & (df['Local_time'] <= end_dt)].copy()
    
    logger.info(f"   Original data: {len(df)} rows")
    logger.info(f"   Filtered data: {len(df_filtered)} rows")
    
    if len(df_filtered) < lookback + config.HORIZON_SEQUENCE:
        raise ValueError(f"Insufficient data in date range. Need at least {lookback + config.HORIZON_SEQUENCE} rows.")
    # ========================================================================
    # STEP 2.5: Create calculated columns (LaTeX expressions)
    # ========================================================================
    
    logger.info("🔧 Step 2.5: Creating calculated columns from LaTeX expressions...")
    
    # Load EFD array from training metadata to get LaTeX expressions
    training_metadata_path = f"{model_dir}/metadata.json"
    with open(training_metadata_path, 'r') as f:
        training_metadata = json.load(f)
    
    # Check if we have stored the EFD definitions (we need to add this during training)
    # For now, we'll need to get it from the database or pass it in the request
    # TEMPORARY WORKAROUND: Load from a saved file if available
    
    efd_definitions_path = f"{model_dir}/efd_definitions.json"
    if os.path.exists(efd_definitions_path):
        with open(efd_definitions_path, 'r') as f:
            efd_array = json.load(f)
        
        import re
        
        for efd in efd_array:
            if efd.get('tag_name') is None and efd.get('latex') is not None:
                logger.info(f"   Creating calculated column '{efd['name']}'...")
                
                tag_pattern = r'\\text\{([^}]+)\}'
                found_tags = re.findall(tag_pattern, efd['latex'])
                
                missing_tags = [tag for tag in found_tags if tag not in df_filtered.columns]
                if missing_tags:
                    raise ValueError(f"Tags not found for {efd['name']}: {missing_tags}")
                
                python_expression = efd['latex']
                python_expression = python_expression.replace('\\left(', '(')
                python_expression = python_expression.replace('\\right)', ')')
                python_expression = python_expression.replace('\\ ', ' ')
                python_expression = python_expression.replace('\\div', '/')
                python_expression = python_expression.replace('\\times', '*')
                python_expression = python_expression.replace('^', '**')
                
                for tag in found_tags:
                    python_expression = python_expression.replace(f'\\text{{{tag}}}', f"df_filtered['{tag}']")
                
                # Zero division fix
                if '/' in python_expression:
                    for tag in found_tags:
                        df_filtered[tag] = df_filtered[tag].replace(0, 0.001)
                
                df_filtered[efd['name']] = eval(python_expression)
                logger.info(f"   ✅ Column '{efd['name']}' created")
    else:
        logger.warning("   ⚠️  No EFD definitions found, skipping calculated columns")
    
    # ========================================================================
    # STEP 3: Get feature names from metadata file (or reconstruct)
    # ========================================================================
    
    logger.info("📋 Step 3: Getting feature names...")
    
    # Load from training metadata
    training_metadata_path = f"{model_dir}/metadata.json"
    with open(training_metadata_path, 'r') as f:
        training_metadata = json.load(f)
    
    feature_names = training_metadata['input_features']
    logger.info(f"   Features: {feature_names}")
    
    # Check if all features exist in filtered data
    missing_features = [f for f in feature_names if f not in df_filtered.columns]
    if missing_features:
        raise ValueError(f"Missing features in data: {missing_features}")
    
    # ========================================================================
    # STEP 4: Create sequences from filtered data
    # ========================================================================
    
    logger.info("🔄 Step 4: Creating sequences from filtered data...")
    
    # Get vessel boundaries (if multiple vessels)
    vessel_groups = df_filtered.groupby('vessel_id').groups
    vessel_boundaries = []
    for vessel_id in sorted(vessel_groups.keys())[1:]:
        first_idx = vessel_groups[vessel_id][0]
        # Adjust index relative to filtered dataframe
        relative_idx = df_filtered.index.get_loc(first_idx)
        vessel_boundaries.append(relative_idx)
    vessel_boundaries = sorted([b for b in vessel_boundaries if b > 0])
    
    # Create sequences (using same parameters as training)
    X_sequences, _, timestamps_sequences = create_sequences(
        df_filtered, 
        feature_names, 
        feature_names,  # input = output for autoencoder
        'Local_time',
        vessel_boundaries, 
        config.LOOKBACK, 
        config.HORIZON_SEQUENCE, 
        config.STEP
    )
    
    logger.info(f"   Created {len(X_sequences)} sequences")
    
    if len(X_sequences) == 0:
        raise ValueError("No sequences created from date range. Try a larger date range.")
    
    # ========================================================================
    # STEP 5: Scale and flatten sequences
    # ========================================================================
    
    logger.info("📊 Step 5: Scaling and flattening sequences...")
    
    # Reshape and scale
    num_sequences = len(X_sequences)
    X_reshaped = X_sequences.reshape(-1, num_features)
    X_scaled = input_scaler.transform(X_reshaped).reshape(X_sequences.shape)
    
    # Flatten for autoencoder
    X_flat = X_scaled.reshape(num_sequences, -1)
    
    logger.info(f"   Flattened shape: {X_flat.shape}")
    
    # ========================================================================
    # STEP 6: Get reconstructions and calculate errors
    # ========================================================================
    
    logger.info("🤖 Step 6: Running autoencoder inference...")
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_flat).to(device)
        reconstructions = model(X_tensor).cpu().numpy()
    
    # INVERSE TRANSFORM BEFORE CALCULATING ERRORS (same as training!)
    logger.info("🔄 Step 6.1: Inverse transforming to original scale...")
    
    X_unflattened = X_scaled.reshape(num_sequences, lookback, num_features)
    reconstructions_unflattened = reconstructions.reshape(num_sequences, lookback, num_features)
    
    # Reshape for inverse transform
    X_for_inverse = X_unflattened.reshape(-1, num_features)
    recon_for_inverse = reconstructions_unflattened.reshape(-1, num_features)
    
    # Inverse transform to original scale
    X_original = input_scaler.inverse_transform(X_for_inverse).reshape(num_sequences, lookback, num_features)
    recon_original = input_scaler.inverse_transform(recon_for_inverse).reshape(num_sequences, lookback, num_features)
    
    # Use LAST TIMESTEP only (same as training!)
    X_original_last = X_original[:, -1, :]  # (num_sequences, num_features)
    recon_original_last = recon_original[:, -1, :]  # (num_sequences, num_features)
    
    # Calculate reconstruction errors ON ORIGINAL SCALE (same as training!)
    reconstruction_errors = np.sqrt(np.sum((X_original_last - recon_original_last) ** 2, axis=1))
    
    logger.info("✅ Errors calculated on original scale")
    
    # Use thresholds from training (NOT recalculated!)
    threshold_lower = threshold_lower_from_training
    threshold_upper = threshold_upper_from_training
    
    logger.info(f"   Using training thresholds: [{threshold_lower:.4f}, {threshold_upper:.4f}]")
    
    # Calculate mean/std of date range errors (for logging only, not for thresholds!)
    mean_error = np.mean(reconstruction_errors)
    std_error = np.std(reconstruction_errors)
    logger.info(f"   Date range error stats: mean={mean_error:.4f}, std={std_error:.4f}")
    
    # Flag anomalies using TRAINING thresholds (only upper)
    anomaly_flags = (reconstruction_errors > threshold_upper)
    
    num_anomalies = np.sum(anomaly_flags)
    
    logger.info(f"   Anomalies found: {num_anomalies}/{num_sequences} ({num_anomalies/num_sequences*100:.1f}%)")
    
    # ========================================================================
    # STEP 7: SHAP Explanation (only if anomalies exist)
    # ========================================================================
    
    if num_anomalies == 0:
        logger.warning("⚠️  No anomalies found in date range. Returning empty SHAP values.")
        
        feature_importance = [
            {
                "feature_name": name,
                "shap_score": 0.0,
                "rank": i+1
            }
            for i, name in enumerate(feature_names)
        ]
    else:
        logger.info("🔍 Step 7: Calculating SHAP values for anomalous sequences...")
        
        # Get anomalous sequences
        anomalous_sequences = X_flat[anomaly_flags]
        
        logger.info(f"   Analyzing {len(anomalous_sequences)} anomalous sequences...")
        
        # Create background data (sample from normal sequences for efficiency)
        normal_sequences = X_flat[~anomaly_flags]
        
        if len(normal_sequences) > 100:
            # Sample 100 random normal sequences for background(with fixed seed)
            np.random.seed(42)
            background_indices = np.random.choice(len(normal_sequences), size=100, replace=False)
            background_data = normal_sequences[background_indices]
        else:
            background_data = normal_sequences
        
        logger.info(f"   Using {len(background_data)} background sequences")
        
        # Convert to torch tensors
        background_tensor = torch.FloatTensor(background_data).to(device)
        
        # Create SHAP DeepExplainer
        # Create SHAP GradientExplainer (better for autoencoders)
        # Create SHAP KernelExplainer for autoencoder
        logger.info("   Creating SHAP explainer...")
        
        # Define prediction function that SHAP can use
        def predict_reconstruction_error(data_sample):
            """
            Given input data, return reconstruction error
            This is what SHAP will explain
            """
            if len(data_sample.shape) == 1:
                data_sample = data_sample.reshape(1, -1)
            
            with torch.no_grad():
                input_tensor = torch.FloatTensor(data_sample).to(device)
                reconstructed = model(input_tensor).cpu().numpy()
                
                # Calculate reconstruction error per sample
                errors = np.sqrt(np.sum((data_sample - reconstructed) ** 2, axis=1))
            
            return errors
        
        np.random.seed(42)
        import random
        random.seed(42)
        
        # Create KernelExplainer with background data
        logger.info(f"   Using {len(background_data)} background sequences")
        explainer = shap.KernelExplainer(predict_reconstruction_error, background_data)
        
        # Calculate SHAP values for anomalous sequences
        logger.info("   Computing SHAP values (this may take 2-3 minutes)...")
        
        # Limit to first 20 anomalies for KernelExplainer (it's slower)
        if len(anomalous_sequences) > 20:
            logger.info("   (Limiting to first 20 anomalies for speed)")
            anomalous_sequences_sample = anomalous_sequences[:20]
        else:
            anomalous_sequences_sample = anomalous_sequences
        
        # KernelExplainer needs nsamples parameter
        shap_values = explainer.shap_values(anomalous_sequences_sample, nsamples=100)
        
        logger.info(f"   SHAP values shape: {shap_values.shape}")
        
        # shap_values shape: (num_anomalies, flattened_features)
        # We need to aggregate to get per-feature importance
        
        # Reshape SHAP values: (num_anomalies, lookback * num_features) -> (num_anomalies, lookback, num_features)
        shap_values_reshaped = shap_values.reshape(len(anomalous_sequences_sample), lookback, num_features)
        
        # Aggregate across sequences and timesteps (mean absolute SHAP value per feature)
        feature_shap_scores = np.mean(np.abs(shap_values_reshaped), axis=(0, 1))  # Average over sequences and timesteps
        
        logger.info("   ✅ SHAP values calculated")
        
        # Normalize to 0-1 range for display
        max_shap = np.max(feature_shap_scores)
        if max_shap > 0:
            feature_shap_scores_normalized = feature_shap_scores / max_shap
        else:
            feature_shap_scores_normalized = feature_shap_scores
        
        # Create feature importance ranking
        feature_importance = []
        for i, (name, score) in enumerate(zip(feature_names, feature_shap_scores_normalized)):
            feature_importance.append({
                "feature_name": name,
                "shap_score": round(float(score), 4),
                "rank": i+1  # Will be re-ranked after sorting
            })
        
        # Sort by SHAP score (descending)
        feature_importance.sort(key=lambda x: x['shap_score'], reverse=True)
        
        # Update ranks
        for i, item in enumerate(feature_importance):
            item['rank'] = i + 1
        
        logger.info("   📊 Top 5 contributing features:")
        for item in feature_importance[:5]:
            logger.info(f"      {item['rank']}. {item['feature_name']}: {item['shap_score']:.4f}")   ###
    
    # ========================================================================
    # STEP 8: Prepare plot data for date range
    # ========================================================================
    
    logger.info("📈 Step 8: Preparing plot data for frontend...")
    
    # Overall plot data
    plot_data_overall = {
    "timestamps": [str(ts) for ts in timestamps_sequences],
    "reconstruction_errors": [round(float(e), 2) for e in reconstruction_errors],
    "anomaly_flags": anomaly_flags.tolist(),
    "threshold_lower": round(float(threshold_lower), 2),
    "threshold_upper": round(float(threshold_upper), 2),
    "metrics": training_overall_metrics  # ← ADD THIS LINE!
}
    
    # Per-feature plot data
    # Inverse transform to get original values
    # X_unflattened = X_scaled.reshape(num_sequences, lookback, num_features)
    # reconstructions_unflattened = reconstructions.reshape(num_sequences, lookback, num_features)
    
    # # Reshape for inverse transform
    # X_for_inverse = X_unflattened.reshape(-1, num_features)
    # recon_for_inverse = reconstructions_unflattened.reshape(-1, num_features)
    
    # X_original = input_scaler.inverse_transform(X_for_inverse).reshape(num_sequences, lookback, num_features)
    # recon_original = input_scaler.inverse_transform(recon_for_inverse).reshape(num_sequences, lookback, num_features)
    
    # Use last timestep only
    plot_data_features = []
    
    for i, feature_name in enumerate(feature_names):
        actual_vals = X_original[:, -1, i]
        reconstructed_vals = recon_original[:, -1, i]
        
        # Get per-feature thresholds from training
        feature_thresholds = per_feature_thresholds_from_training.get(feature_name, {})

        feature_rank = next(
        (item['rank'] for item in feature_importance if item['feature_name'] == feature_name),None)
        # Add rank to metrics
        feature_metrics = feature_thresholds.get('metrics', {})
        if feature_metrics and feature_rank is not None:
            feature_metrics['rank'] = feature_rank
        
        plot_data_features.append({
            "feature_name": feature_name,
            "actual": [round(float(v), 2) for v in actual_vals],
            "reconstructed": [round(float(v), 2) for v in reconstructed_vals],
            "threshold_lower": feature_thresholds.get('threshold_lower'),
            "threshold_upper": feature_thresholds.get('threshold_upper'),
            "timestamps": [str(ts) for ts in timestamps_sequences],
            # "metrics": feature_thresholds.get('metrics')
            "metrics": feature_metrics
        })
    
    logger.info("   ✅ Plot data prepared")
    
    # ========================================================================
    # STEP 9: Return results
    # ========================================================================
    
    logger.info("=" * 80)
    logger.info("✅ SHAP EXPLANATION COMPLETED!")
    logger.info("=" * 80)
    
    return {
        "success": True,
        "date_range": {
            "start": start_date,
            "end": end_date,
            "total_sequences": int(num_sequences),
            "anomalies_found": int(num_anomalies),
            "anomaly_percentage": round(float(num_anomalies/num_sequences*100), 2)
        },
        "feature_importance": feature_importance,
        "anomaly_detection_overall": plot_data_overall,  # ← CHANGED THIS LINE
        "plot_data": plot_data_features                   # ← CHANGED THIS LINE
    }