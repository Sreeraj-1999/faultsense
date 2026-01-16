import torch
import numpy as np
import pandas as pd
import pickle
import json
import logging
from datetime import datetime
import shap
from sklearn.preprocessing import StandardScaler
from autoencoder2 import DenseAutoencoder
# from models_and_utils import create_sequences
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
    # VALIDATION 1: Check for invalid date range (time travel!)
    # ========================================================================

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    if start_dt > end_dt:
        logger.error(f"❌ Invalid date range: start_date ({start_date}) is after end_date ({end_date})")
        return {
            "success": False,
            "is_anomaly": False,
            "message": "Invalid date range: Start date cannot be after end date",
            "date_range": None,
            "feature_importance": [],
            "anomaly_detection_overall": None,
            "plot_data": []
        }

    logger.info("   ✅ Date range validation passed")
    
    # ========================================================================
    # STEP 1: Load saved autoencoder model and metadata
    # ========================================================================
    
    logger.info("📦 Step 1: Loading saved autoencoder model...")
    
    equipment_file = equipment.replace('>>', '_')
    model_dir = f"models/{vessel_imo}_{equipment_file}"
    autoencoder_dir = f"{model_dir}/autoencoder"
    
    # Load metadata
    # Load metadata
    metadata_path = f"{autoencoder_dir}/autoencoder_metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    input_dim = metadata['input_dim']  # Now just 17!
    encoder_dims = metadata['encoder_dims']
    latent_dim = metadata['latent_dim']
    num_features = metadata['num_features']  # 17
    dropout = metadata['best_params']['dropout']

    logger.info(f"   Input dim: {input_dim}, Latent dim: {latent_dim}")
    logger.info(f"   Model type: Row-based (no sequences)")
    
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
    
    # df_filtered = df[(df['Local_time'] >= start_dt) & (df['Local_time'] <= end_dt)].copy()  #replaced by down
    
    # logger.info(f"   Original data: {len(df)} rows")
    # logger.info(f"   Filtered data: {len(df_filtered)} rows")
    
    # if len(df_filtered) < 1:
    #     raise ValueError(f"No data found in date range {start_date} to {end_date}.")



    # logger.info(f"   ✅ Found {len(df_filtered)} rows in date range")

    df_filtered = df[(df['Local_time'] >= start_dt) & (df['Local_time'] <= end_dt)].copy()

    logger.info(f"   Original data: {len(df)} rows")
    logger.info(f"   Filtered data: {len(df_filtered)} rows")

    if len(df_filtered) < 1:
        logger.warning(f"⚠️  No data found in date range {start_date} to {end_date}")
        return {
            "success": False,
            "is_anomaly": False,
            "message": f"No data found in specified date range ({start_date} to {end_date})",
            "date_range": {
                "start": start_date,
                "end": end_date,
                "total_rows": 0,
                "anomalies_found": 0,
                "anomaly_percentage": 0.0
            },
            "feature_importance": [],
            "anomaly_detection_overall": None,
            "plot_data": []
        }

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
    # STEP 4: Extract rows from filtered data (NO sequences!)
    # ========================================================================

    logger.info("🔄 Step 4: Extracting rows from filtered data...")

    # Get feature data directly (no sequences!)
    X_rows = df_filtered[feature_names].values  # Shape: (num_rows, 17)
    timestamps_rows = df_filtered['Local_time'].tolist()

    num_rows = len(X_rows)

    logger.info(f"   Extracted {num_rows} rows")
    logger.info(f"   Shape: {X_rows.shape}")  # (num_rows, 17)

    if num_rows == 0:
        raise ValueError("No data found in date range.")
    
    # ========================================================================
    # STEP 5: Scale rows (NO flattening needed!)
    # ========================================================================

    logger.info("📊 Step 5: Scaling rows...")

    # Scale rows directly (no reshaping needed!)
    X_scaled = input_scaler.transform(X_rows)  # Shape: (num_rows, 17)

    logger.info(f"   Scaled shape: {X_scaled.shape}")  # (num_rows, 17)
    
    # ========================================================================
    # STEP 6: Get reconstructions and calculate errors
    # ========================================================================
    
    logger.info("🤖 Step 6: Running autoencoder inference...")

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled).to(device)  # (num_rows, 17)
        reconstructions = model(X_tensor).cpu().numpy()    # (num_rows, 17)

    # INVERSE TRANSFORM BEFORE CALCULATING ERRORS (same as training!)
    logger.info("🔄 Step 6.1: Inverse transforming to original scale...")

    # Inverse transform directly (no complex reshaping!)
    X_original = input_scaler.inverse_transform(X_scaled)          # (num_rows, 17)
    recon_original = input_scaler.inverse_transform(reconstructions)  # (num_rows, 17)

    # Calculate reconstruction errors ON ORIGINAL SCALE (same as training!)
    reconstruction_errors = np.sqrt(np.sum((X_original - recon_original) ** 2, axis=1))

    logger.info("✅ Errors calculated on original scale")
    logger.info(f"   Errors shape: {reconstruction_errors.shape}")  # (num_rows,)
    
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

    logger.info(f"   Anomalies found: {num_anomalies}/{num_rows} ({num_anomalies/num_rows*100:.1f}%)")
    
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
        
        # Get anomalous rows
        anomalous_rows = X_scaled[anomaly_flags]

        logger.info(f"   Analyzing {len(anomalous_rows)} anomalous rows...")

        # Create background data (sample from normal rows for efficiency)
        normal_rows = X_scaled[~anomaly_flags]

        if len(normal_rows) > 100:
            # Sample 100 random normal rows for background
            np.random.seed(42)
            background_indices = np.random.choice(len(normal_rows), size=100, replace=False)
            background_data = normal_rows[background_indices]
            logger.info(f"   Using {len(background_data)} background rows from query data")
        elif len(normal_rows) > 0:
            # Use all normal rows if less than 100
            background_data = normal_rows
            logger.info(f"   Using {len(background_data)} background rows from query data")
        else:
            # NO NORMAL DATA in query! Use TRAINING data as background!
            logger.warning("   ⚠️  No normal data in query range! Using training data as background...")
            
            # Load clean data from training
            clean_data_path = f"{model_dir}/clean_data_for_shap.pkl"
            
            if os.path.exists(clean_data_path):
                # Load pre-saved clean training data
                with open(clean_data_path, 'rb') as f:
                    training_clean_data = pickle.load(f)
                
                # Sample 100 rows from training
                if len(training_clean_data) > 100:
                    np.random.seed(42)
                    bg_indices = np.random.choice(len(training_clean_data), size=100, replace=False)
                    background_data = training_clean_data[bg_indices]
                else:
                    background_data = training_clean_data
                
                logger.info(f"   ✅ Using {len(background_data)} background rows from training data")
            else:
                # Fallback: Use small sample from QUERY data (even if anomalous)
                logger.warning("   ⚠️  No training background data saved! Using query data as fallback...")
                sample_size = min(50, len(X_scaled))
                np.random.seed(42)
                bg_indices = np.random.choice(len(X_scaled), size=sample_size, replace=False)
                background_data = X_scaled[bg_indices]
                logger.info(f"   Using {len(background_data)} rows from query as background (not ideal!)")
        
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
        if len(anomalous_rows) > 20:
            logger.info("   (Limiting to first 20 anomalies for speed)")
            anomalous_rows_sample = anomalous_rows[:20]
        else:
            anomalous_rows_sample = anomalous_rows

        # KernelExplainer needs nsamples parameter
        shap_values = explainer.shap_values(anomalous_rows_sample, nsamples=100)
        
        logger.info(f"   SHAP values shape: {shap_values.shape}")
        
        # shap_values shape: (num_anomalies, num_features)
        # Already in correct format - just aggregate across anomalies!

        # Aggregate across anomalies (mean absolute SHAP value per feature)
        feature_shap_scores = np.mean(np.abs(shap_values), axis=0)  # Shape: (17,)
        
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
    # Overall plot data (with frozen metrics from training!)
    plot_data_overall = {
        "timestamps": [str(ts) for ts in timestamps_rows],  # ← Changed from timestamps_sequences
        "reconstruction_errors": [round(float(e), 2) for e in reconstruction_errors],
        "anomaly_flags": anomaly_flags.tolist(),
        "threshold_lower": round(float(threshold_lower), 2),
        "threshold_upper": round(float(threshold_upper), 2),
        "metrics": training_overall_metrics
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
        actual_vals = X_original[:, i]      # ← No more [:, -1, i]
        reconstructed_vals = recon_original[:, i]  # ← No more [:, -1, i]
        
        # Get per-feature thresholds AND metrics from training
        feature_thresholds = per_feature_thresholds_from_training.get(feature_name, {})
        
        # Find rank from feature_importance
        feature_rank = next(
            (item['rank'] for item in feature_importance if item['feature_name'] == feature_name),
            None
        )
        
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
            "timestamps": [str(ts) for ts in timestamps_rows],  # ← Changed from timestamps_sequences
            "metrics": feature_metrics
        })
    
    logger.info("   ✅ Plot data prepared")
    
    # ========================================================================
    # STEP 9: Return results
    # ========================================================================
    
    logger.info("=" * 80)
    logger.info("✅ SHAP EXPLANATION COMPLETED!")
    logger.info("=" * 80)
    
#     return {
#     "success": True,
#     "date_range": {
#         "start": start_date,
#         "end": end_date,
#         "total_rows": int(num_rows),  # ← Changed from total_sequences
#         "anomalies_found": int(num_anomalies),
#         "anomaly_percentage": round(float(num_anomalies/num_rows*100), 2)  # ← Changed denominator
#     },
#     "feature_importance": feature_importance,
#     "anomaly_detection_overall": plot_data_overall,
#     "plot_data": plot_data_features
# }

    return {
    "success": True,
    "is_anomaly": bool(num_anomalies > 0),  # ← NEW! True if any anomalies found
    "message": None,  # ← NEW! No error message for successful queries
    "date_range": {
        "start": start_date,
        "end": end_date,
        "total_rows": int(num_rows),
        "anomalies_found": int(num_anomalies),
        "anomaly_percentage": round(float(num_anomalies/num_rows*100), 2)
    },
    "feature_importance": feature_importance,
    "anomaly_detection_overall": plot_data_overall,
    "plot_data": plot_data_features
}