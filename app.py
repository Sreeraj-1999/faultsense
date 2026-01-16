from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, List
import pandas as pd
import numpy as np
import math
import logging
from sklearn.feature_selection import mutual_info_regression
import asyncpg
import asyncio
from scipy.stats import entropy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import json
import pickle
from datetime import datetime
import os
import json
import config
from models_and_utils import (
    LSTMTransformerSeq2Seq, TimeSeriesDataset,
    create_sequences, train_model, objective,
    generate_square_subsequent_mask
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
async def get_db_connection():
    """Create database connection"""
    try:
        # Use URL-encoded password: @ becomes %40, $ becomes %24
        # DATABASE_URL = "postgresql://memphis:Memphis%401234%24@192.168.18.175/obanext5"
        DATABASE_URL = config.DATABASE_URL
        logger.info(f"🔗 Attempting to connect to database at 192.168.18.175...")
        
        conn = await asyncio.wait_for(
            asyncpg.connect(DATABASE_URL), 
            timeout=10.0
        )
        logger.info("✅ Database connection successful!")
        return conn
    except asyncio.TimeoutError:
        logger.error("❌ Database connection timeout")
        raise HTTPException(status_code=500, detail="Database connection timeout")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
app = FastAPI()

# Pydantic models
class DataModel(BaseModel):
    id: int
    fk_equipmentmaster: int
    fk_subequipmentmaster: int
    fk_vessel: int
    tag_flag: bool
    name: Optional[str]
    tag_name: Optional[str]
    lower_limit: float
    upper_limit: float
    aggregation: str
    latex: Optional[str]

class InputModel(BaseModel):
    success: bool
    data: List[DataModel]  # ARRAY of DataModel objects
    metadata: str  # This is the IMO number

@app.post("/tag/correlation/")
async def tag_correlation(input_json: InputModel):
    logger.info("🚀 Starting mutual information analysis...")
    
    imo_number = input_json.metadata
    data_items = input_json.data
    logger.info(f"📊 Input received - IMO: {imo_number}, Number of items: {len(data_items)}")

    # Validate inputs
    logger.info("✅ Step 1: Validating inputs...")
    if not imo_number or not isinstance(imo_number, str):
        logger.error("❌ Invalid IMO number")
        raise HTTPException(status_code=400, detail="Invalid IMO number")
    
    if not data_items or len(data_items) == 0:
        logger.error("❌ No data items provided")
        raise HTTPException(status_code=400, detail="No data items provided")
    logger.info("✅ Input validation passed")

    # Read Excel file based on IMO number
    logger.info(f"✅ Step 2: Reading Excel file for IMO {imo_number}...")
    try:
        # excel_path = f"data/imo_{imo_number}.xlsx"
        # excel_path=r"C:\Users\User\Desktop\siemens\freya_schulte\imo_9665671_ME1_FMS_act_kgPh@AVG_dump.csv"
        # excel_path=r"C:\Users\User\Desktop\siemens\freya_schulte\training_data_averaged.csv"
        # excel_path = r"C:\Users\User\Desktop\siemens\freya_schulte\combined_vessels_15min.csv"
        excel_path = config.CSV_PATH_COMBINED
        df = pd.read_csv(excel_path)
        logger.info(f"✅ Excel file loaded with shape: {df.shape}")
        # ===== INSERT THIS SECTION HERE =====
        logger.info("⏱️  Step 2.1: Aggregating data to 1-hour intervals...")
        time_col = None
        if 'TI_UTC_act_ts@AVG' in df.columns:
            time_col = 'TI_UTC_act_ts@AVG'
        elif 'time' in df.columns:
            time_col = 'time'
        else:
            logger.error("❌ No time column found")
            raise HTTPException(status_code=400, detail="Time column not found in data")
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)

        
        columns_to_drop = []
        time_columns = ['time', 'TI_UTC_act_ts@AVG']
        # columns_to_drop.extend([col for col in time_columns if col in df.columns])
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                unique_values = df[col].dropna().unique()
                if len(unique_values) <= 2 and set(unique_values).issubset({0, 1, 0.0, 1.0}):
                    columns_to_drop.append(col)
        non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
        columns_to_drop.extend(non_numeric)

        # Remove duplicates and drop
        columns_to_drop = list(set(columns_to_drop))
        df = df.drop(columns=columns_to_drop)

        logger.info(f"🗑️  Dropped {len(columns_to_drop)} columns (time, boolean, non-numeric)")
        logger.info(f"✅ Reduced dataframe shape: {df.shape}")
        logger.info(f"🔍 ME_NO_1_TC_RPM exists: {'ME_NO_1_TC_RPM' in df.columns}")
        logger.info(f"🔍 ME_NO_2_TC_RPM exists: {'ME_NO_2_TC_RPM' in df.columns}")            
        # logger.info(f"✅ Available columns: {list(df.columns)}")
    except Exception as e:
        logger.error(f"❌ Error reading Excel file: {e}")
        raise HTTPException(status_code=400, detail=f"Error reading Excel file for IMO {imo_number}: {str(e)}")

    # Process each data item to create calculated columns if needed
    logger.info(f"✅ Step 3: Processing {len(data_items)} data items...")
    target_tags = []
    
    for i, item in enumerate(data_items, 1):
        logger.info(f"🔄 Processing item {i}/{len(data_items)}: ID {item.id}")
        
        # Check if it's a calculated column
        is_calculated_column = item.tag_name is None and item.name and item.latex
        
        if is_calculated_column:
            logger.info(f"   📊 Creating calculated column '{item.name}' from LaTeX...")
            try:
                # Parse LaTeX equation to extract tag names
                import re
                
                # Extract tag names from LaTeX (look for \text{tag_name} pattern)
                tag_pattern = r'\\text\{([^}]+)\}'
                found_tags = re.findall(tag_pattern, item.latex)
                logger.info(f"   📋 Found tags in equation: {found_tags}")
                
                # Check if all referenced tags exist in DataFrame
                missing_tags = [tag for tag in found_tags if tag not in df.columns]
                if missing_tags:
                    logger.error(f"❌ Missing tags in Excel data for item {item.id}: {missing_tags}")
                    raise HTTPException(status_code=404, detail=f"Tags not found in vessel data for item {item.id}: {missing_tags}")
                
                # Convert LaTeX to Python expression
                python_expression = item.latex
                
                # Replace LaTeX syntax with Python syntax
                python_expression = python_expression.replace('\\left(', '(')
                python_expression = python_expression.replace('\\right)', ')')
                python_expression = python_expression.replace('\\ ', ' ')
                python_expression = python_expression.replace('\\div', '/')
                python_expression = python_expression.replace('\\times', '*')
                python_expression = python_expression.replace('^', '**')
                
                # Replace \text{tag_name} with df['tag_name']
                for tag in found_tags:
                    python_expression = python_expression.replace(f'\\text{{{tag}}}', f"df['{tag}']")
                if '/' in python_expression:
                    logger.info(f"   🔧 Applying zero division fix...")
                    for tag in found_tags:
                        df[tag] = df[tag].replace(0, 0.001)
                    logger.info(f"   ✅ Zero division fix applied")
                
                logger.info(f"   🔄 Python expression: {python_expression}")
                
                # Calculate the new column
                df[item.name] = eval(python_expression)
                logger.info(f"   ✅ Calculated column '{item.name}' created successfully")
                
                # Add to target tags list
                target_tags.append(item.name)
                
            except Exception as e:
                logger.error(f"❌ Error creating calculated column for item {item.id}: {e}")
                raise HTTPException(status_code=400, detail=f"Error creating calculated column for item {item.id}: {str(e)}")
        
        else:
            # Regular existing tag
            if not item.tag_name:
                logger.error(f"❌ Item {item.id}: Neither tag_name nor valid name/latex provided")
                raise HTTPException(status_code=400, detail=f"Item {item.id}: Neither tag_name nor valid name/latex provided")
            
            logger.info(f"   📊 Using existing tag '{item.tag_name}'")
            
            # Check if tag exists in DataFrame
            if item.tag_name not in df.columns:
                logger.error(f"❌ Tag {item.tag_name} not found in Excel data for item {item.id}")
                raise HTTPException(status_code=404, detail=f"Tag {item.tag_name} not found in vessel data for item {item.id}")
            
            # Add to target tags list
            target_tags.append(item.tag_name)
    
    logger.info(f"✅ Step 3 completed. Target tags for analysis: {target_tags}")

    # Now run MI analysis for each target tag
    logger.info("✅ Step 4: Starting mutual information analysis for all target tags...")
    all_results = {}
    
    for tag_index, target_tag in enumerate(target_tags, 1):
        logger.info(f"🎯 Analyzing target tag {tag_index}/{len(target_tags)}: '{target_tag}'")
        
        # Ensure the target tag is numeric
        if not pd.api.types.is_numeric_dtype(df[target_tag]):
            logger.error(f"❌ Tag {target_tag} is not numeric. Type: {df[target_tag].dtype}")
            raise HTTPException(status_code=400, detail=f"Tag {target_tag} is not numeric")
        
        # Find numeric columns (excluding the current target tag and other target tags)
        other_targets = [t for t in target_tags if t != target_tag]
        numeric_columns = [col for col in df.columns 
                          if col != target_tag 
                          and pd.api.types.is_numeric_dtype(df[col])]
        
        logger.info(f"   📊 Found {len(numeric_columns)} numeric columns for analysis")
        
        
        # Compute mutual information with numeric columns
        mutual_info_results = {}
        skipped_columns = []
        processed_columns = []
        
        for i, col in enumerate(numeric_columns, 1):
            # logger.info(f"   📈 Analyzing column {i}/{len(numeric_columns)}: '{col}'")
            
            # Get data for both columns, dropping NaN values
            original_count = len(df)
            valid_data = df[[target_tag, col]].dropna()
            valid_count = len(valid_data)
            
            # Need at least 2 data points for mutual information
            if valid_count < 2:
                logger.warning(f"      ⚠️  Skipping '{col}': insufficient data points ({valid_count} < 2)")
                skipped_columns.append(f"{col}: insufficient data")
                continue
                
            # Check variance
            target_variance = valid_data[target_tag].var()
            col_variance = valid_data[col].var()
            
            # If target tag has zero variance, skip
            if target_variance == 0 or col_variance == 0:
                # logger.warning(f"      ⚠️  Skipping '{col}': target tag has zero variance")
                skipped_columns.append(f"{col}: target zero variance")
                continue
                
            try:
                # Prepare data for mutual information calculation
                X = valid_data[[col]].values  # Features (2D array required)
                y = valid_data[target_tag].values  # Target (1D array)
                
                # Calculate mutual information
                mi_score = mutual_info_regression(X, y, random_state=42)
                mi_value = mi_score[0]  # mutual_info_regression returns array
                
                # Check if MI is valid (not NaN or infinite)
                if pd.isna(mi_value) or math.isinf(mi_value):
                    logger.warning(f"      ⚠️  Skipping '{col}': invalid MI value")
                    skipped_columns.append(f"{col}: invalid MI")
                    continue
                if mi_value < config.MI_THRESHOLD: ##### ADDED
                    skipped_columns.append(f"{col}: weak correlation")
                    continue

                mi_score = round(float(mi_value), 4)
                            
                mutual_info_results[col] = mi_score
                
                processed_columns.append(col)
                # logger.info(f"      ✅ Successfully calculated: {mi_percentage}%")
                
            except Exception as e:
                logger.error(f"      ❌ Error calculating MI for '{col}': {e}")
                skipped_columns.append(f"{col}: calculation error")
                continue
        
        # mutual_info_results = dict(sorted(mutual_info_results.items(), key=lambda x: x[1], reverse=True)[:25])
        mutual_info_results = dict(sorted(mutual_info_results.items(), key=lambda x: x[1], reverse=True)[:config.TOP_N_CORRELATIONS])
        # Store results for this target tag
        all_results[target_tag] = {
            "correlations": mutual_info_results,
            "processed_columns": len(processed_columns),
            "skipped_columns": len(skipped_columns)
        }
        
        logger.info(f"   ✅ Completed analysis for '{target_tag}': {len(mutual_info_results)} correlations")

    # Final summary
    logger.info("✅ Step 5: All mutual information analysis completed!")
    logger.info(f"🎯 Analyzed {len(target_tags)} target tags")
    
    
    logger.info("✅ Step 6: Saving results to database...")
    try:
        conn = await get_db_connection()
        
        for i, item in enumerate(data_items):
            target_tag = target_tags[i]
            result = all_results[target_tag]
            
            # Transform correlations to the required format
            correlations_list = [
                {
                    "tag": tag_name,
                    "percentage": percentage,
                    "flag": False,# All flags are False as requested
                    "aggregation":"mean"
                }
                for tag_name, percentage in result["correlations"].items()
            ]
            
            # Create MlResponse JSON
            ml_response = {
                target_tag: {
                    "correlations": correlations_list,
                    "processed_columns": result["processed_columns"],
                    "skipped_columns": result["skipped_columns"]
                }
            }
            
            # Insert into database - using item.id as fk_efdmaster
            logger.info(f"   DEBUG: item.id = {item.id}")
            logger.info(f"   DEBUG: target_tag = {target_tag}")
            logger.info(f"   DEBUG: ml_response keys = {list(ml_response.keys())}")
            await conn.execute("""
                INSERT INTO "ImportantFeaturesMaster" ("fk_efdmaster", "MlResponse")
                VALUES ($1, $2)
            """, item.id, json.dumps(ml_response))
            
            logger.info(f"   💾 Saved results for item {item.id} ({target_tag})")
        
        await conn.close()
        logger.info("✅ All results saved to database successfully!")
        
    except Exception as e:
        logger.error(f"❌ Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    # Return success response
    return {
        "success": True,
        "imo_number": imo_number,
        "total_items": len(data_items),
        "target_tags": target_tags,
        "message": "Results saved to database successfully"
    }
@app.post("/tag/ml")
async def train_ml_model(request: dict):
    logger.info("🚀 Starting ML Model Training from API...")
    logger.info(f"📦 Received request keys: {list(request.keys())}") 
    if 'data' in request:
            # New structure: {'success': bool, 'data': {...}}
            ml_data = request['data']
    else:
        ml_data = request.get('ML_Data_sent', request)
    vessel_imo = ml_data['metadata']['vessel_imo']
    fk_vessel = ml_data['metadata']['fk_vessel']
    # equipment = ml_data['metadata']['equipment']
    equipment_db = ml_data['metadata']['equipment']  # Keep >> for database
    equipment_file = ml_data['metadata']['equipment'].replace('>>', '_')  # Use _ for file paths
    efd_array = ml_data['EFD']

        # Check ModelStatusMaster table
    conn = await get_db_connection()

    existing = await conn.fetchrow("""
        SELECT ml_ready, ts_response FROM "ModelStatusMaster" 
        WHERE fk_vessel = $1 AND equipment_hierarchy = $2
    """, fk_vessel, equipment_db)

    if existing:
        if existing['ml_ready']:
            await conn.close()
            logger.info("✅ Model already trained, skipping...")
            return {"success": True, "message": "Model already trained", "status": "skipped"}
        else:
            logger.info("🔄 ml_ready=False, retraining...")
    else:
        logger.info("➕ New entry, inserting into ModelStatusMaster...")
        await conn.execute("""
            INSERT INTO "ModelStatusMaster" (fk_vessel, equipment_hierarchy, ml_ready)
            VALUES ($1, $2, $3)
        """, fk_vessel, equipment_db, True)

    await conn.close()
    try:
        correlation_array = ml_data['correlation']
        logger.info(f"📊 Vessel IMO: {vessel_imo}, Equipment: {equipment_db}")
        
        logger.info(f"📊 Vessel IMO: {vessel_imo}")
        logger.info(f"📋 EFDs: {len(efd_array)}, Correlations: {len(correlation_array)}")
        
        # Load df
        # csv_path = r"C:\Users\User\Desktop\siemens\freya_schulte\training_data_averaged.csv"
        # csv_path = r"C:\Users\User\Desktop\siemens\freya_schulte\combined_vessels_15min.csv"
        csv_path = config.CSV_PATH_COMBINED
        df = pd.read_csv(csv_path)
        df = df.sort_values(['vessel_id', 'Local_time']).reset_index(drop=True)
        df['Local_time'] = pd.to_datetime(df['Local_time'], format='mixed', dayfirst=True)
        
        # Create calculated columns (COPY from Code 1)
        import re
        for efd in efd_array:
            if efd.get('tag_name') is None and efd.get('latex') is not None:
                logger.info(f"Creating calculated column '{efd['name']}'...")
                
                tag_pattern = r'\\text\{([^}]+)\}'
                found_tags = re.findall(tag_pattern, efd['latex'])
                
                missing_tags = [tag for tag in found_tags if tag not in df.columns]
                if missing_tags:
                    raise HTTPException(status_code=404, detail=f"Tags not found: {missing_tags}")
                
                python_expression = efd['latex']
                python_expression = python_expression.replace('\\left(', '(')
                python_expression = python_expression.replace('\\right)', ')')
                python_expression = python_expression.replace('\\ ', ' ')
                python_expression = python_expression.replace('\\div', '/')
                python_expression = python_expression.replace('\\times', '*')
                python_expression = python_expression.replace('^', '**')
                
                for tag in found_tags:
                    python_expression = python_expression.replace(f'\\text{{{tag}}}', f"df['{tag}']")
                
                # Zero division fix
                if '/' in python_expression:
                    logger.info(f"Applying zero division fix...")
                    for tag in found_tags:
                        df[tag] = df[tag].replace(0, 0.001)
                
                df[efd['name']] = eval(python_expression)
                logger.info(f"✅ Column '{efd['name']}' created")
        
        # Build feature lists
        load_feature = 'ME_Load@AVG'
        efd_features = [efd.get('tag_name') or efd.get('name') for efd in efd_array]
        # important_features = [c['tag'] for c in correlation_array if c['flag']]
        important_features = [c['tag'] for c in correlation_array if c['flag']]
        
        input_cols = [load_feature] + efd_features
        seen = set(input_cols)
        for feat in important_features:
            if feat not in seen:
                input_cols.append(feat)
                seen.add(feat)
        output_cols = [load_feature] + efd_features
        
        logger.info(f"✅ Input: {len(input_cols)}, Output: {len(output_cols)}")
        
        # Check columns exist
        missing = [col for col in input_cols + output_cols if col not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")
        
        # NOW continue with rest of training (vessel boundaries, sequences, etc.)
        # Get vessel boundaries
        vessel_groups = df.groupby('vessel_id').groups
        vessel_boundaries = []
        for vessel_id in sorted(vessel_groups.keys())[1:]:
            vessel_boundaries.append(vessel_groups[vessel_id][0])
        vessel_boundaries = sorted([b for b in vessel_boundaries if b > 0])
        
        # Create sequences
        # X, y, timestamps = create_sequences(
        #     df, input_cols, output_cols, 'Local_time',
        #     vessel_boundaries, 672, 96, 12 #24
        # )
        X, y, timestamps = create_sequences(df, input_cols, output_cols, 'Local_time', vessel_boundaries, config.LOOKBACK, config.HORIZON_SEQUENCE, config.STEP)
        
        logger.info(f"✅ Created {len(X)} sequences")

        logger.info("🔍 Checking for NaN...")
        nan_cols = []
        for col in input_cols + output_cols:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                nan_cols.append((col, nan_count))

        if nan_cols:
            df_before = len(df)
            df = df.dropna(subset=input_cols + output_cols)
            df_after = len(df)
            logger.warning(f"⚠️  Dropped {df_before - df_after} rows with NaN")
            # X, y, timestamps = create_sequences(df, input_cols, output_cols, 'Local_time', vessel_boundaries, 672, 96, 12) #24
            X, y, timestamps = create_sequences(df, input_cols, output_cols, 'Local_time', vessel_boundaries, config.LOOKBACK, config.HORIZON_SEQUENCE, config.STEP)
            logger.info(f"✅ Created {len(X)} sequences after cleaning")  

        if len(X) < 10:
            raise HTTPException(status_code=400, detail=f"Insufficient sequences: {len(X)}") 


        logger.info("🔀 Shuffling...")
        indices = np.arange(len(X))
        np.random.seed(42)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        timestamps = [timestamps[i] for i in indices]
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        timestamps_train = timestamps[:split_idx]
        timestamps_val = timestamps[split_idx:]
        logger.info(f"✅ Train: {len(X_train)}, Val: {len(X_val)}")
        
        # Scale
        logger.info("📊 Scaling...")
        input_scaler = StandardScaler()
        output_scaler = StandardScaler()
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
        y_train_reshaped = y_train.reshape(-1, y_train.shape[-1])
        y_val_reshaped = y_val.reshape(-1, y_val.shape[-1])
        X_train_scaled = input_scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
        X_val_scaled = input_scaler.transform(X_val_reshaped).reshape(X_val.shape)
        y_train_scaled = output_scaler.fit_transform(y_train_reshaped).reshape(y_train.shape)
        y_val_scaled = output_scaler.transform(y_val_reshaped).reshape(y_val.shape)
        logger.info("✅ Scaling complete")
        
        # Datasets
        train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled, timestamps_train)
        val_dataset = TimeSeriesDataset(X_val_scaled, y_val_scaled, timestamps_val)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🖥️  Device: {device}")
        
        # Optuna
        logger.info("🔍 Optuna tuning (30 trials)...")
        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: objective(trial, train_dataset, val_dataset, X_train.shape[-1], y_train.shape[-1], device),
            n_trials=1, show_progress_bar=True
        )
        best_params = study.best_params
        logger.info(f"✅ Best params: {best_params}")
        
        # Train
        logger.info("🏋️ Training final model...")
        final_model = LSTMTransformerSeq2Seq(
            num_layers=best_params['num_layers'], d_model=best_params['d_model'],
            num_heads=best_params['num_heads'], dff=best_params['dff'],
            input_dim=X_train.shape[-1], output_dim=y_train.shape[-1],
            dropout_rate=best_params['dropout']
        ).to(device)
        train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)
        criterion = nn.HuberLoss(delta=1.0)
        optimizer = optim.AdamW(final_model.parameters(), lr=best_params['lr'], weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        final_model, train_losses, val_losses, best_val_loss = train_model(
            final_model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=5, patience=15, device=device
        )
        
        # Save
        # model_dir = "models/combined_vessels"
        # Save
        model_dir = f"models/{vessel_imo}_{equipment_file}"
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
            "vessel_imo": vessel_imo, "input_features": input_cols, "output_features": output_cols,
            "input_dim": int(X_train.shape[-1]), "output_dim": int(y_train.shape[-1]),
            "lookback": config.LOOKBACK, "horizon": config.HORIZON_METADATA, "step": config.STEP, "best_params": best_params,
            "best_val_loss": float(best_val_loss), "final_train_loss": float(train_losses[-1]),
            "final_val_loss": float(val_losses[-1]), "num_epochs": len(train_losses),
            "training_date": str(datetime.now()), "num_sequences": int(len(X)),
            "train_sequences": int(len(X_train)), "val_sequences": int(len(X_val))
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        efd_definitions_path = f"{model_dir}/efd_definitions.json"
        with open(efd_definitions_path, 'w') as f:
            json.dump(efd_array, f, indent=2)    
        logger.info("🎉 Training completed!")

        logger.info("📊 Generating predictions for validation set...")
        final_model.eval()
        all_predictions = []
        all_actuals = []

        with torch.no_grad():
            for src, tgt, _ in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                
                batch_size = src.size(0)
                horizon = tgt.size(1)
                output_dim = tgt.size(2)
                
                enc_output = final_model.encoder(src)
                tgt_input = torch.zeros(batch_size, horizon, output_dim).to(device)
                tgt_input[:, 0:1, :] = src[:, -1:, :output_dim]
                
                for t in range(1, horizon):
                    mask = generate_square_subsequent_mask(t, device)
                    dec_output = final_model.decoder(tgt_input[:, :t, :], enc_output, tgt_mask=mask)
                    tgt_input[:, t:t+1, :] = dec_output[:, -1:, :]
                
                mask = generate_square_subsequent_mask(horizon, device)
                output = final_model.decoder(tgt_input, enc_output, tgt_mask=mask)
                
                all_predictions.append(output.cpu().numpy())
                all_actuals.append(tgt.cpu().numpy())

        predictions = np.concatenate(all_predictions)
        actuals = np.concatenate(all_actuals)

        # Inverse transform to original scale
        predictions_reshaped = predictions.reshape(-1, predictions.shape[-1])
        actuals_reshaped = actuals.reshape(-1, actuals.shape[-1])

        predictions_original = output_scaler.inverse_transform(predictions_reshaped).reshape(predictions.shape)
        actuals_original = output_scaler.inverse_transform(actuals_reshaped).reshape(actuals.shape)
        logger.info("📊 Calculating metrics for each feature...")

        # Create per-feature results
        results = []
        for i, feature_name in enumerate(output_cols):
            actual_flat = actuals_original[:, :, i].flatten()
            predicted_flat = predictions_original[:, :, i].flatten()
            
            rmse = np.sqrt(mean_squared_error(actual_flat, predicted_flat))
            mae = mean_absolute_error(actual_flat, predicted_flat)
            r2 = r2_score(actual_flat, predicted_flat)
            
            logger.info(f"📈 {feature_name}:")
            logger.info(f"   RMSE: {rmse:.4f}")
            logger.info(f"   MAE:  {mae:.4f}")
            logger.info(f"   R²:   {r2:.4f}")
            results.append({
                "feature_name": feature_name,
                "actual": actual_flat.tolist(),
                "predicted": predicted_flat.tolist(),
                "timestamps": [str(ts) for ts in timestamps_val],
                "metrics": {
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "r2": float(r2)
                }
            })

        logger.info("✅ Predictions generated successfully!")
        for result in results:
            sorted_indices = np.argsort(result['timestamps'])
            result['timestamps'] = [result['timestamps'][i] for i in sorted_indices]
            result['actual'] = [result['actual'][i] for i in sorted_indices]
            result['predicted'] = [result['predicted'][i] for i in sorted_indices]
        conn = await get_db_connection()
        ml_response_data = {
            "plot_data": results,
            "error_message": None
        }
        await conn.execute("""
            UPDATE "ModelStatusMaster" 
            SET ts_response = $1, ml_ready = $2
            WHERE fk_vessel = $3 AND equipment_hierarchy = $4
        """, json.dumps(ml_response_data), True, fk_vessel, equipment_db)
        await conn.close()
        autoencoder_results = {}
        logger.info("")
        logger.info("=" * 100)
        logger.info("🤖 STARTING AUTOENCODER TRAINING FOR ANOMALY DETECTION")
        logger.info("=" * 100)

        try:
            from autoencoder import train_autoencoder
            
            # Train autoencoder with same scaled data used for TS model
            autoencoder_results = train_autoencoder(
                X_train_scaled=X_train_scaled,
                X_val_scaled=X_val_scaled,
                device=device,
                save_dir=model_dir,
                feature_names=input_cols,
                input_scaler=input_scaler,
                timestamps_train=timestamps_train,  # ← ADD
                timestamps_val=timestamps_val        # ← ADD
            )
            logger.info("💾 Saving autoencoder results to database...")
            conn = await get_db_connection()
            
            
            # ae_response_data = {
            #     "plot_data": autoencoder_results['plot_data'],  # ← ADD THIS (same as TS model!)
            #     "error_message": None
            # }
            ae_response_data = {
            "plot_data": autoencoder_results['plot_data'],
            "anomaly_detection_overall": autoencoder_results['anomaly_detection_overall'],
            "error_message": None
        }

            await conn.execute("""
                UPDATE "ModelStatusMaster" 
                SET ae_response = $1
                WHERE fk_vessel = $2 AND equipment_hierarchy = $3
            """, json.dumps(ae_response_data), fk_vessel, equipment_db)

            await conn.close()
            logger.info("✅ Autoencoder results saved to database!")


            logger.info("")
            logger.info("=" * 100)
            logger.info("✅ AUTOENCODER TRAINING COMPLETED!")
            logger.info("=" * 100)
            logger.info(f"📊 Validation Metrics:")
            logger.info(f"   R²:   {autoencoder_results['r2']:.6f}")
            logger.info(f"   RMSE: {autoencoder_results['rmse']:.6f}")
            logger.info(f"   MAE:  {autoencoder_results['mae']:.6f}")
            logger.info("=" * 100)
            logger.info("")
        
        # Future: Add threshold formula and anomaly detection here
        # threshold = mean(reconstruction_errors) + 1.5 * std(reconstruction_errors)
        
        except Exception as e:
            logger.error(f"❌ Autoencoder training failed: {e}")
            
            conn = await get_db_connection()
            ae_response_data = {"error_message": str(e)}
            await conn.execute("""
                UPDATE "ModelStatusMaster" 
                SET ae_response = $1
                WHERE fk_vessel = $2 AND equipment_hierarchy = $3
            """, json.dumps(ae_response_data), fk_vessel, equipment_db)
            await conn.close()
            
            logger.error(f"   This won't affect TS model training (already completed)")
            logger.info("")
        logger.info("")
        logger.info("=" * 100)
        logger.info("🧹 STARTING FFN TRAINING ON CLEAN DATA")
        logger.info("=" * 100)
        
        try:
            from reference_model import train_clean_data_ffn
            # Check if autoencoder results are available
            if not autoencoder_results or 'anomaly_detection_overall' not in autoencoder_results:
                raise ValueError("Autoencoder results not available - cannot train FFN without clean data flags")
            
            all_actual_ffn = np.concatenate([
                input_scaler.inverse_transform(X_train_scaled[:, -1, :]),
                input_scaler.inverse_transform(X_val_scaled[:, -1, :])
            ], axis=0)
            
            anomaly_flags_ffn = autoencoder_results['anomaly_detection_overall']['anomaly_flags']
            
            # Get timestamps (combined train + val, matching anomaly_flags order)
            all_timestamps_ffn = timestamps_train + timestamps_val
            
            # Filter timestamps to keep only clean sequences
            clean_mask = ~np.array(anomaly_flags_ffn)
            timestamps_clean_ffn = [ts for i, ts in enumerate(all_timestamps_ffn) if clean_mask[i]]
            
            # Feature names
            input_feature_names_ffn = [load_feature] + important_features
            output_feature_names_ffn = efd_features
            
            # Train FFN
            ffn_results = train_clean_data_ffn(
                all_actual_data=all_actual_ffn,
                anomaly_flags=anomaly_flags_ffn,
                input_feature_names=input_feature_names_ffn,
                output_feature_names=output_feature_names_ffn,
                all_feature_names=input_cols,
                timestamps_clean=timestamps_clean_ffn,  # ← ADD THIS
                save_dir=model_dir,
                device=device
            )
            
            logger.info("")
            logger.info("=" * 100)
            logger.info("✅ FFN TRAINING COMPLETED!")
            logger.info("=" * 100)
            logger.info(f"📊 Clean sequences used: {ffn_results['num_clean_sequences']}")
            logger.info(f"📁 Excel saved: {ffn_results['excel_path']}")
            logger.info(f"🎯 Best val loss: {ffn_results['best_val_loss']:.6f}")
            logger.info("=" * 100)
            logger.info("")
            # Save FFN results to database
            logger.info("💾 Saving FFN results to database (ref_response)...")
            conn = await get_db_connection()
            
            ref_response_data = {
                "plot_data": ffn_results['plot_data'],
                "error_message": None
            }
            
            await conn.execute("""
                UPDATE "ModelStatusMaster" 
                SET ref_response = $1
                WHERE fk_vessel = $2 AND equipment_hierarchy = $3
            """, json.dumps(ref_response_data), fk_vessel, equipment_db)
            
            await conn.close()
            logger.info("✅ FFN results saved to ref_response!")
            
        except Exception as e:
            logger.error(f"❌ FFN training failed: {e}")
            logger.error(f"   This won't affect TS/AE models (already completed)")
            # Save error to database
            try:
                conn = await get_db_connection()
                ref_response_data = {"error_message": str(e)}
                await conn.execute("""
                    UPDATE "ModelStatusMaster" 
                    SET ref_response = $1
                    WHERE fk_vessel = $2 AND equipment_hierarchy = $3
                """, json.dumps(ref_response_data), fk_vessel, equipment_db)
                await conn.close()
            except:
                pass                                  
            
        
        return {
            "success": True, "message": "Training completed",
            "model_path": model_path, "best_val_loss": float(best_val_loss),
            "sequences": len(X), "input_features": len(input_cols), "output_features": len(output_cols),"plot_data": results ,
            "autoencoder_r2": autoencoder_results.get('r2', None)
        }
    except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            conn = await get_db_connection()
            ml_response_data = {"error_message": str(e)}
            await conn.execute("""
                UPDATE "ModelStatusMaster" 
                SET ts_response = $1, ml_ready = $2
                WHERE fk_vessel = $3 AND equipment_hierarchy = $4
            """, json.dumps(ml_response_data), False, fk_vessel, equipment_db)
            await conn.close()
            raise HTTPException(status_code=500, detail=str(e))
@app.post("/autoencoder/explain")
async def explain_autoencoder_anomalies_endpoint(request: dict):
    """
    Use SHAP to explain which features contribute to anomalies in a date range
    
    Request body:
    {
        "start_date": "2025-08-05",
        "end_date": "2025-10-06",
        "vessel_imo": "9665669",
        "equipment": "ME_CYL"
    }
    """
    logger.info("🔍 Starting SHAP anomaly explanation...")
    
    try:
        # Extract parameters
        start_date = request.get('start_date')
        end_date = request.get('end_date')
        vessel_imo = request.get('vessel_imo')
        equipment = request.get('equipment')
        
        # Validate
        if not all([start_date, end_date, vessel_imo, equipment]):
            raise HTTPException(status_code=400, detail="Missing required parameters")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Call SHAP explainer
        from shap_explainer import explain_autoencoder_anomalies
        
        results = explain_autoencoder_anomalies(
            start_date=start_date,
            end_date=end_date,
            vessel_imo=vessel_imo,
            equipment=equipment,
            device=device
        )
        
        return results
        
    except Exception as e:
        logger.error(f"❌ SHAP explanation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # uvicorn.run(app, host="192.168.18.206", port=8000)
    uvicorn.run(app, host=config.SERVER_HOST, port=config.SERVER_PORT)        