#!/usr/bin/env python3
"""
NBEATS Balance Prediction - Complete Production Pipeline
======================================================

This script provides a complete end-to-end balance prediction system using NBEATSx
with comprehensive null handling, SQLite-based hyperparameter optimization, and
detailed visualizations.

Author: Balance Prediction Team
Date: July 2025
Version: 1.0 (Production Ready)
"""

# =============================================================================
# BLOCK 1: IMPORTS AND DEPENDENCIES
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import os
import time
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Machine Learning and Forecasting
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATSx
from neuralforecast.losses.pytorch import DistributionLoss
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Hyperparameter Optimization
import optuna
from optuna.storages import RDBStorage
from optuna.trial import Trial

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("🚀 NBEATS BALANCE PREDICTION SYSTEM")
print("="*80)
print("✅ All dependencies imported successfully")
print("✅ Production-ready pipeline initialized")
print("="*80)

# =============================================================================
# BLOCK 2: DATA LOADING AND INITIAL INSPECTION
# =============================================================================

def load_and_inspect_data(file_path="nbeatx_final_functions/featured_shihara.xlsx"):
    """
    Load balance data and perform initial inspection.
    
    Args:
        file_path (str): Path to the Excel file containing balance data
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    
    print("\n📂 LOADING AND INSPECTING DATA")
    print("="*60)
    
    try:
        # Load data
        df = pd.read_excel(file_path)
        logger.info(f"Data loaded successfully from {file_path}")
        
        # Basic information
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📅 Columns: {list(df.columns)}")
        
        # Data types
        print(f"\n📈 Data types:")
        for col, dtype in df.dtypes.items():
            print(f"   {col}: {dtype}")
        
        # Date range analysis
        if 'Date' in df.columns:
            print(f"\n📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
            print(f"📅 Total days: {(df['Date'].max() - df['Date'].min()).days}")
        
        # Target variable analysis
        if 'Normalized_Balance' in df.columns:
            target = df['Normalized_Balance']
            print(f"\n💰 Balance statistics:")
            print(f"   Min: {target.min():.4f}")
            print(f"   Max: {target.max():.4f}")
            print(f"   Mean: {target.mean():.4f}")
            print(f"   Std: {target.std():.4f}")
        
        print(f"\n✅ Data loading complete: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        print(f"❌ Error loading data: {str(e)}")
        return None

# =============================================================================
# BLOCK 3: COMPREHENSIVE NULL VALUE HANDLING
# =============================================================================

def detect_null_patterns(df, verbose=True):
    """
    Analyze null value patterns in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        verbose (bool): Print detailed analysis
        
    Returns:
        dict: Null pattern analysis results
    """
    
    if verbose:
        print("\n🔍 NULL PATTERN ANALYSIS")
        print("="*50)
    
    analysis = {
        'null_by_column': df.isnull().sum(),
        'total_nulls': df.isnull().sum().sum(),
        'null_percentage': (df.isnull().sum().sum() / df.size) * 100,
        'problematic_columns': []
    }
    
    # Check each column
    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        
        if null_count > 0:
            analysis['problematic_columns'].append({
                'column': col,
                'null_count': null_count,
                'null_percentage': null_pct
            })
            
            if verbose:
                print(f"⚠️  {col}: {null_count} nulls ({null_pct:.2f}%)")
    
    if verbose:
        if analysis['total_nulls'] == 0:
            print("✅ No null values found in dataset")
        else:
            print(f"\n📊 Total nulls: {analysis['total_nulls']}")
            print(f"📊 Overall null percentage: {analysis['null_percentage']:.2f}%")
    
    return analysis

def handle_null_values(df, method='auto', verbose=True):
    """
    Comprehensive null value handling for balance prediction dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        method (str): Handling method ('auto', 'conservative', 'aggressive')
        verbose (bool): Print detailed information
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    
    if verbose:
        print("\n🛠️ COMPREHENSIVE NULL VALUE HANDLING")
        print("="*60)
    
    df_cleaned = df.copy()
    original_nulls = df_cleaned.isnull().sum().sum()
    
    # Column categorization
    date_columns = [col for col in df_cleaned.columns if 'date' in col.lower()]
    target_columns = [col for col in df_cleaned.columns if 'normalized_balance' in col.lower() or col.lower() == 'y']
    lag_features = [col for col in df_cleaned.columns if 'ago' in col.lower() or 'lag' in col.lower()]
    rolling_features = [col for col in df_cleaned.columns if 'rolling' in col.lower() or ('mean' in col.lower() and 'd' in col.lower())]
    categorical_features = [col for col in df_cleaned.columns if df_cleaned[col].dtype == 'object' and col not in date_columns]
    
    if verbose:
        print(f"📋 Column categorization:")
        print(f"   📅 Date columns: {date_columns}")
        print(f"   🎯 Target columns: {target_columns}")
        print(f"   ⏮️ Lag features: {lag_features}")
        print(f"   📊 Rolling features: {rolling_features}")
        print(f"   🏷️ Categorical features: {categorical_features}")
    
    # Handle each category
    
    # 1. Date columns - Remove rows with null dates
    for col in date_columns:
        if df_cleaned[col].isnull().any():
            before_count = len(df_cleaned)
            df_cleaned = df_cleaned.dropna(subset=[col])
            removed = before_count - len(df_cleaned)
            if verbose:
                print(f"📅 {col}: Removed {removed} rows with null dates")
    
    # 2. Target columns - Linear interpolation + statistical fill
    for col in target_columns:
        if df_cleaned[col].isnull().any():
            null_count = df_cleaned[col].isnull().sum()
            if verbose:
                print(f"🎯 {col}: Handling {null_count} nulls with interpolation")
            
            # Linear interpolation
            df_cleaned[col] = df_cleaned[col].interpolate(method='linear')
            # Fill remaining with forward/backward fill
            df_cleaned[col] = df_cleaned[col].fillna(method='ffill').fillna(method='bfill')
            # Final fallback to mean
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
    
    # 3. Lag features - Forward fill (logical for lags)
    for col in lag_features:
        if df_cleaned[col].isnull().any():
            null_count = df_cleaned[col].isnull().sum()
            if verbose:
                print(f"⏮️ {col}: Handling {null_count} nulls with forward fill")
            
            df_cleaned[col] = df_cleaned[col].fillna(method='ffill')
            df_cleaned[col] = df_cleaned[col].fillna(method='bfill')
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
    
    # 4. Rolling features - Interpolation + median
    for col in rolling_features:
        if df_cleaned[col].isnull().any():
            null_count = df_cleaned[col].isnull().sum()
            if verbose:
                print(f"📊 {col}: Handling {null_count} nulls with interpolation + median")
            
            df_cleaned[col] = df_cleaned[col].interpolate(method='linear')
            if 'std' in col.lower():
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
            else:
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
    
    # 5. Categorical features - Mode fill
    for col in categorical_features:
        if df_cleaned[col].isnull().any():
            null_count = df_cleaned[col].isnull().sum()
            if verbose:
                print(f"🏷️ {col}: Handling {null_count} nulls with mode")
            
            mode_value = df_cleaned[col].mode()
            if len(mode_value) > 0:
                df_cleaned[col] = df_cleaned[col].fillna(mode_value[0])
            else:
                df_cleaned[col] = df_cleaned[col].fillna('unknown')
    
    # 6. Remaining numeric columns - Interpolation + mean
    remaining_cols = [col for col in df_cleaned.columns 
                     if col not in date_columns + target_columns + lag_features + rolling_features + categorical_features
                     and df_cleaned[col].dtype in ['int64', 'float64']]
    
    for col in remaining_cols:
        if df_cleaned[col].isnull().any():
            null_count = df_cleaned[col].isnull().sum()
            if verbose:
                print(f"🔢 {col}: Handling {null_count} nulls with interpolation + mean")
            
            df_cleaned[col] = df_cleaned[col].interpolate(method='linear')
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
    
    # Final validation
    final_nulls = df_cleaned.isnull().sum().sum()
    
    if verbose:
        print(f"\n✅ NULL HANDLING COMPLETE")
        print(f"📊 Original nulls: {original_nulls}")
        print(f"📊 Final nulls: {final_nulls}")
        print(f"📊 Rows preserved: {len(df_cleaned)}/{len(df)} ({(len(df_cleaned)/len(df)*100):.1f}%)")
        
        if final_nulls > 0:
            print(f"⚠️ Warning: {final_nulls} nulls remain")
        else:
            print("🎉 All nulls successfully handled!")
    
    return df_cleaned

# =============================================================================
# BLOCK 4: DATA SPLITTING AND PREPARATION
# =============================================================================

def prepare_train_test_split(df, test_days=30, validation_days=15, verbose=True):
    """
    Split data into train, validation, and test sets for time series.
    
    Args:
        df (pd.DataFrame): Cleaned input dataframe
        test_days (int): Number of days for test set
        validation_days (int): Number of days for validation set
        verbose (bool): Print split information
        
    Returns:
        tuple: (train_df, val_df, test_df, split_info)
    """
    
    if verbose:
        print("\n📊 PREPARING TRAIN/VALIDATION/TEST SPLIT")
        print("="*60)
    
    # Ensure data is sorted by date
    df_sorted = df.sort_values('Date').reset_index(drop=True)
    
    # Calculate split indices
    total_days = len(df_sorted)
    test_start = total_days - test_days
    val_start = test_start - validation_days
    
    # Create splits
    train_df = df_sorted.iloc[:val_start].copy()
    val_df = df_sorted.iloc[val_start:test_start].copy()
    test_df = df_sorted.iloc[test_start:].copy()
    
    split_info = {
        'total_days': total_days,
        'train_days': len(train_df),
        'val_days': len(val_df),
        'test_days': len(test_df),
        'train_date_range': (train_df['Date'].min(), train_df['Date'].max()),
        'val_date_range': (val_df['Date'].min(), val_df['Date'].max()),
        'test_date_range': (test_df['Date'].min(), test_df['Date'].max())
    }
    
    if verbose:
        print(f"📈 Training set: {len(train_df)} days ({len(train_df)/total_days*100:.1f}%)")
        print(f"   📅 Date range: {split_info['train_date_range'][0]} to {split_info['train_date_range'][1]}")
        
        print(f"📊 Validation set: {len(val_df)} days ({len(val_df)/total_days*100:.1f}%)")
        print(f"   📅 Date range: {split_info['val_date_range'][0]} to {split_info['val_date_range'][1]}")
        
        print(f"🧪 Test set: {len(test_df)} days ({len(test_df)/total_days*100:.1f}%)")
        print(f"   📅 Date range: {split_info['test_date_range'][0]} to {split_info['test_date_range'][1]}")
        
        print(f"\n✅ Data split complete: {len(train_df)} + {len(val_df)} + {len(test_df)} = {total_days} days")
    
    return train_df, val_df, test_df, split_info

def prepare_neuralforecast_data(train_df, val_df, test_df, verbose=True):
    """
    Convert dataframes to NeuralForecast format.
    
    Args:
        train_df, val_df, test_df: Split dataframes
        verbose (bool): Print conversion info
        
    Returns:
        tuple: (train_data, val_data, test_data, feature_lists)
    """
    
    if verbose:
        print("\n🔄 CONVERTING TO NEURALFORECAST FORMAT")
        print("="*60)
    
    # Convert to NeuralForecast format
    def convert_to_nf_format(df):
        nf_df = df.copy()
        nf_df['unique_id'] = 'balance'
        nf_df = nf_df.rename(columns={'Date': 'ds', 'Normalized_Balance': 'y'})
        return nf_df
    
    train_data = convert_to_nf_format(train_df)
    val_data = convert_to_nf_format(val_df)
    test_data = convert_to_nf_format(test_df)
    
    # Identify available features
    potential_future = ['dayofweek_sin', 'dayofweek_cos']
    potential_historical = [
        'balance_changed', 'balance_1d_ago', 'balance_7d_ago', 
        'balance_30d_ago', 'rolling_mean_30d', 'rolling_std_30d'
    ]
    
    future_features = [feat for feat in potential_future if feat in train_data.columns]
    historical_features = [feat for feat in potential_historical if feat in train_data.columns]
    
    feature_lists = {
        'future_features': future_features,
        'historical_features': historical_features
    }
    
    if verbose:
        print(f"🔧 Future features: {future_features}")
        print(f"🔧 Historical features: {historical_features}")
        print(f"✅ NeuralForecast format conversion complete")
    
    return train_data, val_data, test_data, feature_lists

# =============================================================================
# BLOCK 5: SQLITE-BASED HYPERPARAMETER OPTIMIZATION
# =============================================================================

def setup_sqlite_storage(storage_path="./optuna_balance_study.db", study_name="nbeats_balance_optimization"):
    """
    Setup SQLite storage for Optuna hyperparameter optimization.
    
    Args:
        storage_path (str): Path to SQLite database
        study_name (str): Name of the optimization study
        
    Returns:
        tuple: (storage, study)
    """
    
    print("\n💾 SETTING UP SQLITE STORAGE FOR HYPERPARAMETER OPTIMIZATION")
    print("="*70)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(storage_path)), exist_ok=True)
    
    # Configure SQLite storage
    storage_url = f"sqlite:///{storage_path}"
    engine_kwargs = {
        'connect_args': {
            'timeout': 300,  # 5 minutes timeout
            'check_same_thread': False,
            'isolation_level': None  # Autocommit mode
        },
        'pool_pre_ping': True,
        'pool_recycle': 1800  # 30 minutes
    }
    
    try:
        # Create storage
        storage = RDBStorage(
            url=storage_url,
            engine_kwargs=engine_kwargs,
            heartbeat_interval=60,
            grace_period=120
        )
        
        # Create or load study
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction='minimize',
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        print(f"✅ SQLite storage created: {storage_path}")
        print(f"✅ Study '{study_name}' initialized")
        print(f"📊 Previous trials: {len(study.trials)}")
        
        return storage, study
        
    except Exception as e:
        logger.error(f"SQLite storage setup failed: {str(e)}")
        print(f"❌ SQLite setup failed: {str(e)}")
        print("🔄 Falling back to in-memory storage...")
        
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        return None, study

def hyperparameter_optimization(train_data, val_data, feature_lists, 
                               horizon=30, n_trials=20, 
                               storage_path="./optuna_balance_study.db",
                               study_name="nbeats_balance_optimization"):
    """
    Perform hyperparameter optimization using Optuna with SQLite storage.
    
    Args:
        train_data, val_data: Training and validation datasets
        feature_lists (dict): Future and historical feature lists
        horizon (int): Forecast horizon
        n_trials (int): Number of optimization trials
        storage_path (str): SQLite database path
        study_name (str): Study name
        
    Returns:
        tuple: (best_parameters, best_model, optimization_results)
    """
    
    print("\n🎯 HYPERPARAMETER OPTIMIZATION WITH SQLITE STORAGE")
    print("="*70)
    
    # Setup storage
    storage, study = setup_sqlite_storage(storage_path, study_name)
    
    # Feature lists
    future_features = feature_lists['future_features']
    historical_features = feature_lists['historical_features']
    
    # Store best model globally
    best_model = None
    best_rmse = float('inf')
    
    def create_future_features(df, horizon):
        """Create future features for prediction"""
        last_date = df['ds'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')
        
        future_df = pd.DataFrame({'ds': future_dates})
        future_df['dayofweek_sin'] = np.sin(2 * np.pi * future_df['ds'].dt.dayofweek / 7)
        future_df['dayofweek_cos'] = np.cos(2 * np.pi * future_df['ds'].dt.dayofweek / 7)
        future_df['unique_id'] = 'balance'
        
        # Handle any nulls
        for col in future_df.columns:
            if future_df[col].isnull().any():
                future_df[col] = future_df[col].fillna(0)
        
        return future_df
    
    def objective(trial: Trial) -> float:
        """Objective function for optimization"""
        nonlocal best_model, best_rmse
        
        trial_start = time.time()
        
        try:
            # Hyperparameter space
            input_size = trial.suggest_int('input_size', 100, 200)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            max_steps = trial.suggest_int('max_steps', 500, 1500)
            batch_size = trial.suggest_int('batch_size', 16, 48)
            n_blocks = [trial.suggest_int(f'n_blocks_{i}', 2, 4) for i in range(3)]
            n_harmonics = trial.suggest_int('n_harmonics', 1, 3)
            n_polynomials = trial.suggest_int('n_polynomials', 1, 3)
            
            # Create model
            model = NBEATSx(
                h=horizon,
                input_size=input_size,
                futr_exog_list=future_features,
                hist_exog_list=historical_features,
                random_seed=42,
                scaler_type='standard',
                learning_rate=learning_rate,
                max_steps=max_steps,
                batch_size=batch_size,
                stack_types=['identity', 'trend', 'seasonality'],
                n_blocks=n_blocks,
                n_harmonics=n_harmonics,
                n_polynomials=n_polynomials,
                loss=DistributionLoss(distribution='Normal', level=[80, 95])
            )
            
            # Create and train forecaster
            forecaster = NeuralForecast(models=[model], freq='D')
            forecaster.fit(df=train_data)
            
            # Generate predictions on validation set
            future_df = create_future_features(train_data, len(val_data))
            forecast_df = forecaster.predict(futr_df=future_df)
            
            # Calculate RMSE
            actual = val_data['y'].values
            predicted = forecast_df['NBEATSx'].values[:len(actual)]
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            
            # Store best model
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = forecaster
                print(f"✅ New best RMSE: {rmse:.6f} (Trial {trial.number})")
            
            trial_time = time.time() - trial_start
            print(f"⏱️ Trial {trial.number}/{n_trials} completed in {trial_time:.1f}s - RMSE: {rmse:.6f}")
            
            return rmse
            
        except Exception as e:
            print(f"❌ Trial {trial.number} failed: {str(e)}")
            return float('inf')
    
    # Run optimization
    print(f"\n🚀 Starting optimization with {n_trials} trials...")
    start_time = time.time()
    
    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=3600,  # 1 hour timeout
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        print("⏹️ Optimization interrupted by user")
    except Exception as e:
        print(f"⚠️ Optimization error: {e}")
    
    optimization_time = time.time() - start_time
    
    # Get best parameters
    if len(study.trials) > 0:
        best_params = study.best_params.copy()
        n_blocks = [best_params[f'n_blocks_{i}'] for i in range(3)]
        best_params['n_blocks'] = n_blocks
        
        # Remove individual n_blocks parameters
        for i in range(3):
            best_params.pop(f'n_blocks_{i}', None)
        
        # Results summary
        results = {
            'best_rmse': study.best_value,
            'best_trial_number': study.best_trial.number,
            'total_trials': len(study.trials),
            'optimization_time': optimization_time,
            'study_name': study_name,
            'storage_path': storage_path
        }
        
        print("\n" + "="*70)
        print("🎉 HYPERPARAMETER OPTIMIZATION COMPLETE")
        print("="*70)
        print(f"📊 Total trials: {len(study.trials)}")
        print(f"🏆 Best RMSE: {study.best_value:.6f}")
        print(f"⚡ Best trial: #{study.best_trial.number}")
        print(f"⏱️ Total time: {optimization_time:.1f}s")
        print(f"💾 Results saved to: {storage_path}")
        print("\n📋 Best hyperparameters:")
        for param, value in best_params.items():
            print(f"   {param}: {value}")
        print("="*70)
        
        return best_params, best_model, results
    
    else:
        print("❌ No successful trials completed")
        return None, None, None

# =============================================================================
# BLOCK 6: MODEL TRAINING AND EVALUATION
# =============================================================================

def train_final_model(train_data, best_params, feature_lists, horizon=30):
    """
    Train the final model with best hyperparameters.
    
    Args:
        train_data: Training dataset
        best_params (dict): Best hyperparameters from optimization
        feature_lists (dict): Feature specifications
        horizon (int): Forecast horizon
        
    Returns:
        NeuralForecast: Trained forecaster
    """
    
    print("\n🏋️ TRAINING FINAL MODEL WITH BEST PARAMETERS")
    print("="*60)
    
    # Extract features
    future_features = feature_lists['future_features']
    historical_features = feature_lists['historical_features']
    
    # Create model with best parameters
    model = NBEATSx(
        h=horizon,
        input_size=best_params['input_size'],
        futr_exog_list=future_features,
        hist_exog_list=historical_features,
        random_seed=42,
        scaler_type='standard',
        learning_rate=best_params['learning_rate'],
        max_steps=best_params['max_steps'],
        batch_size=best_params['batch_size'],
        stack_types=['identity', 'trend', 'seasonality'],
        n_blocks=best_params['n_blocks'],
        n_harmonics=best_params['n_harmonics'],
        n_polynomials=best_params['n_polynomials'],
        loss=DistributionLoss(distribution='Normal', level=[80, 95])
    )
    
    # Create and train forecaster
    forecaster = NeuralForecast(models=[model], freq='D')
    
    print("🔄 Training model...")
    start_time = time.time()
    forecaster.fit(df=train_data)
    training_time = time.time() - start_time
    
    print(f"✅ Model training complete in {training_time:.1f}s")
    print(f"📊 Training data: {len(train_data)} samples")
    print(f"🎯 Model ready for forecasting")
    
    return forecaster

def evaluate_model_performance(forecaster, train_data, test_data, feature_lists, horizon=30):
    """
    Evaluate model performance on test set.
    
    Args:
        forecaster: Trained forecaster
        train_data, test_data: Training and test datasets
        feature_lists (dict): Feature specifications
        horizon (int): Forecast horizon
        
    Returns:
        dict: Evaluation results
    """
    
    print("\n📈 EVALUATING MODEL PERFORMANCE")
    print("="*60)
    
    def create_test_features(train_data, test_data):
        """Create features for test period"""
        future_df = test_data[['ds', 'unique_id']].copy()
        
        # Add time features
        future_df['dayofweek_sin'] = np.sin(2 * np.pi * future_df['ds'].dt.dayofweek / 7)
        future_df['dayofweek_cos'] = np.cos(2 * np.pi * future_df['ds'].dt.dayofweek / 7)
        
        # Handle nulls
        for col in future_df.columns:
            if future_df[col].isnull().any():
                future_df[col] = future_df[col].fillna(0)
        
        return future_df
    
    # Generate predictions
    print("🔮 Generating predictions...")
    future_df = create_test_features(train_data, test_data)
    forecast_df = forecaster.predict(futr_df=future_df)
    
    # Calculate metrics
    actual_values = test_data['y'].values
    predicted_values = forecast_df['NBEATSx'].values
    
    # Ensure same length
    min_len = min(len(actual_values), len(predicted_values))
    actual_values = actual_values[:min_len]
    predicted_values = predicted_values[:min_len]
    
    # Calculate metrics
    mae = mean_absolute_error(actual_values, predicted_values)
    mse = mean_squared_error(actual_values, predicted_values)
    rmse = np.sqrt(mse)
    
    # Additional metrics
    mape = np.mean(np.abs((actual_values - predicted_values) / np.maximum(np.abs(actual_values), 1e-8))) * 100
    correlation = np.corrcoef(actual_values, predicted_values)[0, 1] if len(actual_values) > 1 else 0
    r2 = r2_score(actual_values, predicted_values)
    
    # Residual analysis
    residuals = actual_values - predicted_values
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)
    
    results = {
        'metrics': {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'correlation': correlation,
            'r2_score': r2
        },
        'residuals': {
            'mean': residual_mean,
            'std': residual_std,
            'values': residuals
        },
        'predictions': {
            'actual': actual_values,
            'predicted': predicted_values,
            'dates': test_data['ds'].values[:min_len]
        },
        'forecast_df': forecast_df
    }
    
    # Print results
    print("📊 PERFORMANCE METRICS:")
    print(f"   MAE:         {mae:.6f}")
    print(f"   RMSE:        {rmse:.6f}")
    print(f"   MAPE:        {mape:.2f}%")
    print(f"   Correlation: {correlation:.6f}")
    print(f"   R² Score:    {r2:.6f}")
    
    print(f"\n📊 RESIDUAL ANALYSIS:")
    print(f"   Mean:        {residual_mean:.6f}")
    print(f"   Std Dev:     {residual_std:.6f}")
    
    # Quality assessment
    if rmse < 0.1:
        quality = "Excellent"
    elif rmse < 0.2:
        quality = "Good"
    elif rmse < 0.5:
        quality = "Fair"
    else:
        quality = "Needs Improvement"
    
    print(f"\n🎯 MODEL QUALITY: {quality}")
    print("="*60)
    
    return results

# =============================================================================
# BLOCK 7: COMPREHENSIVE VISUALIZATION
# =============================================================================

def create_comprehensive_visualizations(evaluation_results, optimization_results=None, 
                                       save_dir="./visualizations"):
    """
    Create comprehensive visualizations for the balance prediction results.
    
    Args:
        evaluation_results (dict): Model evaluation results
        optimization_results (dict): Hyperparameter optimization results
        save_dir (str): Directory to save plots
    """
    
    print("\n📊 CREATING COMPREHENSIVE VISUALIZATIONS")
    print("="*60)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Actual vs Predicted Time Series Plot
    print("📈 Creating actual vs predicted plot...")
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    dates = evaluation_results['predictions']['dates']
    actual = evaluation_results['predictions']['actual']
    predicted = evaluation_results['predictions']['predicted']
    
    # Plot time series
    ax.plot(dates, actual, label='Actual Balance', linewidth=2, color='#2E86AB', alpha=0.8)
    ax.plot(dates, predicted, label='Predicted Balance', linewidth=2, color='#A23B72', alpha=0.8)
    
    # Add confidence intervals if available
    if 'NBEATSx-lo-95' in evaluation_results['forecast_df'].columns:
        lo_95 = evaluation_results['forecast_df']['NBEATSx-lo-95'].values[:len(dates)]
        hi_95 = evaluation_results['forecast_df']['NBEATSx-hi-95'].values[:len(dates)]
        ax.fill_between(dates, lo_95, hi_95, alpha=0.2, color='#A23B72', label='95% Confidence Interval')
    
    # Formatting
    ax.set_title('Balance Prediction: Actual vs Predicted', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Normalized Balance', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add metrics text box
    metrics = evaluation_results['metrics']
    textstr = f"""Performance Metrics:
RMSE: {metrics['rmse']:.6f}
MAE: {metrics['mae']:.6f}
MAPE: {metrics['mape']:.2f}%
Correlation: {metrics['correlation']:.6f}
R²: {metrics['r2_score']:.6f}"""
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/actual_vs_predicted.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/actual_vs_predicted.png")
    
    # 2. Residual Analysis Plot
    print("📊 Creating residual analysis plot...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    residuals = evaluation_results['residuals']['values']
    
    # Residuals over time
    ax1.plot(dates, residuals, color='#F18F01', alpha=0.7)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.set_title('Residuals Over Time')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Residuals')
    ax1.grid(True, alpha=0.3)
    
    # Residual histogram
    ax2.hist(residuals, bins=30, color='#C73E1D', alpha=0.7, edgecolor='black')
    ax2.set_title('Residual Distribution')
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot (Normal Distribution)')
    ax3.grid(True, alpha=0.3)
    
    # Predicted vs Residuals
    ax4.scatter(predicted, residuals, alpha=0.6, color='#2E86AB')
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax4.set_title('Predicted vs Residuals')
    ax4.set_xlabel('Predicted Values')
    ax4.set_ylabel('Residuals')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/residual_analysis.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/residual_analysis.png")
    
    # 3. Scatter Plot: Actual vs Predicted
    print("🎯 Creating scatter plot...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot
    ax.scatter(actual, predicted, alpha=0.6, s=50, color='#2E86AB')
    
    # Perfect prediction line
    min_val = min(min(actual), min(predicted))
    max_val = max(max(actual), max(predicted))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Calculate and display R²
    r2 = metrics['r2_score']
    ax.text(0.05, 0.95, f'R² = {r2:.6f}', transform=ax.transAxes, fontsize=14,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('Actual Balance', fontsize=12)
    ax.set_ylabel('Predicted Balance', fontsize=12)
    ax.set_title('Actual vs Predicted Balance (Scatter Plot)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/scatter_actual_vs_predicted.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/scatter_actual_vs_predicted.png")
    
    # 4. Model Performance Summary
    print("📋 Creating performance summary...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Create summary text
    summary_text = f"""
    NBEATS BALANCE PREDICTION - MODEL PERFORMANCE SUMMARY
    =====================================================
    
    📊 DATASET INFORMATION:
    • Test Period: {len(actual)} days
    • Date Range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}
    
    📈 PERFORMANCE METRICS:
    • Root Mean Square Error (RMSE): {metrics['rmse']:.6f}
    • Mean Absolute Error (MAE): {metrics['mae']:.6f}
    • Mean Absolute Percentage Error (MAPE): {metrics['mape']:.2f}%
    • Correlation Coefficient: {metrics['correlation']:.6f}
    • R² Score: {metrics['r2_score']:.6f}
    
    📊 RESIDUAL STATISTICS:
    • Residual Mean: {evaluation_results['residuals']['mean']:.6f}
    • Residual Std Dev: {evaluation_results['residuals']['std']:.6f}
    
    🎯 MODEL QUALITY ASSESSMENT:
    """
    
    # Add quality assessment
    if metrics['rmse'] < 0.1:
        summary_text += "• Overall Quality: EXCELLENT ✅\n"
        summary_text += "• Model shows high accuracy and reliability\n"
    elif metrics['rmse'] < 0.2:
        summary_text += "• Overall Quality: GOOD ✅\n"
        summary_text += "• Model performs well for most predictions\n"
    elif metrics['rmse'] < 0.5:
        summary_text += "• Overall Quality: FAIR ⚠️\n"
        summary_text += "• Model shows moderate performance\n"
    else:
        summary_text += "• Overall Quality: NEEDS IMPROVEMENT ❌\n"
        summary_text += "• Consider parameter tuning or feature engineering\n"
    
    if optimization_results:
        summary_text += f"""
    ⚙️ OPTIMIZATION DETAILS:
    • Total Trials: {optimization_results['total_trials']}
    • Best Trial: #{optimization_results['best_trial_number']}
    • Optimization Time: {optimization_results['optimization_time']:.1f} seconds
    • Storage: {optimization_results['storage_path']}
    """
    
    summary_text += f"""
    
    📅 Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/performance_summary.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/performance_summary.png")
    
    # Close all plots to free memory
    plt.close('all')
    
    print(f"\n🎉 All visualizations saved to: {save_dir}")
    print("📊 Generated plots:")
    print("   • actual_vs_predicted.png - Time series comparison")
    print("   • residual_analysis.png - Comprehensive residual analysis")
    print("   • scatter_actual_vs_predicted.png - Scatter plot with R²")
    print("   • performance_summary.png - Complete model summary")
    print("="*60)

# =============================================================================
# BLOCK 8: MAIN EXECUTION PIPELINE
# =============================================================================

def main_pipeline(data_file="nbeatx_final_functions/featured_shihara.xlsx", 
                 n_trials=20, 
                 test_days=30, 
                 horizon=30):
    """
    Execute the complete NBEATS balance prediction pipeline.
    
    Args:
        data_file (str): Path to input data file
        n_trials (int): Number of hyperparameter optimization trials
        test_days (int): Number of days for test set
        horizon (int): Forecast horizon
    """
    
    print("🚀 STARTING COMPLETE NBEATS BALANCE PREDICTION PIPELINE")
    print("="*80)
    print(f"📂 Data file: {data_file}")
    print(f"🎯 Optimization trials: {n_trials}")
    print(f"🧪 Test days: {test_days}")
    print(f"🔮 Forecast horizon: {horizon}")
    print("="*80)
    
    pipeline_start = time.time()
    
    try:
        # BLOCK 1: Load and inspect data
        df = load_and_inspect_data(data_file)
        if df is None:
            return None
        
        # BLOCK 2: Handle null values
        null_analysis = detect_null_patterns(df)
        df_clean = handle_null_values(df, method='auto', verbose=True)
        
        # BLOCK 3: Split data
        train_df, val_df, test_df, split_info = prepare_train_test_split(
            df_clean, test_days=test_days, validation_days=15, verbose=True
        )
        
        # BLOCK 4: Prepare for NeuralForecast
        train_data, val_data, test_data, feature_lists = prepare_neuralforecast_data(
            train_df, val_df, test_df, verbose=True
        )
        
        # BLOCK 5: Hyperparameter optimization
        best_params, best_model, optimization_results = hyperparameter_optimization(
            train_data, val_data, feature_lists,
            horizon=horizon, n_trials=n_trials
        )
        
        if best_params is None:
            print("❌ Hyperparameter optimization failed")
            return None
        
        # BLOCK 6: Evaluate model (use pre-trained model from optimization)
        evaluation_results = evaluate_model_performance(
            best_model, train_data, test_data, feature_lists, horizon=horizon
        )
        
        # BLOCK 7: Create visualizations
        create_comprehensive_visualizations(
            evaluation_results, optimization_results, save_dir="./visualizations"
        )
        
        # Pipeline summary
        pipeline_time = time.time() - pipeline_start
        
        print("\n" + "="*80)
        print("🎉 PIPELINE EXECUTION COMPLETE")
        print("="*80)
        print(f"⏱️ Total execution time: {pipeline_time:.1f} seconds")
        print(f"📊 Final RMSE: {evaluation_results['metrics']['rmse']:.6f}")
        print(f"📊 Final R²: {evaluation_results['metrics']['r2_score']:.6f}")
        print(f"💾 Results saved to: ./visualizations/")
        print(f"💾 Optimization data: ./optuna_balance_study.db")
        
        # Return comprehensive results
        pipeline_results = {
            'data_info': {
                'original_shape': df.shape,
                'clean_shape': df_clean.shape,
                'split_info': split_info
            },
            'null_analysis': null_analysis,
            'best_parameters': best_params,
            'optimization_results': optimization_results,
            'evaluation_results': evaluation_results,
            'execution_time': pipeline_time
        }
        
        print("✅ All results compiled and ready")
        print("="*80)
        
        return pipeline_results
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        print(f"❌ Pipeline failed: {str(e)}")
        return None

# =============================================================================
# BLOCK 9: SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    Execute the complete pipeline when script is run directly.
    """
    
    print("🎬 EXECUTING NBEATS BALANCE PREDICTION SYSTEM")
    print("="*80)
    
    # Configuration
    CONFIG = {
        'data_file': "nbeatx_final_functions/featured_shihara.xlsx",
        'n_trials': 15,  # Adjust based on time constraints
        'test_days': 30,
        'horizon': 30
    }
    
    print("📋 CONFIGURATION:")
    for key, value in CONFIG.items():
        print(f"   {key}: {value}")
    print("="*80)
    
    # Execute pipeline
    results = main_pipeline(**CONFIG)
    
    if results:
        print("\n🎊 SUCCESS! Balance prediction pipeline completed successfully!")
        print("📁 Check the following outputs:")
        print("   • ./visualizations/ - All generated plots")
        print("   • ./optuna_balance_study.db - Hyperparameter optimization results")
        print("   • Console output - Detailed performance metrics")
    else:
        print("\n❌ Pipeline execution failed. Check error messages above.")
    
    print("\n" + "="*80)
    print("🏁 SCRIPT EXECUTION COMPLETE")
    print("="*80)

# =============================================================================
# END OF SCRIPT
# =============================================================================
