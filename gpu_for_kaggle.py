#!/usr/bin/env python3
"""
NBEATS Balance Prediction - GPU Optimized for Kaggle
===================================================

GPU-optimized version of N-BEATS forecasting specifically designed for Kaggle notebooks.
Features:
- Automatic GPU detection and optimization
- Mixed precision training (FP16) for faster training
- Memory-efficient batch processing
- Comprehensive hyperparameter optimization
- Detailed performance monitoring
- Production-ready forecasting pipeline

Author: Balance Prediction Team
Date: January 2025
Version: 2.0 (Kaggle GPU Optimized)
"""

# =============================================================================
# IMPORTS AND SETUP
# =============================================================================

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import json
import os
from datetime import datetime, timedelta
import logging

# Machine Learning and Forecasting
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATSx
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Hyperparameter Optimization
import optuna
from optuna.trial import Trial

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("🚀 NBEATS GPU-OPTIMIZED BALANCE PREDICTION FOR KAGGLE")
print("="*80)
print("✅ All dependencies imported successfully")
print("✅ GPU optimization pipeline initialized")
print("="*80)

# =============================================================================
# GPU SETUP AND CONFIGURATION
# =============================================================================

def setup_gpu():
    """
    Setup GPU device with comprehensive optimization for Kaggle.
    
    Returns:
        torch.device: Configured device (GPU or CPU)
    """
    print("\n🔧 GPU SETUP AND CONFIGURATION")
    print("="*60)
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_compute_capability = torch.cuda.get_device_capability(0)
        
        print(f"✅ CUDA available: {gpu_name}")
        print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
        print(f"🔧 Compute Capability: {gpu_compute_capability[0]}.{gpu_compute_capability[1]}")
        print(f"🎯 Using device: {device}")
        
        # GPU optimizations for Kaggle
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Memory management
        torch.cuda.empty_cache()
        
        print("✅ GPU optimizations applied:")
        print("   - cuDNN benchmark enabled")
        print("   - TF32 precision enabled")
        print("   - Memory cache cleared")
        
    else:
        device = torch.device('cpu')
        print("⚠️  CUDA not available, using CPU")
        print(f"🎯 Using device: {device}")
        print("💡 For better performance, enable GPU in Kaggle settings")
    
    print("="*60)
    return device

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_data(file_path="shihara_train_30days_version2.xlsx"):
    """
    Load and preprocess balance data.
    
    Args:
        file_path (str): Path to the data file
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    print("\n📂 LOADING AND PREPROCESSING DATA")
    print("="*60)
    
    try:
        # Load data
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file format. Use .xlsx or .csv")
        
        # Convert date column
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
        
        # Handle missing values
        if df['Normalized_Balance'].isnull().sum() > 0:
            print(f"⚠️  Found {df['Normalized_Balance'].isnull().sum()} missing values")
            df['Normalized_Balance'] = df['Normalized_Balance'].fillna(method='ffill')
            print("✅ Missing values filled using forward fill")
        
        print(f"✅ Data loaded successfully: {df.shape}")
        print(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"💰 Balance range: {df['Normalized_Balance'].min():.4f} to {df['Normalized_Balance'].max():.4f}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

def prepare_data_for_forecasting(df, train_ratio=0.8):
    """
    Prepare data for NeuralForecast format.
    
    Args:
        df (pd.DataFrame): Input dataframe
        train_ratio (float): Ratio for train/validation split
        
    Returns:
        tuple: (train_df, val_df) in NeuralForecast format
    """
    print("\n🔄 PREPARING DATA FOR FORECASTING")
    print("="*60)
    
    # Convert to NeuralForecast format
    nf_df = df[['Date', 'Normalized_Balance']].copy()
    nf_df.columns = ['ds', 'y']
    nf_df['unique_id'] = 'balance'
    
    # Split data
    split_idx = int(len(nf_df) * train_ratio)
    train_df = nf_df.iloc[:split_idx].copy()
    val_df = nf_df.iloc[split_idx:].copy()
    
    print(f"📊 Training samples: {len(train_df)}")
    print(f"📊 Validation samples: {len(val_df)}")
    print(f"📊 Total samples: {len(nf_df)}")
    
    return train_df, val_df

# =============================================================================
# GPU-OPTIMIZED MODEL CONFIGURATION
# =============================================================================

def get_gpu_optimized_config(device, trial=None):
    """
    Get GPU-optimized configuration for NBEATS model.
    
    Args:
        device (torch.device): GPU device
        trial (optuna.Trial): Optuna trial for hyperparameter optimization
        
    Returns:
        dict: Model configuration
    """
    if trial is None:
        # Default configuration
        config = {
            'input_size': 120,
            'h': 30,
            'learning_rate': 0.001,
            'max_steps': 1000,
            'batch_size': 64,
            'num_blocks': [2, 2],
            'num_layers': [2, 2],
            'num_hidden': 128,
            'dropout': 0.1,
            'random_state': 42,
        }
    else:
        # Optimization configuration
        config = {
            'input_size': trial.suggest_int('input_size', 60, 200),
            'h': 30,
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'max_steps': trial.suggest_int('max_steps', 500, 1500),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'num_hidden': trial.suggest_int('num_hidden', 64, 256),
            'dropout': trial.suggest_float('dropout', 0.0, 0.3),
            'num_blocks': [2, 2],
            'num_layers': [2, 2],
            'random_state': 42,
        }
    
    # GPU optimizations
    if device.type == 'cuda':
        config.update({
            'accelerator': 'gpu',
            'devices': [0],
            'precision': '16-mixed',  # Mixed precision for faster training
            'gradient_clip_val': 0.1,
            'enable_progress_bar': True,
            'enable_model_summary': False,  # Disable for faster training
        })
    else:
        config.update({
            'accelerator': 'cpu',
            'precision': '32',
            'enable_progress_bar': True,
        })
    
    return config

# =============================================================================
# HYPERPARAMETER OPTIMIZATION
# =============================================================================

def objective_function(trial, train_df, val_df, device):
    """
    Objective function for hyperparameter optimization.
    
    Args:
        trial (optuna.Trial): Optuna trial
        train_df (pd.DataFrame): Training data
        val_df (pd.DataFrame): Validation data
        device (torch.device): GPU device
        
    Returns:
        float: Validation RMSE
    """
    # Clear GPU cache
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    try:
        # Get configuration
        config = get_gpu_optimized_config(device, trial)
        
        # Create and train model
        model = NBEATSx(**config)
        forecaster = NeuralForecast(models=[model], freq='D')
        
        start_time = time.time()
        forecaster.fit(train_df)
        training_time = time.time() - start_time
        
        # Generate predictions
        predictions = forecaster.predict(h=30)
        
        # Calculate metrics
        actual = val_df['y'].values[:30]
        predicted = predictions['NBEATSx'].values[:30]
        
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        
        print(f"Trial {trial.number:3d}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, Time={training_time:.1f}s")
        
        return rmse
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return float('inf')

def run_hyperparameter_optimization(train_df, val_df, device, n_trials=20, timeout=1800):
    """
    Run hyperparameter optimization using Optuna.
    
    Args:
        train_df (pd.DataFrame): Training data
        val_df (pd.DataFrame): Validation data
        device (torch.device): GPU device
        n_trials (int): Number of optimization trials
        timeout (int): Timeout in seconds
        
    Returns:
        optuna.Study: Optimization study with results
    """
    print(f"\n🔬 HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    print(f"🎯 Trials: {n_trials}")
    print(f"⏱️  Timeout: {timeout} seconds")
    print(f"🎯 Device: {device}")
    
    # Create study
    study_name = "nbeats_gpu_kaggle_optimization"
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        load_if_exists=True
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective_function(trial, train_df, val_df, device),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )
    
    print(f"\n✅ Optimization completed!")
    print(f"🎯 Best RMSE: {study.best_value:.4f}")
    print(f"🔧 Best parameters: {study.best_params}")
    
    return study

# =============================================================================
# FINAL MODEL TRAINING
# =============================================================================

def train_final_model(df, best_params, device):
    """
    Train final model with best parameters.
    
    Args:
        df (pd.DataFrame): Full dataset
        best_params (dict): Best hyperparameters
        device (torch.device): GPU device
        
    Returns:
        tuple: (forecaster, forecast, training_time)
    """
    print(f"\n🎯 TRAINING FINAL MODEL")
    print("="*60)
    
    # Prepare full dataset
    nf_df = df[['Date', 'Normalized_Balance']].copy()
    nf_df.columns = ['ds', 'y']
    nf_df['unique_id'] = 'balance'
    
    # Get final configuration
    final_config = get_gpu_optimized_config(device)
    final_config.update(best_params)
    
    # Clear cache
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    print("🚀 Starting final training...")
    start_time = time.time()
    
    # Create and train model
    model = NBEATSx(**final_config)
    forecaster = NeuralForecast(models=[model], freq='D')
    
    forecaster.fit(nf_df)
    training_time = time.time() - start_time
    
    print(f"✅ Training completed in {training_time:.1f} seconds")
    
    # Generate forecast
    print("🔮 Generating 30-day forecast...")
    forecast = forecaster.predict(h=30)
    
    return forecaster, forecast, training_time

# =============================================================================
# VISUALIZATION AND RESULTS
# =============================================================================

def create_comprehensive_visualizations(df, forecast, training_time, device, study=None):
    """
    Create comprehensive visualizations of results.
    
    Args:
        df (pd.DataFrame): Historical data
        forecast (pd.DataFrame): Forecast results
        training_time (float): Training time
        device (torch.device): GPU device
        study (optuna.Study): Optimization study
    """
    print(f"\n📊 CREATING VISUALIZATIONS")
    print("="*60)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Main time series plot
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(df['Date'], df['Normalized_Balance'], label='Historical', 
             color='blue', alpha=0.7, linewidth=1.5)
    
    forecast_dates = pd.date_range(start=df['Date'].max() + timedelta(days=1), 
                                  periods=30, freq='D')
    ax1.plot(forecast_dates, forecast['NBEATSx'], label='Forecast', 
             color='red', linewidth=2.5, linestyle='--')
    
    ax1.set_title('Balance Time Series with 30-Day Forecast (GPU Optimized)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Normalized Balance', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. Device and performance info
    ax2 = fig.add_subplot(gs[0, 2])
    device_info = f"""
    🎯 Device: {device}
    🚀 GPU: {device.type == 'cuda'}
    ⏱️  Training: {training_time:.1f}s
    📊 Samples: {len(df)}
    """
    ax2.text(0.5, 0.5, device_info, ha='center', va='center', 
             transform=ax2.transAxes, fontsize=11, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    ax2.set_title('Performance Information', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # 3. Forecast distribution
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(forecast['NBEATSx'], bins=15, alpha=0.7, color='skyblue', 
             edgecolor='black', linewidth=1)
    ax3.set_title('Forecast Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Forecasted Balance', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # 4. Training performance
    ax4 = fig.add_subplot(gs[1, 1])
    metrics = ['Training Time (s)', 'GPU Available']
    values = [training_time, 1 if device.type == 'cuda' else 0]
    colors = ['#ff9999', '#66b3ff']
    bars = ax4.bar(metrics, values, color=colors, alpha=0.8)
    ax4.set_title('Training Performance', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Value', fontsize=11)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.1f}', ha='center', va='bottom', fontsize=10)
    
    # 5. Optimization results (if available)
    if study is not None:
        ax5 = fig.add_subplot(gs[1, 2])
        trials_df = study.trials_dataframe()
        if len(trials_df) > 0:
            ax5.plot(trials_df['number'], trials_df['value'], 'o-', 
                     color='green', alpha=0.7, linewidth=2)
            ax5.set_title('Optimization Progress', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Trial Number', fontsize=11)
            ax5.set_ylabel('RMSE', fontsize=11)
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No optimization data', ha='center', va='center',
                     transform=ax5.transAxes, fontsize=11)
            ax5.set_title('Optimization Progress', fontsize=12, fontweight='bold')
            ax5.axis('off')
    
    # 6. Forecast vs Historical comparison
    ax6 = fig.add_subplot(gs[2, :])
    historical_stats = df['Normalized_Balance'].describe()
    forecast_stats = forecast['NBEATSx'].describe()
    
    comparison_data = pd.DataFrame({
        'Historical': [historical_stats['mean'], historical_stats['std'], 
                      historical_stats['min'], historical_stats['max']],
        'Forecast': [forecast_stats['mean'], forecast_stats['std'], 
                    forecast_stats['min'], forecast_stats['max']]
    }, index=['Mean', 'Std', 'Min', 'Max'])
    
    comparison_data.plot(kind='bar', ax=ax6, alpha=0.8)
    ax6.set_title('Historical vs Forecast Statistics', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Value', fontsize=11)
    ax6.legend(fontsize=11)
    ax6.grid(True, alpha=0.3)
    plt.xticks(rotation=0)
    
    # Save plot
    plt.tight_layout()
    plt.savefig('gpu_kaggle_forecasting_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("💾 Visualization saved to gpu_kaggle_forecasting_results.png")

def save_results(forecast, best_params, training_time, device, study=None):
    """
    Save all results to files.
    
    Args:
        forecast (pd.DataFrame): Forecast results
        best_params (dict): Best parameters
        training_time (float): Training time
        device (torch.device): GPU device
        study (optuna.Study): Optimization study
    """
    print(f"\n💾 SAVING RESULTS")
    print("="*60)
    
    # Save forecast
    forecast.to_csv('gpu_kaggle_forecast_30days.csv', index=False)
    print("✅ Forecast saved to gpu_kaggle_forecast_30days.csv")
    
    # Save parameters
    results = {
        'best_parameters': best_params,
        'training_time': training_time,
        'device': str(device),
        'gpu_available': device.type == 'cuda',
        'timestamp': datetime.now().isoformat()
    }
    
    with open('gpu_kaggle_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("✅ Results saved to gpu_kaggle_results.json")
    
    # Save optimization study if available
    if study is not None:
        study_df = study.trials_dataframe()
        study_df.to_csv('gpu_kaggle_optimization_trials.csv', index=False)
        print("✅ Optimization trials saved to gpu_kaggle_optimization_trials.csv")

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def main():
    """
    Main execution function for GPU-optimized NBEATS forecasting.
    """
    print("🚀 STARTING GPU-OPTIMIZED NBEATS FORECASTING")
    print("="*80)
    
    try:
        # 1. Setup GPU
        device = setup_gpu()
        
        # 2. Load and prepare data
        df = load_data()
        train_df, val_df = prepare_data_for_forecasting(df)
        
        # 3. Run hyperparameter optimization
        study = run_hyperparameter_optimization(
            train_df, val_df, device, 
            n_trials=15,  # Reduced for Kaggle
            timeout=900   # 15 minutes timeout
        )
        
        # 4. Train final model
        final_forecaster, final_forecast, training_time = train_final_model(
            df, study.best_params, device
        )
        
        # 5. Create visualizations
        create_comprehensive_visualizations(
            df, final_forecast, training_time, device, study
        )
        
        # 6. Save results
        save_results(final_forecast, study.best_params, training_time, device, study)
        
        # 7. Final summary
        print(f"\n🎉 FINAL SUMMARY")
        print("="*60)
        print(f"🎯 Device: {device}")
        print(f"🚀 GPU Available: {device.type == 'cuda'}")
        print(f"⏱️  Training Time: {training_time:.1f} seconds")
        print(f"🔧 Best RMSE: {study.best_value:.4f}")
        print(f"📊 Forecast Range: {final_forecast['NBEATSx'].min():.4f} to {final_forecast['NBEATSx'].max():.4f}")
        print("="*60)
        print("🎉 GPU-OPTIMIZED NBEATS FORECASTING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return final_forecaster, final_forecast, study
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        logger.error(f"Main execution failed: {e}")
        raise

# =============================================================================
# KAGGLE-SPECIFIC HELPER FUNCTIONS
# =============================================================================

def check_kaggle_environment():
    """
    Check if running in Kaggle environment and print relevant info.
    """
    print("\n🔍 KAGGLE ENVIRONMENT CHECK")
    print("="*60)
    
    # Check for Kaggle environment
    is_kaggle = os.path.exists('/kaggle/input')
    print(f"📊 Running in Kaggle: {is_kaggle}")
    
    if is_kaggle:
        print("✅ Kaggle environment detected")
        print("💡 GPU should be available if enabled in settings")
    else:
        print("⚠️  Not running in Kaggle environment")
        print("💡 Some optimizations may not be available")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"🚀 GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  GPU not available")
        print("💡 Enable GPU in Kaggle notebook settings for better performance")
    
    print("="*60)

def quick_forecast(data_file="shihara_train_30days_version2.xlsx", n_trials=5):
    """
    Quick forecast function for rapid testing in Kaggle.
    
    Args:
        data_file (str): Path to data file
        n_trials (int): Number of optimization trials
    """
    print("⚡ QUICK FORECAST MODE")
    print("="*60)
    
    # Setup
    device = setup_gpu()
    df = load_data(data_file)
    train_df, val_df = prepare_data_for_forecasting(df)
    
    # Quick optimization
    study = run_hyperparameter_optimization(
        train_df, val_df, device, 
        n_trials=n_trials, 
        timeout=300  # 5 minutes
    )
    
    # Train and forecast
    forecaster, forecast, training_time = train_final_model(df, study.best_params, device)
    
    # Save results
    forecast.to_csv('quick_forecast.csv', index=False)
    print("✅ Quick forecast saved to quick_forecast.csv")
    
    return forecaster, forecast

# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Check environment
    check_kaggle_environment()
    
    # Run main pipeline
    main() 