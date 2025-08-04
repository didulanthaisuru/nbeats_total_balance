#!/usr/bin/env python3
"""
NBEATS Balance Prediction - GPU Optimized Version
================================================

GPU-optimized version of N-BEATS forecasting with CUDA support.
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time
import json
from datetime import datetime, timedelta

# Machine Learning and Forecasting
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATSx
from sklearn.metrics import mean_squared_error

# Hyperparameter Optimization
import optuna
from optuna.trial import Trial

warnings.filterwarnings('ignore')

# =============================================================================
# GPU SETUP
# =============================================================================

def setup_gpu():
    """Setup GPU device with fallback to CPU"""
    print("🔧 GPU SETUP AND CONFIGURATION")
    print("="*50)
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ CUDA available: {gpu_name} ({gpu_memory:.1f} GB)")
        print(f"🎯 Using device: {device}")
        
        # Set memory management
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
    else:
        device = torch.device('cpu')
        print("⚠️  CUDA not available, using CPU")
        print(f"🎯 Using device: {device}")
    
    print("="*50)
    return device

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(file_path="shihara_train_30days_version2.xlsx"):
    """Load and preprocess data"""
    print("📂 LOADING DATA")
    print("="*50)
    
    df = pd.read_excel(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"✅ Data loaded: {df.shape}")
    print(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
    return df

# =============================================================================
# GPU-OPTIMIZED MODEL
# =============================================================================

def create_gpu_model(device):
    """Create GPU-optimized NBEATS model"""
    print("🔧 CREATING GPU-OPTIMIZED MODEL")
    print("="*50)
    
    # Base configuration
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
    
    # GPU optimizations
    if device.type == 'cuda':
        config.update({
            'accelerator': 'gpu',
            'devices': [0],
            'precision': '16-mixed',
            'gradient_clip_val': 0.1,
        })
        print("✅ GPU optimizations applied")
    else:
        config.update({
            'accelerator': 'cpu',
            'precision': '32',
        })
        print("✅ CPU configuration applied")
    
    return config

# =============================================================================
# HYPERPARAMETER OPTIMIZATION
# =============================================================================

def prepare_data(df):
    """Prepare data for NeuralForecast"""
    nf_df = df[['Date', 'Normalized_Balance']].copy()
    nf_df.columns = ['ds', 'y']
    nf_df['unique_id'] = 'balance'
    
    split_idx = int(len(nf_df) * 0.8)
    train_df = nf_df.iloc[:split_idx].copy()
    val_df = nf_df.iloc[split_idx:].copy()
    
    return train_df, val_df

def objective(trial: Trial, train_df, val_df, device):
    """Objective function for optimization"""
    
    # Clear GPU cache
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    params = {
        'input_size': trial.suggest_int('input_size', 60, 200),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'max_steps': trial.suggest_int('max_steps', 500, 1500),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        'num_hidden': trial.suggest_int('num_hidden', 64, 256),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3),
    }
    
    # GPU parameters
    if device.type == 'cuda':
        params.update({
            'accelerator': 'gpu',
            'devices': [0],
            'precision': '16-mixed',
        })
    else:
        params.update({
            'accelerator': 'cpu',
            'precision': '32',
        })
    
    try:
        model = NBEATSx(**params)
        forecaster = NeuralForecast(models=[model], freq='D')
        
        start_time = time.time()
        forecaster.fit(train_df)
        training_time = time.time() - start_time
        
        predictions = forecaster.predict(h=30)
        
        actual = val_df['y'].values[:30]
        predicted = predictions['NBEATSx'].values[:30]
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        
        print(f"Trial {trial.number}: RMSE={rmse:.4f}, Time={training_time:.1f}s")
        return rmse
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return float('inf')

# =============================================================================
# FINAL MODEL TRAINING
# =============================================================================

def train_final_model(df, best_params, device):
    """Train final model"""
    print("🎯 TRAINING FINAL MODEL")
    print("="*50)
    
    nf_df = df[['Date', 'Normalized_Balance']].copy()
    nf_df.columns = ['ds', 'y']
    nf_df['unique_id'] = 'balance'
    
    final_config = {
        'input_size': best_params.get('input_size', 120),
        'h': 30,
        'learning_rate': best_params.get('learning_rate', 0.001),
        'max_steps': best_params.get('max_steps', 1000),
        'batch_size': best_params.get('batch_size', 64),
        'num_hidden': best_params.get('num_hidden', 128),
        'dropout': best_params.get('dropout', 0.1),
        'random_state': 42,
    }
    
    if device.type == 'cuda':
        final_config.update({
            'accelerator': 'gpu',
            'devices': [0],
            'precision': '16-mixed',
            'gradient_clip_val': 0.1,
        })
    else:
        final_config.update({
            'accelerator': 'cpu',
            'precision': '32',
        })
    
    # Clear cache
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    start_time = time.time()
    
    model = NBEATSx(**final_config)
    forecaster = NeuralForecast(models=[model], freq='D')
    
    print("🚀 Starting training...")
    forecaster.fit(nf_df)
    
    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time:.1f} seconds")
    
    print("🔮 Generating forecast...")
    forecast = forecaster.predict(h=30)
    
    return forecaster, forecast, training_time

# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visualizations(df, forecast, training_time, device):
    """Create visualizations"""
    print("📊 CREATING VISUALIZATIONS")
    print("="*50)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('NBEATS GPU-Optimized Balance Forecasting Results', fontsize=16, fontweight='bold')
    
    # Time series plot
    ax1 = axes[0, 0]
    ax1.plot(df['Date'], df['Normalized_Balance'], label='Historical', color='blue', alpha=0.7)
    forecast_dates = pd.date_range(start=df['Date'].max() + timedelta(days=1), periods=30, freq='D')
    ax1.plot(forecast_dates, forecast['NBEATSx'], label='Forecast', color='red', linewidth=2)
    ax1.set_title('Balance Time Series with 30-Day Forecast')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Normalized Balance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Device info
    ax2 = axes[0, 1]
    device_info = f"Device: {device}\nGPU Available: {device.type == 'cuda'}\nTraining Time: {training_time:.1f}s"
    ax2.text(0.5, 0.5, device_info, ha='center', va='center', transform=ax2.transAxes, fontsize=12)
    ax2.set_title('Device Information')
    ax2.axis('off')
    
    # Training performance
    ax3 = axes[1, 0]
    metrics = ['Training Time (s)', 'Device Type']
    values = [training_time, 1 if device.type == 'cuda' else 0]
    colors = ['#ff9999', '#66b3ff']
    ax3.bar(metrics, values, color=colors)
    ax3.set_title('Training Performance')
    ax3.set_ylabel('Value')
    
    # Forecast distribution
    ax4 = axes[1, 1]
    ax4.hist(forecast['NBEATSx'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_title('Forecast Distribution')
    ax4.set_xlabel('Forecasted Balance')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gpu_forecasting_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("💾 Visualization saved to gpu_forecasting_results.png")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution"""
    print("🚀 NBEATS GPU-OPTIMIZED BALANCE PREDICTION SYSTEM")
    print("="*80)
    
    # Setup GPU
    device = setup_gpu()
    
    # Load data
    df = load_data()
    
    # Prepare data
    train_df, val_df = prepare_data(df)
    print(f"📊 Training data: {len(train_df)} samples")
    print(f"📊 Validation data: {len(val_df)} samples")
    
    # Setup optimization
    study_name = "nbeats_gpu_optimization"
    storage_url = f"sqlite:///optuna_study_{study_name}.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction='minimize',
        load_if_exists=True
    )
    
    print(f"\n🔬 Starting hyperparameter optimization")
    print(f"💾 Study: {study_name}")
    
    # Run optimization
    n_trials = 10  # Reduced for demonstration
    study.optimize(lambda trial: objective(trial, train_df, val_df, device), 
                  n_trials=n_trials, timeout=1800)  # 30 minutes timeout
    
    print(f"\n✅ Optimization completed!")
    print(f"🎯 Best RMSE: {study.best_value:.4f}")
    print(f"🔧 Best parameters: {study.best_params}")
    
    # Save parameters
    with open('best_gpu_params.json', 'w') as f:
        json.dump(study.best_params, f, indent=2)
    
    # Train final model
    final_forecaster, final_forecast, training_time = train_final_model(df, study.best_params, device)
    
    # Save forecast
    final_forecast.to_csv('gpu_forecast_30days.csv', index=False)
    print("💾 Forecast saved to gpu_forecast_30days.csv")
    
    # Create visualizations
    create_visualizations(df, final_forecast, training_time, device)
    
    # Summary
    print(f"\n📊 PERFORMANCE SUMMARY")
    print("="*50)
    print(f"🎯 Device: {device}")
    print(f"🚀 GPU Available: {device.type == 'cuda'}")
    print(f"⏱️  Training Time: {training_time:.1f} seconds")
    print(f"🔧 Best RMSE: {study.best_value:.4f}")
    print("="*50)
    print("🎉 GPU-OPTIMIZED NBEATS FORECASTING COMPLETED!")
    print("="*50)

if __name__ == "__main__":
    main() 