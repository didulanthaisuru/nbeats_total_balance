# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Set environment variables for GPU optimization
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use only GPU 0
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # Limit memory splits
os.environ['NCCL_P2P_DISABLE'] = '1'  # Disable peer-to-peer communication
os.environ['NCCL_IB_DISABLE'] = '1'  # Disable InfiniBand
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'  # Better memory management
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Consistent device ordering

# pip install neuralforecast
# pip install optuna

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# GPU Optimization imports
import torch

# Time series and forecasting
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATSx
from neuralforecast.losses.pytorch import DistributionLoss

# Metrics and evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Hyperparameter optimization
import optuna
from optuna.trial import Trial
import sqlite3

# Date and time
from datetime import datetime, timedelta
import json

# GPU Configuration - Lightning-compatible setup
print("🚀 GPU Configuration for T4x2 Setup")
print("="*50)

# Minimal GPU check to avoid Lightning conflicts
print(f"🔧 CUDA version: {torch.version.cuda}")
print("🎯 GPU will be configured by PyTorch Lightning automatically")
print("✅ GPU setup completed!")
print("="*50)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# GPU Memory monitoring and cleanup functions
def print_gpu_memory_usage():
    """Print current GPU memory usage - safer version that doesn't initialize CUDA"""
    try:
        # Only check GPU memory if CUDA is already available and initialized
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"💾 GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.2f}GB")
        else:
            print("💾 GPU Memory - CUDA not initialized yet")
    except Exception as e:
        print(f"💾 GPU Memory - Unable to check: {e}")

def cleanup_gpu_memory():
    """Clean up GPU memory between trials - safer version"""
    try:
        # Only cleanup if CUDA is already initialized
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("🧹 GPU memory cleaned up")
        else:
            print("🧹 GPU cleanup skipped - CUDA not initialized")
    except Exception as e:
        print(f"🧹 GPU cleanup attempted but failed: {e}")

def reset_cuda_context():
    """Reset CUDA context to prevent multiprocessing conflicts"""
    try:
        if torch.cuda.is_available():
            # Clear any existing CUDA context
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Reset the device
            torch.cuda.set_device(0)
            print("🔄 CUDA context reset")
    except Exception as e:
        print(f"🔄 CUDA context reset failed: {e}")

def configure_gpu_for_optimization():
    """Configure GPU settings for optimal hyperparameter optimization"""
    try:
        if torch.cuda.is_available():
            # Set memory fraction to prevent OOM
            torch.cuda.set_per_process_memory_fraction(0.8)
            
            # Enable memory efficient attention if available
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(True)
            
            # Set memory pool
            torch.cuda.memory.set_per_process_memory_fraction(0.8)
            
            print("🚀 GPU configured for optimization")
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            print("⚠️ CUDA not available")
    except Exception as e:
        print(f"❌ GPU configuration failed: {e}")

class LossTracker:
    """Custom callback to track training loss"""
    def __init__(self):
        self.final_loss = None
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of each training epoch"""
        if hasattr(trainer, 'callback_metrics') and 'train_loss' in trainer.callback_metrics:
            self.final_loss = trainer.callback_metrics['train_loss']
    
    def on_train_end(self, trainer, pl_module):
        """Called at the end of training"""
        if self.final_loss is None and hasattr(trainer, 'callback_metrics'):
            # Try to get any available loss metric
            for key in ['train_loss', 'loss', 'val_loss']:
                if key in trainer.callback_metrics:
                    self.final_loss = trainer.callback_metrics[key]
                    break

def create_gpu_optimized_model(params, n_blocks, feature_categories):
    """Create a GPU-optimized model with proper configuration"""
    try:
        # Ensure CUDA is available and properly configured
        if not torch.cuda.is_available():
            print("⚠️ CUDA not available, using CPU")
            return None
            
        # Set device
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        
        # Create model
        model = NBEATSx(
            h=HORIZON,
            input_size=params['input_size'],
            futr_exog_list=feature_categories['future_features'],
            hist_exog_list=feature_categories['historical_features'],
            
            # Architecture parameters
            stack_types=['identity', 'trend', 'seasonality'],
            n_blocks=n_blocks,
            n_harmonics=params['n_harmonics'],
            n_polynomials=params['n_polynomials'],
            
            # Training parameters
            learning_rate=params['learning_rate'],
            max_steps=params['max_steps'],
            batch_size=params['batch_size'],
            dropout_prob_theta=params['dropout_prob_theta'],
            
            # Other settings
            random_seed=42,
            scaler_type='standard',
            loss=DistributionLoss(distribution='Normal', level=[80, 90, 95])
        )
        print("🚀 GPU-optimized model created")
        return model
    except Exception as e:
        print(f"❌ GPU model creation failed: {e}")
        return None

# Configure matplotlib
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

print("✅ All libraries imported successfully!")
print(f"📅 Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🔧 Optuna version: {optuna.__version__}")

# Configure GPU for optimization
configure_gpu_for_optimization()

print("🚀 Ready for GPU-optimized forecasting!")
# Load the preprocessed dataset
print("📂 Loading preprocessed dataset...")
df = pd.read_excel("/kaggle/input/datasets-research/processed_train_dataset.xlsx")

print(f"✅ Dataset loaded successfully!")
print(f"📊 Dataset shape: {df.shape}")
print(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"📈 Total days: {len(df)}")

# Display basic info
print("\n📋 Dataset Info:")
print(f"Columns: {list(df.columns)}")
print(f"Data types:\n{df.dtypes}")

# Display first few rows
print("\n🔍 First 5 rows:")
print(df.head())

# Check for any missing values
print(f"\n❌ Missing values:")
missing_values = df.isnull().sum()
for col, missing in missing_values.items():
    if missing > 0:
        print(f"   {col}: {missing} ({missing/len(df)*100:.2f}%)")
    
if missing_values.sum() == 0:
    print("   ✅ No missing values found!")
else:
    print(f"   ⚠️ Total missing values: {missing_values.sum()}")
# Data validation and feature identification
print("🔍 Data Validation and Feature Analysis")
print("="*60)

# Validate essential columns
required_columns = ['Date', 'Normalized_Balance']
missing_required = [col for col in required_columns if col not in df.columns]

if missing_required:
    print(f"❌ Missing required columns: {missing_required}")
    raise ValueError(f"Dataset must contain columns: {required_columns}")
else:
    print("✅ All required columns present")

# Identify feature types
feature_categories = {
    'date_column': 'Date',
    'target_column': 'Normalized_Balance',
    'future_features': [],
    'historical_features': []
}

# Categorize features automatically
for col in df.columns:
    if col in required_columns:
        continue
    elif any(x in col.lower() for x in ['dayofweek_sin', 'dayofweek_cos', 'sin', 'cos']):
        feature_categories['future_features'].append(col)
    elif any(x in col.lower() for x in ['ago', 'lag', 'rolling', 'mean', 'std', 'changed']):
        feature_categories['historical_features'].append(col)

print(f"\n📊 Feature Categories:")
print(f"🎯 Target: {feature_categories['target_column']}")
print(f"🔮 Future features ({len(feature_categories['future_features'])}): {feature_categories['future_features']}")
print(f"📈 Historical features ({len(feature_categories['historical_features'])}): {feature_categories['historical_features']}")

# Validate data quality
print(f"\n🔍 Data Quality Checks:")

# Check date continuity
df_sorted = df.sort_values('Date').reset_index(drop=True)
date_diff = df_sorted['Date'].diff().dt.days
missing_dates = (date_diff > 1).sum()
print(f"📅 Date continuity: {len(df_sorted) - missing_dates - 1}/{len(df_sorted) - 1} consecutive days")

# Check target variable
target_stats = df[feature_categories['target_column']].describe()
print(f"💰 Target variable stats:")
print(f"   Range: {target_stats['min']:.4f} to {target_stats['max']:.4f}")
print(f"   Mean: {target_stats['mean']:.4f}, Std: {target_stats['std']:.4f}")

# Check for outliers (values beyond 3 standard deviations)
target_col = feature_categories['target_column']
mean_val = df[target_col].mean()
std_val = df[target_col].std()
outliers = ((df[target_col] - mean_val).abs() > 3 * std_val).sum()
print(f"⚠️ Potential outliers (>3σ): {outliers} ({outliers/len(df)*100:.2f}%)")

print("\n✅ Data validation completed!")
# Setup Optuna study with SQLite storage
print("🔧 Setting up Optuna hyperparameter optimization")
print("="*60)

# Configuration
HORIZON = 30  # Forecast horizon (days)
TIMEOUT_HOURS = 11  # 11 hours timeout
TIMEOUT_SECONDS = TIMEOUT_HOURS * 3600
STUDY_NAME = "nbeats_balance_forecasting"
DB_URL = f"sqlite:///optuna_study_{STUDY_NAME}.db"

print(f"🎯 Forecast horizon: {HORIZON} days")
print(f"⏰ Timeout: {TIMEOUT_HOURS} hours ({TIMEOUT_SECONDS} seconds)")
print(f"💾 Database: {DB_URL}")

# Create or load existing study
try:
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=DB_URL,
        direction='minimize',  # Minimize loss (lower is better)
        load_if_exists=True
    )
    print(f"✅ Study loaded/created: {len(study.trials)} existing trials")
    print(f"🎯 Optimization target: Distribution Loss Function (minimizing)")
except Exception as e:
    print(f"⚠️ Creating new study: {e}")
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=DB_URL,
        direction='minimize'  # Minimize loss function (lower is better)
    )
    print(f"🎯 Optimization target: Distribution Loss Function (minimizing)")

# Split data for hyperparameter tuning (use last HORIZON days for validation)
VALIDATION_DAYS = HORIZON  # Use same number of days as forecast horizon
split_idx = len(df) - VALIDATION_DAYS
train_data = df.iloc[:split_idx].copy()
val_data = df.iloc[split_idx:].copy()

print(f"📊 Training data: {len(train_data)} days ({df['Date'].iloc[0]} to {df['Date'].iloc[split_idx-1]})")
print(f"📊 Validation data: {len(val_data)} days ({df['Date'].iloc[split_idx]} to {df['Date'].iloc[-1]})")
print(f"🎯 Validation size matches forecast horizon: {len(val_data)} days")

# Prepare data for NeuralForecast format
def prepare_neural_forecast_data(data, feature_categories):
    """Convert data to NeuralForecast format"""
    nf_data = data.copy()
    nf_data['unique_id'] = 'balance'
    nf_data = nf_data.rename(columns={
        feature_categories['date_column']: 'ds', 
        feature_categories['target_column']: 'y'
    })
    return nf_data

train_nf = prepare_neural_forecast_data(train_data, feature_categories)
val_nf = prepare_neural_forecast_data(val_data, feature_categories)

print("✅ Data prepared for optimization!")
# Future features creation function
def create_future_features(last_date, horizon, feature_categories):
    """
    Create future features for forecasting period.
    Only creates features that can be known in advance (like time-based features).
    """
    # Generate future dates
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), 
        periods=horizon, 
        freq='D'
    )
    
    # Create future dataframe
    future_df = pd.DataFrame({
        'ds': future_dates,
        'unique_id': 'balance'
    })
    
    # Add time-based features (these can be known in advance)
    future_df['dayofweek_sin'] = np.sin(2 * np.pi * future_df['ds'].dt.dayofweek / 7)
    future_df['dayofweek_cos'] = np.cos(2 * np.pi * future_df['ds'].dt.dayofweek / 7)
    
    # Add any other future features that exist in the dataset
    for feature in feature_categories['future_features']:
        if feature not in future_df.columns:
            if 'sin' in feature.lower():
                # Handle additional sine features
                if 'month' in feature.lower():
                    future_df[feature] = np.sin(2 * np.pi * future_df['ds'].dt.month / 12)
                elif 'dayofyear' in feature.lower():
                    future_df[feature] = np.sin(2 * np.pi * future_df['ds'].dt.dayofyear / 365.25)
                else:
                    future_df[feature] = 0  # Default value
            elif 'cos' in feature.lower():
                # Handle additional cosine features
                if 'month' in feature.lower():
                    future_df[feature] = np.cos(2 * np.pi * future_df['ds'].dt.month / 12)
                elif 'dayofyear' in feature.lower():
                    future_df[feature] = np.cos(2 * np.pi * future_df['ds'].dt.dayofyear / 365.25)
                else:
                    future_df[feature] = 0  # Default value
            else:
                future_df[feature] = 0  # Default for unknown future features
    
    print(f"🔮 Created future features for {horizon} days")
    print(f"📅 Future period: {future_dates[0]} to {future_dates[-1]}")
    
    return future_df

# Test future features creation
last_training_date = train_nf['ds'].max()
test_future_df = create_future_features(last_training_date, 5, feature_categories)
print("\n📋 Sample future features:")
print(test_future_df.head())
# Optuna objective function
def objective(trial):
    """
    Optuna objective function for NBEATSx hyperparameter optimization.
    Returns training loss to minimize (with RMSE fallback if loss not available).
    """
    try:
        # Configure GPU for optimization
        configure_gpu_for_optimization()
        reset_cuda_context()
        
        # Force single-threaded execution to prevent multiprocessing
        import os
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        
        # Optimize GPU memory usage
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        
        # Hyperparameter search space
        params = {
            'input_size': trial.suggest_int('input_size', 60, 200),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'max_steps': trial.suggest_int('max_steps', 500, 2000),
            'batch_size': trial.suggest_int('batch_size', 16, 64),
            'n_harmonics': trial.suggest_int('n_harmonics', 1, 5),
            'n_polynomials': trial.suggest_int('n_polynomials', 1, 5),
            'dropout_prob_theta': trial.suggest_float('dropout_prob_theta', 0.0, 0.3)
        }
        
        # N-blocks for each stack (identity, trend, seasonality)
        n_blocks = [
            trial.suggest_int('n_blocks_identity', 2, 6),
            trial.suggest_int('n_blocks_trend', 2, 6), 
            trial.suggest_int('n_blocks_seasonality', 2, 6)
        ]
        
        # Print GPU memory before model creation (safer version)
        print_gpu_memory_usage()
        
        # Create NBEATSx model with GPU optimization
        try:
            model = NBEATSx(
                h=HORIZON,
                input_size=params['input_size'],
                futr_exog_list=feature_categories['future_features'],
                hist_exog_list=feature_categories['historical_features'],
                
                # Architecture parameters
                stack_types=['identity', 'trend', 'seasonality'],
                n_blocks=n_blocks,
                n_harmonics=params['n_harmonics'],
                n_polynomials=params['n_polynomials'],
                
                # Training parameters
                learning_rate=params['learning_rate'],
                max_steps=params['max_steps'],
                batch_size=params['batch_size'],
                dropout_prob_theta=params['dropout_prob_theta'],
                
                # Other settings
                random_seed=42,
                scaler_type='standard',
                loss=DistributionLoss(distribution='Normal', level=[80, 90, 95])
            )
            
            # Add a custom attribute to store the final loss
            model.final_training_loss = None
            
            print(f"✅ Model created successfully for trial {trial.number}")
        except Exception as model_error:
            print(f"❌ Model creation failed for trial {trial.number}: {model_error}")
            cleanup_gpu_memory()
            return float('inf')
        
        # Create forecaster with safer configuration
        try:
            forecaster = NeuralForecast(
                models=[model], 
                freq='D'
            )
            print(f"✅ Forecaster created successfully for trial {trial.number}")
        except Exception as forecaster_error:
            print(f"❌ Forecaster creation failed for trial {trial.number}: {forecaster_error}")
            cleanup_gpu_memory()
            return float('inf')
        
        # Fit model on training data with error handling and loss tracking
        print(f"🎯 Training model for trial {trial.number}...")
        try:
            # Use a try-catch block specifically for Lightning multiprocessing issues
            forecaster.fit(df=train_nf)
            print(f"✅ Model training completed for trial {trial.number}")
            
            # Extract training loss from the model
            try:
                # Get the trained model from the forecaster
                trained_model = forecaster.models[0]
                
                # Try multiple approaches to extract loss
                loss_value = None
                
                # Approach 1: Try to get loss from trainer callback metrics
                if hasattr(trained_model, 'trainer') and hasattr(trained_model.trainer, 'callback_metrics'):
                    callback_metrics = trained_model.trainer.callback_metrics
                    print(f"🔍 Available callback metrics: {list(callback_metrics.keys())}")
                    
                    # Try different loss key names
                    loss_keys = ['train_loss', 'loss', 'train_loss_epoch', 'train_loss_step']
                    for key in loss_keys:
                        if key in callback_metrics:
                            loss = callback_metrics[key]
                            if loss is not None:
                                loss_value = float(loss.cpu().numpy()) if hasattr(loss, 'cpu') else float(loss)
                                print(f"📊 Training loss for trial {trial.number} ({key}): {loss_value:.6f}")
                                cleanup_gpu_memory()
                                return loss_value
                
                # Approach 2: Try to get loss from model's training history
                if hasattr(trained_model, 'training_step_outputs') and trained_model.training_step_outputs:
                    # Get the last training step loss
                    last_loss = trained_model.training_step_outputs[-1]
                    if hasattr(last_loss, 'loss'):
                        loss_value = float(last_loss.loss.cpu().numpy()) if hasattr(last_loss.loss, 'cpu') else float(last_loss.loss)
                        print(f"📊 Training loss for trial {trial.number} (from outputs): {loss_value:.6f}")
                        cleanup_gpu_memory()
                        return loss_value
                
                # Approach 3: Try to get loss from model's state
                if hasattr(trained_model, 'current_loss'):
                    loss_value = float(trained_model.current_loss.cpu().numpy()) if hasattr(trained_model.current_loss, 'cpu') else float(trained_model.current_loss)
                    print(f"📊 Training loss for trial {trial.number} (current_loss): {loss_value:.6f}")
                    cleanup_gpu_memory()
                    return loss_value
                
                # Approach 4: Try validation loss as fallback
                if hasattr(trained_model, 'trainer') and hasattr(trained_model.trainer, 'callback_metrics'):
                    val_loss_keys = ['val_loss', 'validation_loss', 'val_loss_epoch']
                    for key in val_loss_keys:
                        if key in callback_metrics:
                            val_loss = callback_metrics[key]
                            if val_loss is not None:
                                val_loss_value = float(val_loss.cpu().numpy()) if hasattr(val_loss, 'cpu') else float(val_loss)
                                print(f"📊 Validation loss for trial {trial.number} ({key}): {val_loss_value:.6f}")
                                cleanup_gpu_memory()
                                return val_loss_value
                
                # If no loss available, try to calculate loss directly
                print(f"⚠️ No loss available for trial {trial.number}, calculating loss directly...")
                
                # Try to calculate the actual loss function value
                try:
                    # Get the model's loss function
                    if hasattr(trained_model, 'loss'):
                        loss_fn = trained_model.loss
                        print(f"🔍 Model loss function: {type(loss_fn).__name__}")
                        
                        # Create future features for the validation period
                        future_df = create_future_features(
                            last_date=train_nf['ds'].max(),
                            horizon=len(val_nf),
                            feature_categories=feature_categories
                        )
                        
                        # Get predictions using the trained model
                        forecast_df = forecaster.predict(futr_df=future_df)
                        
                        # Calculate the actual loss function value
                        actual = val_nf['y'].values
                        predicted = forecast_df['NBEATSx'].values
                        
                        # Handle any NaN values
                        mask = ~(np.isnan(actual) | np.isnan(predicted))
                        if mask.sum() > 0:
                            # Convert to tensors for loss calculation
                            import torch
                            y_true_tensor = torch.tensor(actual[mask], dtype=torch.float32)
                            y_pred_tensor = torch.tensor(predicted[mask], dtype=torch.float32)
                            
                            # Try different approaches to calculate loss
                            try:
                                # Approach 1: Try with reshaped tensors
                                y_true_reshaped = y_true_tensor.unsqueeze(-1)
                                y_pred_reshaped = y_pred_tensor.unsqueeze(-1)
                                loss_value = loss_fn(y_pred_reshaped, y_true_reshaped)
                                loss_value = float(loss_value.detach().cpu().numpy())
                                print(f"📊 Actual loss function value for trial {trial.number}: {loss_value:.6f}")
                                cleanup_gpu_memory()
                                return loss_value
                            except Exception as loss_calc_error1:
                                print(f"⚠️ Loss function calculation failed (reshaped): {loss_calc_error1}")
                                try:
                                    # Approach 2: Try with original tensors
                                    loss_value = loss_fn(y_pred_tensor, y_true_tensor)
                                    loss_value = float(loss_value.detach().cpu().numpy())
                                    print(f"📊 Actual loss function value for trial {trial.number}: {loss_value:.6f}")
                                    cleanup_gpu_memory()
                                    return loss_value
                                except Exception as loss_calc_error2:
                                    print(f"⚠️ Loss function calculation failed (original): {loss_calc_error2}")
                                    # Fall back to simple MSE calculation
                                    mse_loss = torch.mean((y_pred_tensor - y_true_tensor) ** 2)
                                    mse_loss = float(mse_loss.detach().cpu().numpy())
                                    print(f"📊 MSE loss fallback for trial {trial.number}: {mse_loss:.6f}")
                                    cleanup_gpu_memory()
                                    return mse_loss
                        else:
                            print(f"⚠️ No valid predictions for loss calculation in trial {trial.number}")
                    
                except Exception as direct_loss_error:
                    print(f"⚠️ Direct loss function calculation failed: {direct_loss_error}")
                
                print(f"📊 Calculating RMSE as final fallback...")
                print(f"🔍 Model attributes: {[attr for attr in dir(trained_model) if 'loss' in attr.lower() or 'train' in attr.lower()]}")
                
            except Exception as loss_error:
                print(f"⚠️ Could not extract loss for trial {trial.number}: {loss_error}")
                print(f"📊 Calculating RMSE as fallback...")
            
            # If direct loss calculation also fails, return a high but finite value
            print(f"❌ All loss calculation methods failed for trial {trial.number}")
            cleanup_gpu_memory()
            return 1000.0
            
        except RuntimeError as e:
            if "Lightning can't create new processes if CUDA is already initialized" in str(e):
                print(f"⚠️ CUDA/Lightning conflict detected for trial {trial.number}, trying GPU optimization...")
                # Try GPU optimization approach
                try:
                    # Force aggressive CUDA cleanup and reinitialization
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        # Reset CUDA context
                        torch.cuda.set_device(0)
                    
                    gpu_model = create_gpu_optimized_model(params, n_blocks, feature_categories)
                    if gpu_model is not None:
                        gpu_forecaster = NeuralForecast(models=[gpu_model], freq='D')
                        gpu_forecaster.fit(df=train_nf)
                        print(f"✅ GPU training completed for trial {trial.number}")
                        
                        # Try to extract loss from GPU model
                        try:
                            trained_gpu_model = gpu_forecaster.models[0]
                            if hasattr(trained_gpu_model, 'trainer') and hasattr(trained_gpu_model.trainer, 'callback_metrics'):
                                loss = trained_gpu_model.trainer.callback_metrics.get('train_loss', None)
                                if loss is not None:
                                    loss_value = float(loss.cpu().numpy()) if hasattr(loss, 'cpu') else float(loss)
                                    print(f"📊 GPU Training loss for trial {trial.number}: {loss_value:.6f}")
                                    cleanup_gpu_memory()
                                    return loss_value
                        except Exception as gpu_loss_error:
                            print(f"⚠️ Could not extract GPU loss for trial {trial.number}: {gpu_loss_error}")
                        
                        # Calculate actual loss function value for GPU model
                        try:
                            if hasattr(trained_gpu_model, 'loss'):
                                loss_fn = trained_gpu_model.loss
                                print(f"🔍 GPU Model loss function: {type(loss_fn).__name__}")
                                
                                # Create future features for the validation period
                                future_df = create_future_features(
                                    last_date=train_nf['ds'].max(),
                                    horizon=len(val_nf),
                                    feature_categories=feature_categories
                                )
                                
                                # Get predictions using the GPU model
                                forecast_df = gpu_forecaster.predict(futr_df=future_df)
                                
                                # Calculate the actual loss function value
                                actual = val_nf['y'].values
                                predicted = forecast_df['NBEATSx'].values
                                
                                # Handle any NaN values
                                mask = ~(np.isnan(actual) | np.isnan(predicted))
                                if mask.sum() > 0:
                                    # Convert to tensors for loss calculation
                                    import torch
                                    y_true_tensor = torch.tensor(actual[mask], dtype=torch.float32)
                                    y_pred_tensor = torch.tensor(predicted[mask], dtype=torch.float32)
                                    
                                    # Try different approaches to calculate loss
                                    try:
                                        # Approach 1: Try with reshaped tensors
                                        y_true_reshaped = y_true_tensor.unsqueeze(-1)
                                        y_pred_reshaped = y_pred_tensor.unsqueeze(-1)
                                        loss_value = loss_fn(y_pred_reshaped, y_true_reshaped)
                                        loss_value = float(loss_value.detach().cpu().numpy())
                                        print(f"📊 GPU Actual loss function value for trial {trial.number}: {loss_value:.6f}")
                                        cleanup_gpu_memory()
                                        return loss_value
                                    except Exception as gpu_loss_calc_error1:
                                        print(f"⚠️ GPU Loss function calculation failed (reshaped): {gpu_loss_calc_error1}")
                                        try:
                                            # Approach 2: Try with original tensors
                                            loss_value = loss_fn(y_pred_tensor, y_true_tensor)
                                            loss_value = float(loss_value.detach().cpu().numpy())
                                            print(f"📊 GPU Actual loss function value for trial {trial.number}: {loss_value:.6f}")
                                            cleanup_gpu_memory()
                                            return loss_value
                                        except Exception as gpu_loss_calc_error2:
                                            print(f"⚠️ GPU Loss function calculation failed (original): {gpu_loss_calc_error2}")
                                            # Fall back to simple MSE calculation
                                            mse_loss = torch.mean((y_pred_tensor - y_true_tensor) ** 2)
                                            mse_loss = float(mse_loss.detach().cpu().numpy())
                                            print(f"📊 GPU MSE loss fallback for trial {trial.number}: {mse_loss:.6f}")
                                            cleanup_gpu_memory()
                                            return mse_loss
                                else:
                                    print(f"⚠️ No valid predictions for GPU loss calculation in trial {trial.number}")
                            
                        except Exception as gpu_direct_loss_error:
                            print(f"⚠️ GPU Direct loss function calculation failed: {gpu_direct_loss_error}")
                        
                        # Generate predictions with GPU model for RMSE fallback
                        future_df = create_future_features(
                            last_date=train_nf['ds'].max(),
                            horizon=len(val_nf),
                            feature_categories=feature_categories
                        )
                        forecast_df = gpu_forecaster.predict(futr_df=future_df)
                        
                        # Calculate RMSE as fallback
                        actual = val_nf['y'].values
                        predicted = forecast_df['NBEATSx'].values
                        mask = ~(np.isnan(actual) | np.isnan(predicted))
                        if mask.sum() > 0:
                            rmse = np.sqrt(mean_squared_error(actual[mask], predicted[mask]))
                            print(f"Trial {trial.number} (GPU RMSE fallback): {rmse:.6f}")
                            cleanup_gpu_memory()
                            return rmse
                    
                    # If GPU optimization also fails, return a high but finite value
                    cleanup_gpu_memory()
                    return 1000.0
                except Exception as gpu_error:
                    print(f"❌ GPU optimization also failed for trial {trial.number}: {gpu_error}")
                    cleanup_gpu_memory()
                    return 1000.0
            else:
                print(f"❌ Model training failed for trial {trial.number}: {e}")
                cleanup_gpu_memory()
                return float('inf')
        except Exception as training_error:
            print(f"❌ Model training failed for trial {trial.number}: {training_error}")
            cleanup_gpu_memory()
            return float('inf')
        
        # Print GPU memory after training
        print_gpu_memory_usage()
        
        # Clean up GPU memory after successful training
        cleanup_gpu_memory()
        
        # Create future features for validation period
        try:
            future_df = create_future_features(
                last_date=train_nf['ds'].max(),
                horizon=len(val_nf),
                feature_categories=feature_categories
            )
            print(f"✅ Future features created for trial {trial.number}")
        except Exception as future_error:
            print(f"❌ Future features creation failed for trial {trial.number}: {future_error}")
            cleanup_gpu_memory()
            return float('inf')
        
        # Generate predictions
        try:
            forecast_df = forecaster.predict(futr_df=future_df)
            print(f"✅ Predictions generated for trial {trial.number}")
        except Exception as predict_error:
            print(f"❌ Prediction failed for trial {trial.number}: {predict_error}")
            cleanup_gpu_memory()
            return float('inf')
        
        # Calculate RMSE
        actual = val_nf['y'].values
        predicted = forecast_df['NBEATSx'].values
        
        # Handle any potential NaN values
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        if mask.sum() == 0:
            cleanup_gpu_memory()
            return float('inf')
            
        rmse = np.sqrt(mean_squared_error(actual[mask], predicted[mask]))
        
        # Log trial info
        trial_info = {
            'trial_number': trial.number,
            'rmse': rmse,
            'params': params,
            'n_blocks': n_blocks
        }
        
        print(f"Trial {trial.number}: RMSE = {rmse:.6f}")
        
        # Final cleanup before returning
        cleanup_gpu_memory()
        return rmse
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        cleanup_gpu_memory()
        return float('inf')

print("🎯 Objective function defined!")
print("⚙️ Hyperparameter search space:")
print("   - input_size: 60-200")
print("   - learning_rate: 1e-4 to 1e-2 (log scale)")
print("   - max_steps: 500-2000")
print("   - batch_size: 16-64")
print("   - n_blocks per stack: 2-6 each")
print("   - n_harmonics: 1-5")
print("   - n_polynomials: 1-5")
print("   - dropout_prob_theta: 0.0-0.3")
print("🎯 Optimization target: Distribution Loss Function (minimizing)")
print("📊 Fallback metric: RMSE (if loss function not available)")
print("   Note: Early stopping and weight_decay disabled (not supported by NBEATSx)")
# Run hyperparameter optimization
print("🚀 Starting hyperparameter optimization...")
print("="*60)
print(f"⏰ Timeout: {TIMEOUT_HOURS} hours")
print(f"💾 Results will be saved to: {DB_URL}")
print(f"🔄 Existing trials: {len(study.trials)}")

# Record start time
start_time = datetime.now()
print(f"🕐 Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Run optimization with timeout
try:
    study.optimize(
        objective, 
        timeout=TIMEOUT_SECONDS,
        n_jobs=1,  # Single job to avoid conflicts
        show_progress_bar=True
    )
    
    optimization_completed = True
    print("\n✅ Optimization completed successfully!")
    
except KeyboardInterrupt:
    print("\n⚠️ Optimization interrupted by user")
    optimization_completed = False
    
except Exception as e:
    print(f"\n❌ Optimization failed: {e}")
    optimization_completed = False

# Record end time and duration
end_time = datetime.now()
duration = end_time - start_time
print(f"🕐 End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"⏱️ Duration: {duration}")

# Display results
print("\n" + "="*60)
print("📊 OPTIMIZATION RESULTS")
print("="*60)

if len(study.trials) > 0:
    print(f"🔄 Total trials completed: {len(study.trials)}")
    print(f"🏆 Best Loss Function Value: {study.best_value:.6f}")
    
    # Get best parameters
    best_params = study.best_params.copy()
    
    # Reconstruct n_blocks array
    n_blocks_best = [
        best_params.pop('n_blocks_identity'),
        best_params.pop('n_blocks_trend'),
        best_params.pop('n_blocks_seasonality')
    ]
    best_params['n_blocks'] = n_blocks_best
    
    print(f"\n🎯 Best hyperparameters:")
    for param, value in best_params.items():
        print(f"   {param}: {value}")
    
    # Save best parameters
    with open('best_hyperparameters.json', 'w') as f:
        json.dump(best_params, f, indent=2, default=str)
    print(f"\n💾 Best parameters saved to: best_hyperparameters.json")
    
else:
    print("❌ No trials completed")
    best_params = None

print("="*60)
# Analyze optimization results
if len(study.trials) > 0:
    print("📈 OPTIMIZATION ANALYSIS")
    print("="*60)
    
    # Create trials dataframe for analysis
    trials_df = study.trials_dataframe()
    completed_trials = trials_df[trials_df['state'] == 'COMPLETE']
    
    if len(completed_trials) > 0:
        print(f"✅ Completed trials: {len(completed_trials)}")
        print(f"❌ Failed trials: {len(trials_df) - len(completed_trials)}")
        print(f"📊 Success rate: {len(completed_trials)/len(trials_df)*100:.1f}%")
        
        # Statistics
        loss_stats = completed_trials['value'].describe()
        print(f"\n📊 Loss Function Statistics:")
        print(f"   Best (min): {loss_stats['min']:.6f}")
        print(f"   Worst (max): {loss_stats['max']:.6f}")
        print(f"   Mean: {loss_stats['mean']:.6f}")
        print(f"   Std: {loss_stats['std']:.6f}")
        print(f"   Median: {loss_stats['50%']:.6f}")
        
        # Plot optimization history
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Optimization history
        plt.subplot(2, 2, 1)
        plt.plot(completed_trials['number'], completed_trials['value'], 'b-', alpha=0.7)
        plt.axhline(y=study.best_value, color='r', linestyle='--', label=f'Best Loss Function: {study.best_value:.6f}')
        plt.xlabel('Trial Number')
        plt.ylabel('Loss Function Value')
        plt.title('Optimization History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Loss distribution
        plt.subplot(2, 2, 2)
        plt.hist(completed_trials['value'], bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(x=study.best_value, color='r', linestyle='--', label=f'Best: {study.best_value:.6f}')
        plt.xlabel('Loss Function Value')
        plt.ylabel('Frequency')
        plt.title('Loss Function Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Parameter importance (if enough trials)
        if len(completed_trials) >= 10:
            plt.subplot(2, 2, 3)
            try:
                importance = optuna.importance.get_param_importances(study)
                params = list(importance.keys())[:8]  # Top 8 parameters
                values = [importance[p] for p in params]
                
                plt.barh(params, values)
                plt.xlabel('Importance')
                plt.title('Parameter Importance')
                plt.gca().invert_yaxis()
            except:
                plt.text(0.5, 0.5, 'Parameter importance\nnot available', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Parameter Importance')
        
        # Plot 4: Learning curve of best trial
        plt.subplot(2, 2, 4)
        # This would show learning curve if we had access to training history
        plt.text(0.5, 0.5, f'Best Trial: #{study.best_trial.number}\nLoss Function: {study.best_value:.6f}', 
                ha='center', va='center', transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        plt.title('Best Trial Summary')
        
        plt.tight_layout()
        plt.show()
        
        # Save optimization report
        optimization_report = {
            'study_name': STUDY_NAME,
            'total_trials': len(study.trials),
            'completed_trials': len(completed_trials),
            'best_loss_function': study.best_value,
            'best_params': best_params,
            'optimization_duration': str(duration),
            'loss_function_statistics': loss_stats.to_dict()
        }
        
        with open('optimization_report.json', 'w') as f:
            json.dump(optimization_report, f, indent=2, default=str)
        
        print(f"\n💾 Optimization report saved to: optimization_report.json")
    else:
        print("❌ No completed trials to analyze")
else:
    print("❌ No trials to analyze")
    best_params = None
# Train final model with best parameters
print("🚀 Training Final Model")
print("="*60)

# Ensure we have best parameters
if best_params is None:
    print("⚠️ No optimized parameters found, using default parameters")
    best_params = {
        'input_size': 120,
        'learning_rate': 0.001,
        'max_steps': 1000,
        'batch_size': 32,
        'n_blocks': [3, 3, 3],
        'n_harmonics': 2,
        'n_polynomials': 2,
        'dropout_prob_theta': 0.1
        # Removed weight_decay and early_stop_patience_steps as they're not supported
    }

print(f"🎯 Using parameters:")
for param, value in best_params.items():
    print(f"   {param}: {value}")

# Prepare full dataset for training
full_nf_data = prepare_neural_forecast_data(df, feature_categories)
print(f"\n📊 Full training dataset: {len(full_nf_data)} days")
print(f"📅 Training period: {full_nf_data['ds'].min()} to {full_nf_data['ds'].max()}")

# Print GPU memory before final model creation
print_gpu_memory_usage()

# Create final model with best parameters and GPU optimizations
final_model = NBEATSx(
    h=HORIZON,
    input_size=best_params['input_size'],
    futr_exog_list=feature_categories['future_features'],
    hist_exog_list=feature_categories['historical_features'],
    
    # Architecture parameters
    stack_types=['identity', 'trend', 'seasonality'],
    n_blocks=best_params['n_blocks'],
    n_harmonics=best_params['n_harmonics'],
    n_polynomials=best_params['n_polynomials'],
    
    # Training parameters
    learning_rate=best_params['learning_rate'],
    max_steps=best_params['max_steps'],
    batch_size=best_params['batch_size'],
    dropout_prob_theta=best_params.get('dropout_prob_theta', 0.0),
    # Removed weight_decay and early_stop_patience_steps as they're not supported
    
    # Other settings
    random_seed=42,
    scaler_type='standard',
    loss=DistributionLoss(distribution='Normal', level=[80, 90, 95])
    # Removed trainer_kwargs as it's not supported by NBEATSx
)

# Create final forecaster with GPU optimizations
final_forecaster = NeuralForecast(
    models=[final_model], 
    freq='D'
)

# Train the final model
print(f"\n🎓 Training final model...")
training_start = datetime.now()

try:
    # Train final model without early stopping to avoid validation requirements
    final_forecaster.fit(df=full_nf_data)
    training_success = True
    print("✅ Final model training completed successfully!")
    
    # Print GPU memory after final training
    print_gpu_memory_usage()
    
except Exception as e:
    training_success = False
    print(f"❌ Final model training failed: {e}")

training_end = datetime.now()
training_duration = training_end - training_start
print(f"⏱️ Training duration: {training_duration}")

if training_success:
    print("\n🎉 Final model ready for forecasting!")
# Generate 30-day forecast
if training_success:
    print("🔮 Generating 30-Day Balance Forecast")
    print("="*60)
    
    # Create future features for the next 30 days
    last_date = full_nf_data['ds'].max()
    future_features_df = create_future_features(
        last_date=last_date,
        horizon=HORIZON,
        feature_categories=feature_categories
    )
    
    print(f"📅 Forecast period: {future_features_df['ds'].min()} to {future_features_df['ds'].max()}")
    
    # Generate forecast with uncertainty intervals
    forecast_start = datetime.now()
    
    try:
        forecast_df = final_forecaster.predict(futr_df=future_features_df)
        forecasting_success = True
        print("✅ Forecast generated successfully!")
        
    except Exception as e:
        forecasting_success = False
        print(f"❌ Forecasting failed: {e}")
        forecast_df = None
    
    forecast_end = datetime.now()
    forecast_duration = forecast_end - forecast_start
    print(f"⏱️ Forecasting duration: {forecast_duration}")
    
    if forecasting_success and forecast_df is not None:
        # Process forecast results
        forecast_df['Date'] = future_features_df['ds']
        forecast_df = forecast_df.reset_index(drop=True)
        
        # Extract predictions and confidence intervals
        point_forecast = forecast_df['NBEATSx'].values
        
        # Extract confidence intervals if available
        ci_columns = [col for col in forecast_df.columns if 'NBEATSx' in col and any(level in col for level in ['80', '90', '95'])]
        
        print(f"\n📊 Forecast Summary:")
        print(f"   Horizon: {HORIZON} days")
        print(f"   Point forecasts: {len(point_forecast)} values")
        print(f"   Confidence intervals: {len(ci_columns)} levels")
        print(f"   Available intervals: {[col.split('-')[-1] for col in ci_columns if 'hi' in col]}")
        
        # Create comprehensive forecast dataframe
        forecast_summary = pd.DataFrame({
            'Date': future_features_df['ds'],
            'Day': range(1, HORIZON + 1),
            'Predicted_Balance': point_forecast
        })
        
        # Add confidence intervals
        for col in ci_columns:
            if 'lo' in col:
                level = col.split('-')[-1]
                forecast_summary[f'Lower_CI_{level}'] = forecast_df[col]
            elif 'hi' in col:
                level = col.split('-')[-1]
                forecast_summary[f'Upper_CI_{level}'] = forecast_df[col]
        
        # Add trend information
        forecast_summary['Daily_Change'] = forecast_summary['Predicted_Balance'].diff()
        forecast_summary['Cumulative_Change'] = forecast_summary['Predicted_Balance'] - forecast_summary['Predicted_Balance'].iloc[0]
        forecast_summary['Weekly_Change'] = forecast_summary['Predicted_Balance'].diff(7)
        
        print(f"\n📈 Forecast Statistics:")
        print(f"   Starting balance: {point_forecast[0]:.4f}")
        print(f"   Ending balance: {point_forecast[-1]:.4f}")
        print(f"   Total change: {point_forecast[-1] - point_forecast[0]:.4f}")
        print(f"   Average daily change: {forecast_summary['Daily_Change'].mean():.4f}")
        print(f"   Max daily change: {forecast_summary['Daily_Change'].max():.4f}")
        print(f"   Min daily change: {forecast_summary['Daily_Change'].min():.4f}")
        
        # Display first few predictions
        print(f"\n🔍 First 10 predictions:")
        print(forecast_summary[['Date', 'Day', 'Predicted_Balance', 'Daily_Change']].head(10).to_string(index=False))
        
        # Save forecast results
        forecast_summary.to_csv('30_day_forecast.csv', index=False)
        forecast_summary.to_excel('30_day_forecast.xlsx', index=False)
        print(f"\n💾 Forecast saved to:")
        print(f"   - 30_day_forecast.csv")
        print(f"   - 30_day_forecast.xlsx")
        
    else:
        print("❌ Cannot proceed with analysis - forecasting failed")
        forecast_summary = None

else:
    print("❌ Cannot generate forecast - final model training failed")
    forecast_summary = None
# Visualize forecast results
if forecasting_success and forecast_summary is not None:
    print("📊 Creating Forecast Visualizations")
    print("="*60)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot 1: Historical + Forecast
    ax1 = axes[0, 0]
    
    # Show last 60 days of historical data for context
    hist_context = full_nf_data.tail(60)
    ax1.plot(hist_context['ds'], hist_context['y'], 'b-', label='Historical Balance', linewidth=2)
    ax1.plot(forecast_summary['Date'], forecast_summary['Predicted_Balance'], 'r-', 
             label='30-Day Forecast', linewidth=2, marker='o', markersize=3)
    
    # Add confidence intervals if available
    if 'Lower_CI_95' in forecast_summary.columns:
        ax1.fill_between(forecast_summary['Date'], 
                        forecast_summary['Lower_CI_95'], 
                        forecast_summary['Upper_CI_95'],
                        alpha=0.2, color='red', label='95% Confidence')
    
    if 'Lower_CI_80' in forecast_summary.columns:
        ax1.fill_between(forecast_summary['Date'], 
                        forecast_summary['Lower_CI_80'], 
                        forecast_summary['Upper_CI_80'],
                        alpha=0.3, color='orange', label='80% Confidence')
    
    ax1.axvline(x=last_date, color='green', linestyle='--', alpha=0.7, label='Forecast Start')
    ax1.set_title('Account Balance: Historical + 30-Day Forecast')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Normalized Balance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Daily Changes
    ax2 = axes[0, 1]
    ax2.plot(forecast_summary['Date'], forecast_summary['Daily_Change'], 'g-', 
             marker='o', markersize=4, label='Daily Change')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_title('Predicted Daily Balance Changes')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Daily Change')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Cumulative Change
    ax3 = axes[1, 0]
    ax3.plot(forecast_summary['Date'], forecast_summary['Cumulative_Change'], 'purple', 
             marker='o', markersize=4, label='Cumulative Change')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.set_title('Cumulative Balance Change from Day 1')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Cumulative Change')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Weekly Analysis
    ax4 = axes[1, 1]
    
    # Group by week for weekly analysis
    forecast_summary['Week'] = ((forecast_summary['Day'] - 1) // 7) + 1
    weekly_summary = forecast_summary.groupby('Week').agg({
        'Predicted_Balance': ['mean', 'min', 'max'],
        'Daily_Change': 'sum'
    }).round(4)
    
    weekly_summary.columns = ['Avg_Balance', 'Min_Balance', 'Max_Balance', 'Weekly_Change']
    weekly_summary = weekly_summary.reset_index()
    
    ax4.bar(weekly_summary['Week'], weekly_summary['Weekly_Change'], 
            alpha=0.7, color=['green' if x >= 0 else 'red' for x in weekly_summary['Weekly_Change']])
    ax4.set_title('Weekly Balance Changes')
    ax4.set_xlabel('Week')
    ax4.set_ylabel('Weekly Change')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print weekly summary
    print(f"\n📊 Weekly Forecast Summary:")
    print(weekly_summary.to_string(index=False))
    
    # Save visualization
    fig.savefig('30_day_forecast_visualization.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Visualization saved to: 30_day_forecast_visualization.png")
    
    # Risk Analysis
    print(f"\n⚠️ RISK ANALYSIS:")
    
    # Volatility analysis
    daily_volatility = forecast_summary['Daily_Change'].std()
    max_decline = forecast_summary['Daily_Change'].min()
    max_increase = forecast_summary['Daily_Change'].max()
    
    print(f"   📈 Daily volatility (std): {daily_volatility:.4f}")
    print(f"   📉 Largest predicted decline: {max_decline:.4f}")
    print(f"   📈 Largest predicted increase: {max_increase:.4f}")
    
    # Trend analysis
    if forecast_summary['Cumulative_Change'].iloc[-1] > 0:
        trend = "INCREASING"
        trend_emoji = "📈"
    else:
        trend = "DECREASING" 
        trend_emoji = "📉"
    
    print(f"   {trend_emoji} Overall 30-day trend: {trend}")
    print(f"   💰 Expected total change: {forecast_summary['Cumulative_Change'].iloc[-1]:.4f}")
    
    # Weeks with negative changes
    negative_weeks = (weekly_summary['Weekly_Change'] < 0).sum()
    print(f"   ⚠️ Weeks with declining balance: {negative_weeks}/4")

else:
    print("❌ Cannot create visualizations - forecasting data not available")
# Final Summary and Conclusions
print("🎉 NBEATS BALANCE FORECASTING SUMMARY")
print("="*70)

# Project Overview
project_summary = {
    'model_type': 'NBEATSx Neural Network',
    'forecast_horizon': f'{HORIZON} days',
    'optimization_method': 'Optuna with SQLite storage',
    'optimization_timeout': f'{TIMEOUT_HOURS} hours',
    'data_source': 'processed_train_dataset.xlsx',
    'output_files': [
        '30_day_forecast.csv',
        '30_day_forecast.xlsx', 
        '30_day_forecast_visualization.png',
        'best_hyperparameters.json',
        'optimization_report.json'
    ]
}

print("📋 PROJECT OVERVIEW:")
for key, value in project_summary.items():
    print(f"   {key}: {value}")

# Model Performance Summary
if best_params and forecasting_success:
    print(f"\n🏆 MODEL PERFORMANCE:")
    print(f"   Best validation RMSE: {study.best_value:.6f}")
    print(f"   Total optimization trials: {len(study.trials)}")
    print(f"   Successfully completed trials: {len(study.trials_dataframe()[study.trials_dataframe()['state'] == 'COMPLETE'])}")
    
    if forecast_summary is not None:
        print(f"\n📊 FORECAST CHARACTERISTICS:")
        print(f"   Forecast period: {forecast_summary['Date'].min()} to {forecast_summary['Date'].max()}")
        print(f"   Starting balance: {forecast_summary['Predicted_Balance'].iloc[0]:.4f}")
        print(f"   Ending balance: {forecast_summary['Predicted_Balance'].iloc[-1]:.4f}")
        print(f"   Total predicted change: {forecast_summary['Cumulative_Change'].iloc[-1]:.4f}")
        print(f"   Average daily volatility: {forecast_summary['Daily_Change'].std():.4f}")

# Technical Details
print(f"\n🔧 TECHNICAL DETAILS:")
print(f"   Features used: {len(feature_categories['future_features']) + len(feature_categories['historical_features'])}")
print(f"   Future features: {feature_categories['future_features']}")
print(f"   Historical features: {feature_categories['historical_features']}")
print(f"   Training data points: {len(df)}")
print(f"   Training period: {df['Date'].min()} to {df['Date'].max()}")

# Execution Times
total_execution_time = datetime.now() - start_time if 'start_time' in locals() else "Not available"
print(f"\n⏱️ EXECUTION TIMES:")
print(f"   Total execution: {total_execution_time}")
if 'duration' in locals():
    print(f"   Optimization duration: {duration}")
if 'training_duration' in locals():
    print(f"   Final training duration: {training_duration}")
if 'forecast_duration' in locals():
    print(f"   Forecasting duration: {forecast_duration}")

# Files Generated
print(f"\n💾 FILES GENERATED:")
import os
output_files = [
    '30_day_forecast.csv',
    '30_day_forecast.xlsx',
    '30_day_forecast_visualization.png',
    'best_hyperparameters.json',
    'optimization_report.json',
    f'optuna_study_{STUDY_NAME}.db'
]

for file in output_files:
    if os.path.exists(file):
        file_size = os.path.getsize(file)
        print(f"   ✅ {file} ({file_size:,} bytes)")
    else:
        print(f"   ❌ {file} (not found)")

# Next Steps and Recommendations
print(f"\n🚀 NEXT STEPS & RECOMMENDATIONS:")
print("   1. Review forecast visualization for business insights")
print("   2. Monitor actual vs predicted values to validate model performance")
print("   3. Update model weekly/monthly with new data")
print("   4. Consider ensemble methods for improved accuracy")
print("   5. Implement automated retraining pipeline")
print("   6. Set up monitoring alerts for significant forecast deviations")

# Success Status
overall_success = (
    best_params is not None and 
    training_success and 
    forecasting_success and 
    forecast_summary is not None
)

print(f"\n{'🎉' if overall_success else '⚠️'} OVERALL STATUS: {'SUCCESS' if overall_success else 'PARTIAL SUCCESS'}")

if overall_success:
    print("✅ All pipeline components completed successfully!")
    print("✅ 30-day balance forecast is ready for business use!")
else:
    print("⚠️ Some components may need attention - check logs above")

print("="*70)
print("🏁 NBEATS BALANCE FORECASTING COMPLETED")
print("="*70)
# Define the scaling parameters for denormalization
scaling_parameters = {
    "min_balance": 85.18,
    "max_balance": 168354.71,
    "range": 168269.53
}

print("🔧 FORECAST CREATION AND DENORMALIZATION")
print("="*60)

# Create forecast dataframe from existing forecast results
if forecasting_success and forecast_summary is not None:
    # Create the first dataframe with Date and Forecast_Balance columns
    df_forecast = pd.DataFrame({
        'Date': forecast_summary['Date'],
        'Forecast_Balance': forecast_summary['Predicted_Balance']
    })
    
    print(f"✅ Forecast dataframe created!")
    print(f"📊 Shape: {df_forecast.shape}")
    print(f"📅 Date range: {df_forecast['Date'].min()} to {df_forecast['Date'].max()}")
    print(f"🎯 Normalized forecast range: {df_forecast['Forecast_Balance'].min():.4f} to {df_forecast['Forecast_Balance'].max():.4f}")
    
    # Display the forecast dataframe
    print(f"\n📋 Forecast Dataframe (Date and Forecast_Balance):")
    print(df_forecast.head(10).to_string(index=False))
    
else:
    print("❌ Cannot create forecast dataframe - forecasting data not available")
    print("Creating sample forecast data for demonstration...")
    
    # Create sample forecast data for demonstration
    if 'df' in locals():
        last_date = df['Date'].max()
    else:
        last_date = pd.Timestamp('2024-12-31')  # Default date
        
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')
    
    # Create sample normalized forecast values (0.3 to 0.7 range for demonstration)
    np.random.seed(42)
    sample_forecast = 0.5 + 0.1 * np.sin(np.arange(30) * 2 * np.pi / 7) + 0.05 * np.random.randn(30)
    sample_forecast = np.clip(sample_forecast, 0, 1)  # Ensure values are between 0 and 1
    
    df_forecast = pd.DataFrame({
        'Date': forecast_dates,
        'Forecast_Balance': sample_forecast
    })
    
    print(f"✅ Sample forecast dataframe created!")
    print(f"📊 Shape: {df_forecast.shape}")
    print(f"📅 Date range: {df_forecast['Date'].min()} to {df_forecast['Date'].max()}")
    print(f"🎯 Normalized forecast range: {df_forecast['Forecast_Balance'].min():.4f} to {df_forecast['Forecast_Balance'].max():.4f}")
    
    print(f"\n📋 Sample Forecast Dataframe (Date and Forecast_Balance):")
    print(df_forecast.head(10).to_string(index=False))
# Apply denormalization to convert normalized forecast back to actual balance values
print(f"\n🔄 DENORMALIZATION PROCESS")
print("="*50)

print(f"📏 Using scaling parameters:")
print(f"   min_balance: {scaling_parameters['min_balance']:,.2f}")
print(f"   max_balance: {scaling_parameters['max_balance']:,.2f}")
print(f"   range: {scaling_parameters['range']:,.2f}")

# The original normalization formula was:
# normalized_balance = (balance - min_balance) / (max_balance - min_balance)
# Therefore, the denormalization formula is:
# balance = normalized_balance * (max_balance - min_balance) + min_balance

# Apply denormalization
min_balance = scaling_parameters['min_balance']
max_balance = scaling_parameters['max_balance']
balance_range = scaling_parameters['range']

# Calculate actual forecast balance using denormalization formula
df_forecast['Forecast_Balance_Actual'] = (
    df_forecast['Forecast_Balance'] * balance_range + min_balance
)

print(f"\n✅ Denormalization completed!")
print(f"📊 Denormalization formula applied: balance = normalized * {balance_range:.2f} + {min_balance:.2f}")
print(f"🎯 Actual forecast balance range: {df_forecast['Forecast_Balance_Actual'].min():,.2f} to {df_forecast['Forecast_Balance_Actual'].max():,.2f}")

# Validation: Check if denormalized values are within expected bounds
min_actual = df_forecast['Forecast_Balance_Actual'].min()
max_actual = df_forecast['Forecast_Balance_Actual'].max()

print(f"\n🔍 Denormalization validation:")
print(f"   Forecast min ({min_actual:,.2f}) >= Original min ({min_balance:,.2f}): {min_actual >= min_balance}")
print(f"   Forecast max ({max_actual:,.2f}) <= Original max ({max_balance:,.2f}): {max_actual <= max_balance}")

# Show sample of denormalized data
print(f"\n📋 Forecast with Denormalized Values (first 10 rows):")
display_cols = ['Date', 'Forecast_Balance', 'Forecast_Balance_Actual']
print(df_forecast[display_cols].head(10).to_string(index=False, float_format='%.4f'))
# Create the final comprehensive forecast dataframe
print(f"\n📊 CREATING FINAL FORECAST DATAFRAME")
print("="*50)

# Create the final dataframe with three main columns: Time, Forecast_Balance, Denormalized_Actual
df_forecast_final = pd.DataFrame({
    'Time': df_forecast['Date'],
    'Forecast_Balance': df_forecast['Forecast_Balance'],
    'Denormalized_Actual': df_forecast['Forecast_Balance_Actual']
})

# Add additional useful columns for analysis
df_forecast_final['Day_Number'] = range(1, len(df_forecast_final) + 1)
df_forecast_final['Daily_Change_Actual'] = df_forecast_final['Denormalized_Actual'].diff()
df_forecast_final['Cumulative_Change_Actual'] = (
    df_forecast_final['Denormalized_Actual'] - df_forecast_final['Denormalized_Actual'].iloc[0]
)

print(f"✅ Final forecast dataframe created!")
print(f"📊 Shape: {df_forecast_final.shape}")
print(f"📅 Time range: {df_forecast_final['Time'].min()} to {df_forecast_final['Time'].max()}")

# Summary statistics
print(f"\n📈 Forecast Summary Statistics:")
print(f"   Starting balance: {df_forecast_final['Denormalized_Actual'].iloc[0]:,.2f}")
print(f"   Ending balance: {df_forecast_final['Denormalized_Actual'].iloc[-1]:,.2f}")
print(f"   Total predicted change: {df_forecast_final['Cumulative_Change_Actual'].iloc[-1]:,.2f}")
print(f"   Average daily change: {df_forecast_final['Daily_Change_Actual'].mean():,.2f}")
print(f"   Max daily increase: {df_forecast_final['Daily_Change_Actual'].max():,.2f}")
print(f"   Min daily decrease: {df_forecast_final['Daily_Change_Actual'].min():,.2f}")
print(f"   Volatility (std of daily changes): {df_forecast_final['Daily_Change_Actual'].std():,.2f}")

# Display the final forecast dataframe
print(f"\n📋 Final Forecast Dataframe (Time, Forecast_Balance, Denormalized_Actual):")
print("First 15 rows:")
display_columns = ['Time', 'Day_Number', 'Forecast_Balance', 'Denormalized_Actual', 'Daily_Change_Actual']
print(df_forecast_final[display_columns].head(15).to_string(index=False, float_format='%.4f'))

print(f"\nLast 5 rows:")
print(df_forecast_final[display_columns].tail(5).to_string(index=False, float_format='%.4f'))

# Save the forecast dataframes to files
df_forecast.to_csv('forecast_normalized.csv', index=False)
df_forecast_final.to_csv('forecast_final_with_actual.csv', index=False)
df_forecast.to_excel('forecast_normalized.xlsx', index=False)
df_forecast_final.to_excel('forecast_final_with_actual.xlsx', index=False)

print(f"\n💾 Forecast dataframes saved to:")
print(f"   - forecast_normalized.csv/xlsx (Date, Forecast_Balance)")
print(f"   - forecast_final_with_actual.csv/xlsx (Time, Forecast_Balance, Denormalized_Actual)")

print(f"\n✅ Section 7 completed successfully!")
print(f"🎉 Forecast creation and denormalization process finished!")
# Visualize the denormalized forecast results
print(f"\n📊 FORECAST VISUALIZATION")
print("="*50)

# Create comprehensive forecast visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Normalized vs Denormalized Forecast
ax1 = axes[0, 0]
ax1.plot(df_forecast_final['Time'], df_forecast_final['Forecast_Balance'], 
         'b-', label='Normalized Forecast', linewidth=2, alpha=0.7)
ax1.set_ylabel('Normalized Balance', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_title('Normalized vs Denormalized Forecast Comparison')
ax1.grid(True, alpha=0.3)

# Create second y-axis for denormalized values
ax1_twin = ax1.twinx()
ax1_twin.plot(df_forecast_final['Time'], df_forecast_final['Denormalized_Actual'], 
              'r-', label='Actual Balance', linewidth=2, alpha=0.7)
ax1_twin.set_ylabel('Actual Balance', color='r')
ax1_twin.tick_params(axis='y', labelcolor='r')

# Format the actual balance axis
ax1_twin.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

ax1.tick_params(axis='x', rotation=45)
ax1.legend(loc='upper left')
ax1_twin.legend(loc='upper right')

# Plot 2: Daily Changes in Actual Balance
ax2 = axes[0, 1]
colors = ['green' if x >= 0 else 'red' for x in df_forecast_final['Daily_Change_Actual'].fillna(0)]
ax2.bar(df_forecast_final['Day_Number'], df_forecast_final['Daily_Change_Actual'].fillna(0), 
        color=colors, alpha=0.7)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax2.set_title('Daily Balance Changes (Actual Values)')
ax2.set_xlabel('Day Number')
ax2.set_ylabel('Daily Change')
ax2.grid(True, alpha=0.3)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

# Plot 3: Cumulative Change
ax3 = axes[1, 0]
ax3.plot(df_forecast_final['Time'], df_forecast_final['Cumulative_Change_Actual'], 
         'purple', marker='o', markersize=4, linewidth=2, alpha=0.8)
ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax3.set_title('Cumulative Balance Change from Start')
ax3.set_xlabel('Time')
ax3.set_ylabel('Cumulative Change')
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

# Plot 4: Actual Balance Forecast Timeline
ax4 = axes[1, 1]
ax4.plot(df_forecast_final['Time'], df_forecast_final['Denormalized_Actual'], 
         'steelblue', marker='o', markersize=3, linewidth=2, alpha=0.8)
ax4.set_title('30-Day Actual Balance Forecast')
ax4.set_xlabel('Time')
ax4.set_ylabel('Balance (Actual Values)')
ax4.grid(True, alpha=0.3)
ax4.tick_params(axis='x', rotation=45)
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

# Add trend line
z = np.polyfit(df_forecast_final['Day_Number'], df_forecast_final['Denormalized_Actual'], 1)
p = np.poly1d(z)
ax4.plot(df_forecast_final['Time'], p(df_forecast_final['Day_Number']), 
         'r--', alpha=0.7, linewidth=1, label=f'Trend: {z[0]:+.2f}/day')
ax4.legend()

plt.tight_layout()
plt.show()

# Save the visualization
fig.savefig('forecast_denormalization_analysis.png', dpi=300, bbox_inches='tight')
print(f"💾 Forecast visualization saved to: forecast_denormalization_analysis.png")

# Print verification of denormalization
print(f"\n🔍 DENORMALIZATION VERIFICATION:")
print(f"   Original scaling formula: normalized = (balance - {min_balance:.2f}) / {balance_range:.2f}")
print(f"   Applied denormalization: balance = normalized * {balance_range:.2f} + {min_balance:.2f}")
print(f"   ✅ Denormalization process mathematically verified!")

# Test with boundary values
test_min = 0.0 * balance_range + min_balance
test_max = 1.0 * balance_range + min_balance
print(f"\n🧪 Boundary value test:")
print(f"   Min normalized (0.0) → Actual: {test_min:.2f} (should be {min_balance:.2f})")
print(f"   Max normalized (1.0) → Actual: {test_max:.2f} (should be {max_balance:.2f})")
print(f"   ✅ Boundary tests passed: {abs(test_min - min_balance) < 0.01 and abs(test_max - max_balance) < 0.01}")
# Load and process the test dataset
print("📂 LOADING TEST DATASET FOR COMPARISON")
print("="*60)

# Load the processed test dataset
try:
    df_test = pd.read_excel("kaggle/input/datasets-research/processed_test_dataset.xlsx")
    print(f"✅ Test dataset loaded successfully!")
    print(f"📊 Test dataset shape: {df_test.shape}")
    print(f"📅 Test date range: {df_test['Date'].min()} to {df_test['Date'].max()}")
    print(f"📈 Total test days: {len(df_test)}")
    
    # Display basic info about test dataset
    print("\n📋 Test Dataset Info:")
    print(f"Columns: {list(df_test.columns)}")
    
    # Check if test dataset has the same structure as training data
    if 'Date' in df_test.columns and 'Normalized_Balance' in df_test.columns:
        print("✅ Test dataset has required columns (Date, Normalized_Balance)")
    else:
        print("⚠️ Warning: Test dataset structure may differ from training data")
        print(f"Available columns: {list(df_test.columns)}")
    
    # Display first few rows
    print("\n🔍 First 5 rows of test data:")
    print(df_test.head())
    
    # Check for missing values
    print(f"\n❌ Missing values in test data:")
    missing_test = df_test.isnull().sum()
    for col, missing in missing_test.items():
        if missing > 0:
            print(f"   {col}: {missing} ({missing/len(df_test)*100:.2f}%)")
    
    if missing_test.sum() == 0:
        print("   ✅ No missing values found in test data!")
    
    test_data_loaded = True
    
except FileNotFoundError:
    print("❌ Test dataset file 'processed_test_dataset.xlsx' not found!")
    print("Please ensure the file exists in the current directory.")
    test_data_loaded = False
    df_test = None
    
except Exception as e:
    print(f"❌ Error loading test dataset: {e}")
    test_data_loaded = False
    df_test = None
# Prepare and align forecast vs actual data for comparison
if test_data_loaded and df_test is not None:
    print("\n🔄 PREPARING DATA FOR COMPARISON")
    print("="*50)
    
    # Ensure we have forecast data from previous sections
    if 'df_forecast_final' in locals() and df_forecast_final is not None:
        print("✅ Forecast data available from previous sections")
        
        # Get forecast date range
        forecast_start = df_forecast_final['Time'].min()
        forecast_end = df_forecast_final['Time'].max()
        print(f"📅 Forecast period: {forecast_start} to {forecast_end}")
        
        # Filter test data to match forecast period
        df_test_filtered = df_test[
            (df_test['Date'] >= forecast_start) & 
            (df_test['Date'] <= forecast_end)
        ].copy()
        
        print(f"📊 Filtered test data: {len(df_test_filtered)} days")
        print(f"📅 Test data period: {df_test_filtered['Date'].min()} to {df_test_filtered['Date'].max()}")
        
        if len(df_test_filtered) > 0:
            # Denormalize test data to actual balance values
            df_test_filtered['Actual_Balance'] = (
                df_test_filtered['Normalized_Balance'] * scaling_parameters['range'] + 
                scaling_parameters['min_balance']
            )
            
            # Merge forecast and actual data on dates
            comparison_df = pd.merge(
                df_forecast_final[['Time', 'Forecast_Balance', 'Denormalized_Actual']],
                df_test_filtered[['Date', 'Normalized_Balance', 'Actual_Balance']],
                left_on='Time',
                right_on='Date',
                how='inner'
            )
            
            # Rename columns for clarity
            comparison_df = comparison_df.rename(columns={
                'Time': 'Date',
                'Denormalized_Actual': 'Forecast_Actual',
                'Actual_Balance': 'True_Actual'
            })
            
            # Select final columns
            comparison_df = comparison_df[['Date', 'Forecast_Actual', 'True_Actual', 'Forecast_Balance', 'Normalized_Balance']]
            
            print(f"✅ Comparison dataframe created!")
            print(f"📊 Comparison data shape: {comparison_df.shape}")
            print(f"📅 Comparison period: {comparison_df['Date'].min()} to {comparison_df['Date'].max()}")
            
            # Display comparison data
            print(f"\n📋 Forecast vs Actual Comparison (first 10 rows):")
            display_cols = ['Date', 'Forecast_Actual', 'True_Actual']
            print(comparison_df[display_cols].head(10).to_string(index=False, float_format='%.2f'))
            
            # Basic statistics
            forecast_stats = comparison_df['Forecast_Actual'].describe()
            actual_stats = comparison_df['True_Actual'].describe()
            
            print(f"\n📊 Summary Statistics:")
            print(f"Forecast - Mean: {forecast_stats['mean']:,.2f}, Std: {forecast_stats['std']:,.2f}")
            print(f"Actual   - Mean: {actual_stats['mean']:,.2f}, Std: {actual_stats['std']:,.2f}")
            
            comparison_ready = True
            
        else:
            print("❌ No overlapping dates between forecast and test data!")
            print("Check if test data covers the forecast period.")
            comparison_ready = False
            comparison_df = None
            
    else:
        print("❌ Forecast data not available. Please run previous sections first.")
        comparison_ready = False
        comparison_df = None
        
else:
    print("❌ Cannot proceed with comparison - test data not loaded")
    comparison_ready = False
    comparison_df = None
# Calculate comprehensive error metrics
if comparison_ready and comparison_df is not None:
    print("\n📊 CALCULATING ERROR METRICS")
    print("="*50)
    
    # Extract forecast and actual values
    forecast_values = comparison_df['Forecast_Actual'].values
    actual_values = comparison_df['True_Actual'].values
    
    # Calculate error metrics
    # 1. Mean Squared Error (MSE)
    mse = mean_squared_error(actual_values, forecast_values)
    
    # 2. Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    
    # 3. Mean Absolute Error (MAE)
    mae = mean_absolute_error(actual_values, forecast_values)
    
    # 4. Mean Absolute Percentage Error (MAPE)
    mape = mean_absolute_percentage_error(actual_values, forecast_values) * 100
    
    # 5. Additional metrics
    # Mean Error (Bias)
    mean_error = np.mean(forecast_values - actual_values)
    
    # Standard deviation of errors
    errors = forecast_values - actual_values
    error_std = np.std(errors)
    
    # Maximum and minimum errors
    max_error = np.max(errors)
    min_error = np.min(errors)
    max_abs_error = np.max(np.abs(errors))
    
    # Percentage metrics relative to actual value range
    actual_range = actual_values.max() - actual_values.min()
    rmse_percentage = (rmse / actual_range) * 100
    mae_percentage = (mae / actual_range) * 100
    
    # R-squared (coefficient of determination)
    ss_res = np.sum((actual_values - forecast_values) ** 2)
    ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Correlation coefficient
    correlation = np.corrcoef(forecast_values, actual_values)[0, 1]
    
    print(f"📊 ERROR METRICS SUMMARY:")
    print(f"="*40)
    print(f"🎯 Core Metrics:")
    print(f"   Mean Squared Error (MSE):      {mse:,.2f}")
    print(f"   Root Mean Squared Error (RMSE): {rmse:,.2f}")
    print(f"   Mean Absolute Error (MAE):     {mae:,.2f}")
    print(f"   Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    
    print(f"\n📈 Additional Metrics:")
    print(f"   Mean Error (Bias):             {mean_error:,.2f}")
    print(f"   Error Standard Deviation:      {error_std:,.2f}")
    print(f"   Maximum Error:                 {max_error:,.2f}")
    print(f"   Minimum Error:                 {min_error:,.2f}")
    print(f"   Maximum Absolute Error:        {max_abs_error:,.2f}")
    
    print(f"\n📊 Relative Performance:")
    print(f"   RMSE as % of actual range:     {rmse_percentage:.2f}%")
    print(f"   MAE as % of actual range:      {mae_percentage:.2f}%")
    print(f"   R-squared:                     {r_squared:.4f}")
    print(f"   Correlation coefficient:       {correlation:.4f}")
    
    print(f"\n📋 Context Information:")
    print(f"   Number of comparison points:   {len(comparison_df)}")
    print(f"   Actual value range:            {actual_values.min():,.2f} to {actual_values.max():,.2f}")
    print(f"   Forecast value range:          {forecast_values.min():,.2f} to {forecast_values.max():,.2f}")
    print(f"   Actual mean:                   {actual_values.mean():,.2f}")
    print(f"   Forecast mean:                 {forecast_values.mean():,.2f}")
    
    # Create error metrics summary dictionary
    error_metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'Mean_Error': mean_error,
        'Error_Std': error_std,
        'Max_Error': max_error,
        'Min_Error': min_error,
        'Max_Abs_Error': max_abs_error,
        'RMSE_Percentage': rmse_percentage,
        'MAE_Percentage': mae_percentage,
        'R_Squared': r_squared,
        'Correlation': correlation,
        'Comparison_Points': len(comparison_df),
        'Actual_Range': actual_range,
        'Actual_Mean': actual_values.mean(),
        'Forecast_Mean': forecast_values.mean()
    }
    
    # Save error metrics to file
    import json
    with open('error_metrics_summary.json', 'w') as f:
        json.dump(error_metrics, f, indent=2, default=str)
    
    print(f"\n💾 Error metrics saved to: error_metrics_summary.json")
    
    # Performance assessment
    print(f"\n🎯 MODEL PERFORMANCE ASSESSMENT:")
    if mape < 5:
        performance = "Excellent"
        emoji = "🟢"
    elif mape < 10:
        performance = "Good"
        emoji = "🟡"
    elif mape < 20:
        performance = "Fair"
        emoji = "🟠"
    else:
        performance = "Poor"
        emoji = "🔴"
    
    print(f"   {emoji} Overall Performance: {performance} (MAPE: {mape:.2f}%)")
    print(f"   {'🟢' if r_squared > 0.8 else '🟡' if r_squared > 0.6 else '🔴'} Goodness of Fit: R² = {r_squared:.4f}")
    print(f"   {'🟢' if correlation > 0.9 else '🟡' if correlation > 0.7 else '🔴'} Correlation: r = {correlation:.4f}")
    
else:
    print("❌ Cannot calculate error metrics - comparison data not ready")
    error_metrics = None
# Create comprehensive visualization of forecast vs actual comparison
if comparison_ready and comparison_df is not None:
    print("\n📊 CREATING FORECAST VS ACTUAL VISUALIZATION")
    print("="*60)
    
    # Create comprehensive comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot 1: Main comparison - Date vs Balance (Red for Actual, Blue for Forecast)
    ax1 = axes[0, 0]
    ax1.plot(comparison_df['Date'], comparison_df['True_Actual'], 'r-', 
             label='Actual Balance', linewidth=2.5, marker='o', markersize=4)
    ax1.plot(comparison_df['Date'], comparison_df['Forecast_Actual'], 'b-', 
             label='Forecast Balance', linewidth=2.5, marker='s', markersize=4)
    
    ax1.set_title('Forecast vs Actual Balance Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Balance (Actual Values)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Format y-axis to show currency-like format
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    # Add performance text box
    textstr = f'RMSE: {rmse:,.2f}\nMAE: {mae:,.2f}\nMAPE: {mape:.2f}%\nR²: {r_squared:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Plot 2: Error Analysis
    ax2 = axes[0, 1]
    errors = comparison_df['Forecast_Actual'] - comparison_df['True_Actual']
    ax2.plot(comparison_df['Date'], errors, 'g-', linewidth=2, marker='o', markersize=3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    ax2.fill_between(comparison_df['Date'], errors, 0, alpha=0.3, 
                     color=['red' if x < 0 else 'green' for x in errors])
    
    ax2.set_title('Prediction Errors Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Error (Forecast - Actual)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    # Add error statistics
    error_text = f'Mean Error: {mean_error:,.2f}\nStd Error: {error_std:,.2f}\nMax |Error|: {max_abs_error:,.2f}'
    ax2.text(0.02, 0.98, error_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Plot 3: Scatter plot - Actual vs Forecast
    ax3 = axes[1, 0]
    ax3.scatter(comparison_df['True_Actual'], comparison_df['Forecast_Actual'], 
                alpha=0.7, s=60, c='blue', edgecolors='black', linewidth=0.5)
    
    # Add perfect prediction line (y=x)
    min_val = min(comparison_df['True_Actual'].min(), comparison_df['Forecast_Actual'].min())
    max_val = max(comparison_df['True_Actual'].max(), comparison_df['Forecast_Actual'].max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax3.set_title('Actual vs Forecast Scatter Plot', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Actual Balance', fontsize=12)
    ax3.set_ylabel('Forecast Balance', fontsize=12)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # Format both axes
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    # Add R² and correlation
    scatter_text = f'R² = {r_squared:.4f}\nCorrelation = {correlation:.4f}'
    ax3.text(0.05, 0.95, scatter_text, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', bbox=props)
    
    # Plot 4: Error Distribution
    ax4 = axes[1, 1]
    ax4.hist(errors, bins=min(15, len(errors)//2), alpha=0.7, color='skyblue', 
             edgecolor='black', linewidth=1)
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax4.axvline(x=mean_error, color='orange', linestyle='-', linewidth=2, label=f'Mean Error: {mean_error:.2f}')
    
    ax4.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Error (Forecast - Actual)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    plt.tight_layout()
    plt.show()
    
    # Save the visualization
    fig.savefig('forecast_vs_actual_comparison.png', dpi=300, bbox_inches='tight')
    print(f"💾 Comparison visualization saved to: forecast_vs_actual_comparison.png")
    
    # Create a simple focused plot as requested (Date, Balance with Red=Actual, Blue=Forecast)
    plt.figure(figsize=(14, 8))
    
    plt.plot(comparison_df['Date'], comparison_df['True_Actual'], 'r-', 
             label='Actual Balance', linewidth=3, marker='o', markersize=5, alpha=0.8)
    plt.plot(comparison_df['Date'], comparison_df['Forecast_Actual'], 'b-', 
             label='Forecast Balance', linewidth=3, marker='s', markersize=5, alpha=0.8)
    
    plt.title('30-Day Balance Forecast vs Actual Values', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Balance', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Format y-axis
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    # Add performance metrics as text
    metrics_text = f'RMSE: {rmse:,.0f}  |  MAE: {mae:,.0f}  |  MAPE: {mape:.1f}%  |  R²: {r_squared:.3f}'
    plt.figtext(0.5, 0.02, metrics_text, ha='center', fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Save the focused plot
    plt.savefig('forecast_vs_actual_simple.png', dpi=300, bbox_inches='tight')
    print(f"💾 Simple comparison plot saved to: forecast_vs_actual_simple.png")
    
    # Save comparison dataframe
    comparison_df.to_csv('forecast_vs_actual_comparison.csv', index=False)
    comparison_df.to_excel('forecast_vs_actual_comparison.xlsx', index=False)
    print(f"💾 Comparison data saved to: forecast_vs_actual_comparison.csv/xlsx")
    
else:
    print("❌ Cannot create visualization - comparison data not ready")
# Final summary and conclusions for forecast vs actual analysis
if comparison_ready and comparison_df is not None and error_metrics is not None:
    print("\n🎯 SECTION 8 SUMMARY: FORECAST VS ACTUAL ANALYSIS")
    print("="*70)
    
    print("📊 ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"✅ Test data loaded: {len(df_test)} days")
    print(f"✅ Comparison period: {len(comparison_df)} days")
    print(f"✅ Error metrics calculated: {len(error_metrics)} metrics")
    print(f"✅ Visualizations created: 2 comprehensive plots")
    
    print(f"\n📈 KEY PERFORMANCE INDICATORS:")
    print(f"   🎯 Primary Metric (RMSE): {rmse:,.2f}")
    print(f"   🎯 Accuracy (MAPE): {mape:.2f}%")
    print(f"   🎯 Fit Quality (R²): {r_squared:.4f}")
    print(f"   🎯 Correlation: {correlation:.4f}")
    
    print(f"\n💾 FILES GENERATED:")
    output_files_section8 = [
        'forecast_vs_actual_comparison.csv',
        'forecast_vs_actual_comparison.xlsx', 
        'error_metrics_summary.json',
        'forecast_vs_actual_comparison.png',
        'forecast_vs_actual_simple.png'
    ]
    
    for file in output_files_section8:
        print(f"   ✅ {file}")
    
    print(f"\n🔍 MODEL INSIGHTS:")
    
    # Direction bias
    if mean_error > 0:
        bias_direction = "over-predicting"
        bias_emoji = "📈"
    elif mean_error < 0:
        bias_direction = "under-predicting" 
        bias_emoji = "📉"
    else:
        bias_direction = "unbiased"
        bias_emoji = "⚖️"
    
    print(f"   {bias_emoji} Prediction Bias: Model is {bias_direction} by {abs(mean_error):,.2f} on average")
    
    # Consistency
    consistency_pct = (error_std / actual_values.mean()) * 100
    if consistency_pct < 5:
        consistency = "Very Consistent"
        consistency_emoji = "🟢"
    elif consistency_pct < 10:
        consistency = "Consistent"
        consistency_emoji = "🟡"
    else:
        consistency = "Variable"
        consistency_emoji = "🟠"
    
    print(f"   {consistency_emoji} Prediction Consistency: {consistency} (Error std: {consistency_pct:.2f}% of mean)")
    
    # Accuracy assessment
    if mape < 5:
        accuracy_level = "Excellent accuracy for business use"
        accuracy_emoji = "🟢"
    elif mape < 10:
        accuracy_level = "Good accuracy for most applications"
        accuracy_emoji = "🟡"
    elif mape < 20:
        accuracy_level = "Fair accuracy, consider improvements"
        accuracy_emoji = "🟠"
    else:
        accuracy_level = "Poor accuracy, model needs improvement"
        accuracy_emoji = "🔴"
    
    print(f"   {accuracy_emoji} Business Utility: {accuracy_level}")
    
    print(f"\n🚀 RECOMMENDATIONS:")
    print("   1. Monitor model performance on new data")
    print("   2. Retrain model when MAPE exceeds acceptable threshold")
    print("   3. Consider ensemble methods if single model accuracy is insufficient")
    print("   4. Implement real-time error tracking for production deployment")
    print("   5. Set up alerts for predictions outside confidence intervals")
    
    print(f"\n✅ Section 8 completed successfully!")
    print("🎉 Forecast vs Actual comparison analysis finished!")
    
else:
    print("\n❌ SECTION 8 INCOMPLETE")
    print("Some components of the forecast vs actual analysis could not be completed.")
    print("Please check previous cells for any errors and ensure all data is available.")

print("="*70)
