# Fixed Parameter NBEATSx Model for Balance Forecasting
# Using optimized hyperparameters from hyperparameter tuning

import numpy as np
import pandas as pd
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

# Date and time
from datetime import datetime, timedelta
import json

# Set environment variables for GPU optimization
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

print("🚀 Fixed Parameter NBEATSx Model")
print("="*50)
print("Using optimized hyperparameters from hyperparameter tuning")
print("="*50)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# GPU Configuration
print("🔧 GPU Configuration")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

# BEST HYPERPARAMETERS FROM OPTIMIZATION
BEST_PARAMS = {
    'input_size': 174,
    'learning_rate': 0.0012908542013841148,
    'max_steps': 1314,
    'batch_size': 20,
    'n_harmonics': 4,
    'n_polynomials': 4,
    'dropout_prob_theta': 0.07362537064064913,
    'n_blocks': [6, 2, 6]  # [identity, trend, seasonality]
}

print(f"\n🎯 Using Best Hyperparameters:")
for param, value in BEST_PARAMS.items():
    print(f"   {param}: {value}")

# Configuration
HORIZON = 30  # Forecast horizon (days)

print(f"\n📊 Configuration:")
print(f"   Forecast horizon: {HORIZON} days")

# Load the preprocessed dataset
print("\n📂 Loading preprocessed dataset...")
df = pd.read_excel("/kaggle/input/datasets-research/processed_train_dataset.xlsx")

print(f"✅ Dataset loaded successfully!")
print(f"📊 Dataset shape: {df.shape}")
print(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"📈 Total days: {len(df)}")

# Data validation and feature identification
print("\n🔍 Data Validation and Feature Analysis")
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

# Prepare full dataset
full_nf_data = prepare_neural_forecast_data(df, feature_categories)
print(f"\n📊 Full training dataset: {len(full_nf_data)} days")
print(f"📅 Training period: {full_nf_data['ds'].min()} to {full_nf_data['ds'].max()}")

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

# Create final model with best parameters
print("\n🚀 Creating Final Model with Best Parameters")
print("="*60)

final_model = NBEATSx(
    h=HORIZON,
    input_size=BEST_PARAMS['input_size'],
    futr_exog_list=feature_categories['future_features'],
    hist_exog_list=feature_categories['historical_features'],
    
    # Architecture parameters
    stack_types=['identity', 'trend', 'seasonality'],
    n_blocks=BEST_PARAMS['n_blocks'],
    n_harmonics=BEST_PARAMS['n_harmonics'],
    n_polynomials=BEST_PARAMS['n_polynomials'],
    
    # Training parameters
    learning_rate=BEST_PARAMS['learning_rate'],
    max_steps=BEST_PARAMS['max_steps'],
    batch_size=BEST_PARAMS['batch_size'],
    dropout_prob_theta=BEST_PARAMS['dropout_prob_theta'],
    
    # Other settings
    random_seed=42,
    scaler_type='standard',
    loss=DistributionLoss(distribution='Normal', level=[80, 90, 95])
)

# Create final forecaster
final_forecaster = NeuralForecast(
    models=[final_model], 
    freq='D'
)

# Train the final model
print(f"\n🎓 Training final model with best parameters...")
training_start = datetime.now()

try:
    final_forecaster.fit(df=full_nf_data)
    training_success = True
    print("✅ Final model training completed successfully!")
    
except Exception as e:
    training_success = False
    print(f"❌ Final model training failed: {e}")

training_end = datetime.now()
training_duration = training_end - training_start
print(f"⏱️ Training duration: {training_duration}")

if training_success:
    print("\n🎉 Final model ready for forecasting!")
    
    # Generate 30-day forecast
    print("\n🔮 Generating 30-Day Balance Forecast")
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
        forecast_summary.to_csv('fixed_parameter_forecast.csv', index=False)
        forecast_summary.to_excel('fixed_parameter_forecast.xlsx', index=False)
        print(f"\n💾 Forecast saved to:")
        print(f"   - fixed_parameter_forecast.csv")
        print(f"   - fixed_parameter_forecast.xlsx")
        
        # Save best parameters
        with open('best_parameters_used.json', 'w') as f:
            json.dump(BEST_PARAMS, f, indent=2, default=str)
        print(f"   - best_parameters_used.json")
        
    else:
        print("❌ Cannot proceed with analysis - forecasting failed")
        forecast_summary = None

else:
    print("❌ Cannot generate forecast - final model training failed")
    forecast_summary = None

# Final Summary
print("\n🎉 FIXED PARAMETER MODEL SUMMARY")
print("="*70)

if training_success and forecasting_success and forecast_summary is not None:
    print("✅ SUCCESS: Model trained and forecast generated!")
    print(f"🎯 Best hyperparameters used: {len(BEST_PARAMS)} parameters")
    print(f"📊 Forecast horizon: {HORIZON} days")
    print(f"⏱️ Training time: {training_duration}")
    print(f"⏱️ Forecasting time: {forecast_duration}")
    
    print(f"\n📈 FORECAST RESULTS:")
    print(f"   Starting balance: {forecast_summary['Predicted_Balance'].iloc[0]:.4f}")
    print(f"   Ending balance: {forecast_summary['Predicted_Balance'].iloc[-1]:.4f}")
    print(f"   Total change: {forecast_summary['Cumulative_Change'].iloc[-1]:.4f}")
    print(f"   Average daily volatility: {forecast_summary['Daily_Change'].std():.4f}")
    
    print(f"\n💾 FILES GENERATED:")
    print("   ✅ fixed_parameter_forecast.csv")
    print("   ✅ fixed_parameter_forecast.xlsx")
    print("   ✅ best_parameters_used.json")
    
    print(f"\n🚀 MODEL READY FOR PRODUCTION USE!")
    
else:
    print("❌ MODEL TRAINING OR FORECASTING FAILED")
    print("Please check the error messages above and try again.")

print("="*70)
print("🏁 Fixed Parameter Model Completed")
print("="*70) 