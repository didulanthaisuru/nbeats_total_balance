# N-BEATS Notebook Usage Guide

A comprehensive step-by-step guide for using the N-BEATS Total Balance Prediction system. This guide explains how to navigate through the notebooks, understand the workflow, and interpret results.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Notebook Execution Order](#notebook-execution-order)
4. [Detailed Notebook Guide](#detailed-notebook-guide)
5. [Understanding Results](#understanding-results)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Usage](#advanced-usage)

## 🎯 Overview

The N-BEATS project consists of 4 main notebooks that work together to create a complete time series forecasting pipeline:

```
Data Flow:
Raw Data → Preprocessing → Model Training → Validation → Results
    ↓           ↓              ↓             ↓          ↓
   📊      f_data_prep    f_forecasting   compare   📈 Analysis
```

## 🔧 Prerequisites

### Environment Setup
```bash
# Install required packages
pip install neuralforecast pandas numpy matplotlib seaborn scikit-learn optuna openpyxl jupyter

# Start Jupyter
jupyter notebook
```

### Data Requirements
- **Training Data**: `shihara_train_30days_version2.xlsx` - Historical balance data for model training
- **Test Data**: `shihara_test_30days_version2.xlsx` - Recent 30 days for validation
- **Format**: Excel files with Date and Balance columns

## 📚 Notebook Execution Order

**IMPORTANT**: Execute notebooks in this exact order for correct results:

### 1️⃣ Data Preprocessing
**File**: `f_data_preprocessing.ipynb`
**Purpose**: Prepare raw data for model training

### 2️⃣ Model Training & Forecasting  
**File**: `f_forecasting.ipynb`
**Purpose**: Train optimized model and generate predictions

### 3️⃣ Model Validation
**File**: `compare.ipynb`  
**Purpose**: Validate model performance against actual test data

### 4️⃣ Hyperparameter Experiments *(Optional)*
**File**: `prediction_hyperparameter.ipynb`
**Purpose**: Additional parameter tuning experiments

---

## 📖 Detailed Notebook Guide

## 1️⃣ Data Preprocessing (`f_data_preprocessing.ipynb`)

### What This Notebook Does:
- Loads raw training and test data
- Creates engineered features (cyclical encoding, lag features, rolling statistics)
- Normalizes data for model training
- Splits data appropriately for training/validation
- Saves preprocessed datasets and scaling parameters

### Key Sections:

#### 📂 **Section 1: Data Loading**
```python
# Load raw data files
train_data = pd.read_excel('shihara_train_30days_version2.xlsx')
test_data = pd.read_excel('shihara_test_30days_version2.xlsx')
```

**What to Check:**
- ✅ Data files load successfully
- ✅ Date columns are recognized as datetime
- ✅ No missing values in critical columns
- ✅ Data ranges make sense

#### 🔧 **Section 2: Feature Engineering**
```python
# Cyclical encoding for day of week
df['dayofweek_sin'] = np.sin(2 * np.pi * df['Date'].dt.dayofweek / 7)
df['dayofweek_cos'] = np.cos(2 * np.pi * df['Date'].dt.dayofweek / 7)

# Lag features
df['balance_1d_ago'] = df['Balance'].shift(1)
df['balance_7d_ago'] = df['Balance'].shift(7)

# Rolling statistics
df['rolling_mean_30d'] = df['Balance'].rolling(window=30).mean()
```

**What to Check:**
- ✅ All engineered features are created
- ✅ No excessive missing values after feature creation
- ✅ Feature distributions look reasonable

#### 📊 **Section 3: Data Normalization**
```python
# Min-max normalization
scaler = MinMaxScaler()
df['Normalized_Balance'] = scaler.fit_transform(df[['Balance']])
```

**What to Check:**
- ✅ Normalized values are between 0 and 1
- ✅ Scaling parameters are saved to `scaling_parameters.json`
- ✅ Original balance values can be recovered

### 📁 **Output Files:**
- `processed_train_dataset.csv` / `.xlsx` - Training data with features
- `processed_test_dataset.csv` / `.xlsx` - Test data with features  
- `scaling_parameters.json` - Normalization parameters for denormalization

### ⏱️ **Expected Runtime:** 2-5 minutes

---

## 2️⃣ Model Training & Forecasting (`f_forecasting.ipynb`)

### What This Notebook Does:
- Loads preprocessed training data
- Optimizes model hyperparameters using Optuna (11-hour timeout)
- Trains final NBEATSx model with best parameters
- Generates 30-day forecast with confidence intervals
- Creates comprehensive visualizations

### Key Sections:

#### 📂 **Section 1: Data Loading & Validation**
```python
# Load preprocessed data
df = pd.read_excel("processed_train_dataset.xlsx")

# Validate data quality
required_columns = ['Date', 'Normalized_Balance']
feature_categories = {
    'future_features': ['dayofweek_sin', 'dayofweek_cos'],
    'historical_features': ['balance_1d_ago', 'balance_7d_ago', 'rolling_mean_30d']
}
```

**What to Check:**
- ✅ Preprocessed data loads correctly
- ✅ All required features are present
- ✅ Date ranges and data quality look good

#### 🔬 **Section 2: Hyperparameter Optimization** *(Longest Section)*
```python
# Optuna optimization setup
STUDY_NAME = "nbeats_balance_forecasting"
TIMEOUT_HOURS = 11
DB_URL = f"sqlite:///optuna_study_{STUDY_NAME}.db"

# Optimization objective function
def objective(trial):
    # Model parameters to optimize
    params = {
        'input_size': trial.suggest_int('input_size', 80, 230),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'max_steps': trial.suggest_int('max_steps', 800, 2000),
        'batch_size': trial.suggest_categorical('batch_size', [16, 24, 32, 48, 64]),
        # ... more parameters
    }
    return rmse_score  # Minimize RMSE
```

**What to Check:**
- ✅ Optimization runs without errors
- ✅ Study database is created
- ✅ RMSE improves over trials
- ✅ Best parameters are reasonable

**⚠️ Important Notes:**
- This section runs for up to 11 hours
- Progress is saved to SQLite database
- Can be interrupted and resumed
- Monitor RMSE improvement trends

#### 🤖 **Section 3: Final Model Training**
```python
# Train with best hyperparameters
model = NBEATSx(
    h=HORIZON,
    input_size=best_params['input_size'],
    learning_rate=best_params['learning_rate'],
    # ... other optimized parameters
)

# Fit the model
nf = NeuralForecast(models=[model], freq='D')
nf.fit(train_df)
```

**What to Check:**
- ✅ Model trains successfully
- ✅ Training convergence looks stable
- ✅ No GPU memory issues

#### 📈 **Section 4: Forecasting**
```python
# Generate 30-day forecast
forecast = nf.predict(futr_df=future_features_df)

# Create forecast summary with confidence intervals
forecast_summary = pd.DataFrame({
    'Date': future_features_df['ds'],
    'Predicted_Balance': point_forecast,
    'Lower_CI_80': forecast['NBEATSx-lo-80'].values,
    'Upper_CI_80': forecast['NBEATSx-hi-80'].values,
    # ... more confidence intervals
})
```

**What to Check:**
- ✅ 30-day predictions are generated
- ✅ Confidence intervals look reasonable
- ✅ Predictions are in normalized scale
- ✅ All forecast dates are correct

#### 💾 **Section 5: Data Denormalization & Saving**
```python
# Load scaling parameters
with open('scaling_parameters.json', 'r') as f:
    scaling_params = json.load(f)

# Denormalize predictions
def denormalize_balance(normalized_value, min_val, max_val):
    return normalized_value * (max_val - min_val) + min_val

df_forecast_final['Denormalized_Actual'] = denormalize_balance(
    df_forecast_final['Forecast_Balance'],
    scaling_params['min_balance'], 
    scaling_params['max_balance']
)
```

**What to Check:**
- ✅ Denormalized values are in original balance scale
- ✅ Forecast values look realistic
- ✅ All output files are saved correctly

### 📁 **Output Files:**
- `30_day_forecast.csv` / `.xlsx` - Clean forecast with confidence intervals
- `forecast_normalized.csv` / `.xlsx` - Normalized predictions
- `forecast_final_with_actual.csv` / `.xlsx` - Denormalized predictions ready for comparison
- `best_hyperparameters.json` - Optimized model parameters
- `optimization_report.json` - Optimization summary
- `optuna_study_nbeats_balance_forecasting.db` - Complete optimization history
- `30_day_forecast_visualization.png` - Forecast visualization

### ⏱️ **Expected Runtime:** 11-13 hours (mostly optimization)

---

## 3️⃣ Model Validation (`compare.ipynb`)

### What This Notebook Does:
- Loads actual test data and model predictions
- Performs comprehensive comparison analysis
- Calculates detailed error metrics
- Creates publication-quality visualizations
- Generates summary reports

### Key Sections:

#### 📂 **Section 1: Data Loading**
```python
# Load test dataset and forecasts
test_data = pd.read_csv('processed_test_dataset.csv')
forecast_data = pd.read_csv('forecast_final_with_actual.csv')

# Load scaling parameters for denormalization
with open('scaling_parameters.json', 'r') as f:
    scaling_params = json.load(f)
```

**What to Check:**
- ✅ Both test and forecast data load correctly
- ✅ Date ranges overlap appropriately
- ✅ Scaling parameters match preprocessing

#### 🔧 **Section 2: Data Alignment & Denormalization**
```python
# Denormalize forecast values
forecast_data['Forecast_Balance_Denormalized'] = denormalize_balance(
    forecast_data['Forecast_Balance'], 
    scaling_params['min_balance'], 
    scaling_params['max_balance']
)

# Merge datasets for comparison
comparison_df = pd.merge(
    test_data[['Time', 'Actual_Test_Balance']], 
    forecast_data[['Time', 'Forecast_Balance_Denormalized']], 
    on='Time', 
    how='inner'
)
```

**What to Check:**
- ✅ Denormalized values are in correct scale
- ✅ Datasets align properly on dates
- ✅ No data loss during merge

#### 📊 **Section 3: Error Metrics Calculation**
```python
# Calculate comprehensive error metrics
comparison_df['Forecast_Error'] = comparison_df['Forecast_Balance_Denormalized'] - comparison_df['Actual_Test_Balance']
comparison_df['Absolute_Error'] = abs(comparison_df['Forecast_Error'])
comparison_df['Percentage_Error'] = (comparison_df['Forecast_Error'] / comparison_df['Actual_Test_Balance']) * 100

# Summary statistics
mae = comparison_df['Absolute_Error'].mean()
rmse = np.sqrt((comparison_df['Forecast_Error']**2).mean())
mape = comparison_df['Absolute_Percentage_Error'].mean()
correlation = comparison_df['Actual_Test_Balance'].corr(comparison_df['Forecast_Balance_Denormalized'])
```

**Key Metrics to Monitor:**
- **MAE (Mean Absolute Error)**: Average prediction error in original units
- **RMSE (Root Mean Square Error)**: Penalizes larger errors more heavily
- **MAPE (Mean Absolute Percentage Error)**: Error as percentage of actual values
- **Correlation**: How well predictions follow actual patterns (-1 to +1)
- **Directional Accuracy**: Percentage of correct trend predictions

#### 📈 **Section 4: Comprehensive Visualizations**
```python
# 4-panel comparison plot
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Time series comparison
axes[0, 0].plot(comparison_df['Time'], comparison_df['Actual_Test_Balance'], 
                label='Actual', marker='o')
axes[0, 0].plot(comparison_df['Time'], comparison_df['Forecast_Balance_Denormalized'], 
                label='Forecast', marker='s')

# Plot 2: Forecast errors
# Plot 3: Scatter plot (actual vs forecast)  
# Plot 4: Absolute percentage errors
```

**What to Check:**
- ✅ Time series plots show reasonable alignment
- ✅ Scatter plot points cluster around diagonal line
- ✅ Error distributions look normal
- ✅ No systematic biases visible

#### 💾 **Section 5: Results Saving**
```python
# Save detailed comparison
final_comparison.to_csv('test_vs_forecast_comparison.csv', index=False)

# Save summary report
summary_report = {
    'Mean_Absolute_Error': mae,
    'Root_Mean_Square_Error': rmse,
    'Mean_Absolute_Percentage_Error': mape,
    'Correlation_Coefficient': correlation,
    # ... more metrics
}

with open('comparison_summary_report.json', 'w') as f:
    json.dump(summary_report, f, indent=4)
```

### 📁 **Output Files:**
- `test_vs_forecast_comparison.csv` - Detailed comparison with all metrics
- `comparison_summary_report.json` - Summary statistics and key metrics

### ⏱️ **Expected Runtime:** 3-5 minutes

---

## 📊 Understanding Results

### Interpreting Key Metrics:

#### 🎯 **Mean Absolute Error (MAE)**
- **What it means**: Average prediction error in original balance units
- **Good values**: < 10% of average balance amount
- **Example**: MAE of 5,000 means predictions are off by $5,000 on average

#### 🎯 **Root Mean Square Error (RMSE)**  
- **What it means**: Penalizes large errors more heavily than MAE
- **Comparison**: RMSE should be close to MAE for consistent errors
- **Large RMSE vs MAE**: Indicates some very large prediction errors

#### 🎯 **Correlation Coefficient**
- **Range**: -1 to +1
- **Good values**: > 0.7 for strong positive correlation
- **What it means**: How well predictions follow actual patterns
- **Negative values**: Predictions move opposite to actual (concerning)

#### 🎯 **Directional Accuracy**
- **Range**: 0% to 100%
- **Good values**: > 60% for trend-following models
- **What it means**: Percentage of correct trend predictions (up/down)

### 📈 **Performance Categories:**

| Metric | Excellent | Good | Fair | Poor |
|--------|-----------|------|------|------|
| **Correlation** | > 0.9 | 0.7-0.9 | 0.5-0.7 | < 0.5 |
| **MAPE** | < 5% | 5-15% | 15-25% | > 25% |
| **Directional Accuracy** | > 80% | 60-80% | 50-60% | < 50% |

### 🔍 **Visual Analysis Guide:**

#### **Time Series Plot (Red=Actual, Blue=Forecast)**
- **Look for**: How closely blue line follows red line
- **Good signs**: Similar trends, magnitude, and timing
- **Warning signs**: Consistent over/under-prediction, phase shifts

#### **Scatter Plot (Actual vs Forecast)**
- **Look for**: Points clustering around diagonal line
- **Good signs**: Tight cluster around y=x line
- **Warning signs**: Systematic deviations, outliers

#### **Error Distribution**
- **Look for**: Normal distribution centered around zero
- **Good signs**: Symmetric, bell-shaped error distribution
- **Warning signs**: Skewed errors, multiple peaks

---

## 🚨 Troubleshooting

### Common Issues & Solutions:

#### **1. Data Loading Errors**
```
Error: FileNotFoundError: 'processed_train_dataset.xlsx' not found
```
**Solution**: Run `f_data_preprocessing.ipynb` first to generate required files.

#### **2. Memory Errors During Optimization**
```
Error: CUDA out of memory
```
**Solutions**:
- Reduce `batch_size` in optimization parameters
- Use CPU instead of GPU: Add `accelerator='cpu'` to model config
- Restart kernel to clear memory

#### **3. Poor Model Performance**
**Symptoms**: Very high RMSE, low correlation, poor visualizations
**Solutions**:
- Check data quality in preprocessing step
- Increase optimization time or trials
- Verify feature engineering is working correctly
- Check for data leakage or scaling issues

#### **4. Optimization Runs Forever**
**Symptoms**: Optuna trials don't improve for hours
**Solutions**:
- Check if database is locked by another process
- Reduce parameter search space
- Use fewer trials with better initial parameters

#### **5. Date Alignment Issues**
```
Error: No overlapping dates for comparison
```
**Solutions**:
- Verify test data covers forecast period
- Check date formats are consistent
- Ensure no timezone issues

### 📞 **Getting Help:**

1. **Check Logs**: Look for error messages in notebook outputs
2. **Verify Files**: Ensure all required input/output files exist
3. **Memory Usage**: Monitor system resources during long runs
4. **Parameter Validation**: Check if hyperparameters are reasonable
5. **Data Inspection**: Manually verify data at each step

---

## 🚀 Advanced Usage

### Customizing the Pipeline:

#### **1. Adjusting Forecast Horizon**
```python
# In f_forecasting.ipynb, change:
HORIZON = 60  # For 60-day forecast instead of 30
```

#### **2. Adding Custom Features**
```python
# In f_data_preprocessing.ipynb, add:
df['custom_feature'] = your_feature_calculation()

# Update feature categories:
feature_categories['future_features'].append('custom_feature')
```

#### **3. Modifying Optimization**
```python
# Extend optimization time:
TIMEOUT_HOURS = 24  # 24-hour optimization

# Custom parameter ranges:
def objective(trial):
    params = {
        'input_size': trial.suggest_int('input_size', 100, 300),  # Wider range
        # ... other parameters
    }
```

#### **4. Production Deployment**
```python
# Save model for reuse:
nf.save('nbeats_production_model/')

# Load for inference:
nf_loaded = NeuralForecast.load('nbeats_production_model/')
```

### 📊 **Monitoring in Production:**

1. **Performance Tracking**: Monitor prediction accuracy over time
2. **Data Drift Detection**: Check if new data differs from training data
3. **Model Retraining**: Retrain periodically with new data
4. **Alert Systems**: Set up alerts for poor performance periods

---

## 📝 Quick Reference

### ✅ **Pre-execution Checklist:**
- [ ] Required data files are in project directory
- [ ] All dependencies are installed
- [ ] Sufficient disk space (>2GB) for optimization database
- [ ] Memory available (8GB+ recommended)

### 📁 **File Dependencies:**
```
Notebook 1 (f_data_preprocessing.ipynb):
  Input:  shihara_train_30days_version2.xlsx, shihara_test_30days_version2.xlsx
  Output: processed_train_dataset.xlsx, processed_test_dataset.xlsx, scaling_parameters.json

Notebook 2 (f_forecasting.ipynb):
  Input:  processed_train_dataset.xlsx, scaling_parameters.json
  Output: forecast_final_with_actual.csv, best_hyperparameters.json, optuna database

Notebook 3 (compare.ipynb):
  Input:  processed_test_dataset.csv, forecast_final_with_actual.csv, scaling_parameters.json
  Output: test_vs_forecast_comparison.csv, comparison_summary_report.json
```

### ⏱️ **Expected Runtime Summary:**
- **Data Preprocessing**: 2-5 minutes
- **Model Training & Forecasting**: 11-13 hours  
- **Model Validation**: 3-5 minutes
- **Total Pipeline**: ~12-14 hours

### 🎯 **Success Indicators:**
- All notebooks run without errors
- Forecast files are generated with reasonable values
- Error metrics are within acceptable ranges
- Visualizations show good model performance
- Summary reports indicate successful validation

---

**📧 For questions or issues, refer to the main [README.md](README.md) or check the individual notebook documentation.**
