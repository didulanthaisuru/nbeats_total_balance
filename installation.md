# NBEATS Balance Prediction - Installation Guide

## 📋 Overview

This guide provides complete installation instructions for the NBEATS Balance Prediction system, including all required dependencies, environment setup, and verification steps.

## 🎯 System Requirements

- **Python**: 3.8 or higher (recommended: 3.9-3.11)
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 8GB RAM (16GB recommended for optimization)
- **Storage**: At least 2GB free space
- **GPU**: Optional but recommended for faster training

## 🔧 Installation Methods

### Method 1: pip install (Recommended)

```bash
# Core packages for NBEATS forecasting
pip install neuralforecast

# Data manipulation and analysis
pip install pandas numpy

# Visualization libraries
pip install matplotlib seaborn

# Machine learning utilities
pip install scikit-learn scipy

# Hyperparameter optimization
pip install optuna

# Excel file support
pip install openpyxl xlrd

# Additional utilities
pip install pathlib logging warnings
```

### Method 2: Install all at once

```bash
pip install neuralforecast pandas numpy matplotlib seaborn scikit-learn scipy optuna openpyxl xlrd
```

### Method 3: Using requirements.txt

Create a `requirements.txt` file with the following content:

```text
# NBEATS Balance Prediction Requirements
# Core forecasting framework
neuralforecast>=1.6.0

# Data processing
pandas>=1.5.0
numpy>=1.21.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# Machine learning
scikit-learn>=1.1.0
scipy>=1.9.0

# Hyperparameter optimization
optuna>=3.0.0

# File handling
openpyxl>=3.0.0
xlrd>=2.0.0

# Deep learning (auto-installed with neuralforecast)
torch>=1.12.0
pytorch-lightning>=1.8.0

# Additional utilities
pathlib2>=2.3.0
```

Then install:
```bash
pip install -r requirements.txt
```

## 🐍 Conda Installation (Alternative)

```bash
# Create conda environment
conda create -n nbeats_env python=3.9

# Activate environment
conda activate nbeats_env

# Install packages
conda install pandas numpy matplotlib seaborn scikit-learn scipy
conda install -c conda-forge openpyxl optuna

# Install neuralforecast via pip (not available in conda)
pip install neuralforecast
```

## 🚀 Quick Setup Verification

Create a test script `test_installation.py`:

```python
#!/usr/bin/env python3
"""
Installation verification script for NBEATS Balance Prediction
"""

def test_imports():
    """Test all required imports"""
    
    print("🔍 Testing package imports...")
    
    try:
        # Core packages
        import pandas as pd
        import numpy as np
        print("✅ pandas, numpy - OK")
        
        # Visualization
        import matplotlib.pyplot as plt
        import seaborn as sns
        print("✅ matplotlib, seaborn - OK")
        
        # Machine learning
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        import scipy
        print("✅ scikit-learn, scipy - OK")
        
        # Forecasting (main package)
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NBEATSx
        from neuralforecast.losses.pytorch import DistributionLoss
        print("✅ neuralforecast, NBEATSx - OK")
        
        # Hyperparameter optimization
        import optuna
        from optuna.storages import RDBStorage
        from optuna.trial import Trial
        print("✅ optuna - OK")
        
        # File handling
        import sqlite3
        import openpyxl
        print("✅ sqlite3, openpyxl - OK")
        
        # Standard library
        import os, time, warnings, logging
        from pathlib import Path
        from datetime import datetime, timedelta
        print("✅ Standard library modules - OK")
        
        print("\n🎉 All packages installed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install missing packages using pip or conda")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test pandas
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        print("✅ Pandas DataFrame creation - OK")
        
        # Test numpy
        arr = np.array([1, 2, 3])
        print("✅ NumPy array creation - OK")
        
        # Test NBEATSx model creation
        from neuralforecast.models import NBEATSx
        model = NBEATSx(h=7, input_size=14)
        print("✅ NBEATSx model creation - OK")
        
        # Test optuna
        import optuna
        study = optuna.create_study()
        print("✅ Optuna study creation - OK")
        
        print("\n🎉 Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def show_versions():
    """Show installed package versions"""
    
    print("\n📦 Package versions:")
    
    packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'sklearn', 'scipy', 'optuna', 'openpyxl'
    ]
    
    for package in packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'Unknown')
            print(f"   {package}: {version}")
        except ImportError:
            print(f"   {package}: Not installed")
    
    # Special case for neuralforecast
    try:
        import neuralforecast
        print(f"   neuralforecast: {neuralforecast.__version__}")
    except ImportError:
        print("   neuralforecast: Not installed")
    except AttributeError:
        print("   neuralforecast: Installed (version unknown)")

if __name__ == "__main__":
    print("🚀 NBEATS Balance Prediction - Installation Verification")
    print("="*60)
    
    # Run tests
    imports_ok = test_imports()
    
    if imports_ok:
        functionality_ok = test_basic_functionality()
        show_versions()
        
        if imports_ok and functionality_ok:
            print("\n" + "="*60)
            print("🎊 SUCCESS! Your environment is ready for NBEATS forecasting!")
            print("You can now run the main pipeline: python final_nbeats.py")
        else:
            print("\n❌ Some tests failed. Please check error messages above.")
    else:
        print("\n❌ Import tests failed. Please install missing packages.")
    
    print("="*60)
```

Run the verification:
```bash
python test_installation.py
```

## 📁 Required Data Files

Ensure your project has the following structure:

```
n_beats_total_balance_prediction/
├── final_nbeats.py                    # Main script
├── installation.md                    # This file
├── test_installation.py               # Verification script
├── requirements.txt                   # Package requirements
├── nbeatx_final_functions/            # Data directory
│   └── featured_shihara.xlsx          # Your data file
├── visualizations/                    # Output plots (created automatically)
└── optuna_balance_study.db           # SQLite database (created automatically)
```

## 🔧 Troubleshooting

### Common Issues

#### 1. PyTorch Installation Issues
```bash
# If neuralforecast fails to install PyTorch automatically
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# For GPU support:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. Optuna SQLite Issues
```bash
# If SQLite timeout errors occur
pip install --upgrade optuna
```

#### 3. Excel File Reading Issues
```bash
# Install additional Excel support
pip install xlsxwriter xlwt
```

#### 4. Matplotlib Backend Issues (Linux/macOS)
```bash
# If matplotlib display issues occur
export MPLBACKEND=Agg
```

### Environment Issues

#### Virtual Environment Setup
```bash
# Create virtual environment
python -m venv nbeats_env

# Activate (Windows)
nbeats_env\Scripts\activate

# Activate (Linux/macOS)
source nbeats_env/bin/activate

# Install packages
pip install -r requirements.txt
```

#### Jupyter Notebook Support
```bash
# If using Jupyter notebooks
pip install jupyter ipykernel
python -m ipykernel install --user --name=nbeats_env
```

## 🚀 Quick Start

After successful installation:

1. **Verify installation**:
   ```bash
   python test_installation.py
   ```

2. **Prepare your data**:
   - Place your Excel file in `nbeatx_final_functions/featured_shihara.xlsx`
   - Ensure it has columns: `Date`, `Normalized_Balance`, and other features

3. **Run the pipeline**:
   ```bash
   python final_nbeats.py
   ```

4. **Check outputs**:
   - Visualizations in `./visualizations/`
   - Hyperparameter results in `./optuna_balance_study.db`
   - Console output for metrics

## 📊 Expected Output

After successful execution, you should see:

- **4 visualization files**:
  - `actual_vs_predicted.png` - Time series comparison
  - `residual_analysis.png` - 4-plot residual diagnostics  
  - `scatter_actual_vs_predicted.png` - Scatter with R²
  - `performance_summary.png` - Complete metrics report

- **SQLite database**: `optuna_balance_study.db` with optimization history

- **Console metrics**: RMSE, MAE, MAPE, R², correlation

## 🔄 Updating Packages

To update all packages to latest versions:

```bash
pip install --upgrade neuralforecast pandas numpy matplotlib seaborn scikit-learn scipy optuna openpyxl
```

## 💡 Performance Tips

1. **For faster training**:
   ```bash
   # Install CUDA version of PyTorch if you have GPU
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **For memory optimization**:
   - Reduce `n_trials` in hyperparameter optimization
   - Use smaller `batch_size` in model configuration

3. **For reproducible results**:
   - Set `random_seed=42` in model configuration
   - Use fixed `n_trials` for consistent optimization

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify all packages are correctly installed
3. Ensure your data file format matches requirements
4. Run the verification script to identify missing dependencies

## 🏁 Final Check

Your installation is complete when:
- ✅ All imports work without errors
- ✅ NBEATSx model can be created
- ✅ Optuna study can be initialized
- ✅ Data files are in correct location
- ✅ Output directories can be created

**You're now ready to run the NBEATS Balance Prediction system!** 🎉
