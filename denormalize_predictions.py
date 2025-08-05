# Denormalize GPU Comparison Predictions
# Create denormalized-prediction.xlsx file

import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("GPU Prediction Denormalization")
print("="*50)

# Load scaling parameters
print("Loading scaling parameters...")
with open('scaling_parameters.json', 'r') as f:
    scaling_params = json.load(f)

print(f"Scaling parameters: {scaling_params}")

# Function to denormalize values
def denormalize_balance(normalized_value, min_val, max_val):
    """Convert normalized value back to original scale"""
    return normalized_value * (max_val - min_val) + min_val

# Load CPU results (normalized values)
print("Loading CPU results...")
cpu_results = pd.read_csv('cpu_result.csv')
cpu_results['Date'] = pd.to_datetime(cpu_results['Date'])

print(f"CPU results shape: {cpu_results.shape}")
print(f"Date range: {cpu_results['Date'].min()} to {cpu_results['Date'].max()}")

# Display original normalized values
print("\nOriginal normalized values (first 5 rows):")
print(cpu_results[['Date', 'Predicted_Balance']].head())

# Denormalize all prediction columns
print("\nDenormalizing predictions...")

# Main prediction column
cpu_results['Predicted_Balance_Denormalized'] = denormalize_balance(
    cpu_results['Predicted_Balance'], 
    scaling_params['min_balance'], 
    scaling_params['max_balance']
)

# Confidence interval columns
confidence_columns = ['Lower_CI_95', 'Lower_CI_90', 'Lower_CI_80', 'Upper_CI_80', 'Upper_CI_90', 'Upper_CI_95']
for col in confidence_columns:
    if col in cpu_results.columns:
        cpu_results[f'{col}_Denormalized'] = denormalize_balance(
            cpu_results[col], 
            scaling_params['min_balance'], 
            scaling_params['max_balance']
        )

# Display denormalized values
print("\nDenormalized values (first 5 rows):")
print(cpu_results[['Date', 'Predicted_Balance', 'Predicted_Balance_Denormalized']].head())

# Create comprehensive denormalized dataset
denormalized_data = cpu_results.copy()

# Add additional useful columns
denormalized_data['Day_Number'] = range(1, len(denormalized_data) + 1)
denormalized_data['Analysis_Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Round numerical values for better readability
numerical_columns = ['Predicted_Balance_Denormalized'] + [f'{col}_Denormalized' for col in confidence_columns if col in cpu_results.columns]
for col in numerical_columns:
    denormalized_data[col] = denormalized_data[col].round(2)

# Also round the original normalized values
denormalized_data['Predicted_Balance'] = denormalized_data['Predicted_Balance'].round(6)
for col in confidence_columns:
    if col in denormalized_data.columns:
        denormalized_data[col] = denormalized_data[col].round(6)

# Calculate daily changes in denormalized values
denormalized_data['Daily_Change_Denormalized'] = denormalized_data['Predicted_Balance_Denormalized'].diff()
denormalized_data['Daily_Change_Denormalized'] = denormalized_data['Daily_Change_Denormalized'].round(2)

# Calculate cumulative change from first prediction
first_prediction = denormalized_data['Predicted_Balance_Denormalized'].iloc[0]
denormalized_data['Cumulative_Change_Denormalized'] = denormalized_data['Predicted_Balance_Denormalized'] - first_prediction
denormalized_data['Cumulative_Change_Denormalized'] = denormalized_data['Cumulative_Change_Denormalized'].round(2)

# Calculate percentage changes
denormalized_data['Daily_Change_Percent'] = (denormalized_data['Daily_Change_Denormalized'] / denormalized_data['Predicted_Balance_Denormalized'].shift(1) * 100).round(2)
denormalized_data['Cumulative_Change_Percent'] = (denormalized_data['Cumulative_Change_Denormalized'] / first_prediction * 100).round(2)

# Create the Excel file with multiple sheets
output_filename = 'denormalized-prediction.xlsx'

print(f"\nCreating Excel file: {output_filename}")

with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
    
    # Sheet 1: Main Denormalized Predictions
    main_columns = ['Date', 'Day_Number', 'Predicted_Balance_Denormalized', 
                   'Daily_Change_Denormalized', 'Cumulative_Change_Denormalized',
                   'Daily_Change_Percent', 'Cumulative_Change_Percent']
    
    main_sheet = denormalized_data[main_columns].copy()
    main_sheet['Date'] = main_sheet['Date'].dt.strftime('%Y-%m-%d')
    main_sheet.to_excel(writer, sheet_name='Denormalized_Predictions', index=False)
    
    # Sheet 2: Confidence Intervals
    ci_columns = ['Date', 'Day_Number', 'Predicted_Balance_Denormalized']
    ci_columns.extend([f'{col}_Denormalized' for col in confidence_columns if col in cpu_results.columns])
    
    ci_sheet = denormalized_data[ci_columns].copy()
    ci_sheet['Date'] = ci_sheet['Date'].dt.strftime('%Y-%m-%d')
    ci_sheet.to_excel(writer, sheet_name='Confidence_Intervals', index=False)
    
    # Sheet 3: Comparison (Normalized vs Denormalized)
    comparison_columns = ['Date', 'Day_Number', 'Predicted_Balance', 'Predicted_Balance_Denormalized']
    comparison_columns.extend([col for col in confidence_columns if col in cpu_results.columns])
    comparison_columns.extend([f'{col}_Denormalized' for col in confidence_columns if col in cpu_results.columns])
    
    comparison_sheet = denormalized_data[comparison_columns].copy()
    comparison_sheet['Date'] = comparison_sheet['Date'].dt.strftime('%Y-%m-%d')
    comparison_sheet.to_excel(writer, sheet_name='Normalized_vs_Denormalized', index=False)
    
    # Sheet 4: Summary Statistics
    summary_data = {
        'Metric': [
            'Analysis_Date',
            'Total_Predictions',
            'Date_Range_Start',
            'Date_Range_End',
            'Min_Predicted_Balance',
            'Max_Predicted_Balance',
            'Mean_Predicted_Balance',
            'Std_Predicted_Balance',
            'Total_Change_Amount',
            'Total_Change_Percent',
            'Min_Daily_Change',
            'Max_Daily_Change',
            'Mean_Daily_Change',
            'Scaling_Min_Value',
            'Scaling_Max_Value',
            'Scaling_Range'
        ],
        'Value': [
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            len(denormalized_data),
            denormalized_data['Date'].min().strftime('%Y-%m-%d'),
            denormalized_data['Date'].max().strftime('%Y-%m-%d'),
            round(denormalized_data['Predicted_Balance_Denormalized'].min(), 2),
            round(denormalized_data['Predicted_Balance_Denormalized'].max(), 2),
            round(denormalized_data['Predicted_Balance_Denormalized'].mean(), 2),
            round(denormalized_data['Predicted_Balance_Denormalized'].std(), 2),
            round(denormalized_data['Cumulative_Change_Denormalized'].iloc[-1], 2),
            round(denormalized_data['Cumulative_Change_Percent'].iloc[-1], 2),
            round(denormalized_data['Daily_Change_Denormalized'].min(), 2),
            round(denormalized_data['Daily_Change_Denormalized'].max(), 2),
            round(denormalized_data['Daily_Change_Denormalized'].mean(), 2),
            scaling_params['min_balance'],
            scaling_params['max_balance'],
            scaling_params['range']
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
    
    # Sheet 5: Change Analysis
    change_columns = ['Date', 'Day_Number', 'Predicted_Balance_Denormalized', 
                     'Daily_Change_Denormalized', 'Daily_Change_Percent',
                     'Cumulative_Change_Denormalized', 'Cumulative_Change_Percent']
    
    change_sheet = denormalized_data[change_columns].copy()
    change_sheet['Date'] = change_sheet['Date'].dt.strftime('%Y-%m-%d')
    change_sheet.to_excel(writer, sheet_name='Change_Analysis', index=False)

print(f"✓ Excel file created successfully: {output_filename}")
print(f"\nFile contains the following sheets:")
print(f"  1. Denormalized_Predictions - Main prediction values")
print(f"  2. Confidence_Intervals - All confidence interval values")
print(f"  3. Normalized_vs_Denormalized - Comparison of normalized and denormalized values")
print(f"  4. Summary_Statistics - Key statistics and metadata")
print(f"  5. Change_Analysis - Daily and cumulative changes")

# Display summary of the denormalized data
print(f"\n" + "="*50)
print("DENORMALIZED PREDICTION SUMMARY")
print("="*50)

print(f"Date Range: {denormalized_data['Date'].min().strftime('%Y-%m-%d')} to {denormalized_data['Date'].max().strftime('%Y-%m-%d')}")
print(f"Number of Predictions: {len(denormalized_data)}")
print(f"Predicted Balance Range: {denormalized_data['Predicted_Balance_Denormalized'].min():.2f} to {denormalized_data['Predicted_Balance_Denormalized'].max():.2f}")
print(f"Mean Predicted Balance: {denormalized_data['Predicted_Balance_Denormalized'].mean():.2f}")
print(f"Total Change: {denormalized_data['Cumulative_Change_Denormalized'].iloc[-1]:.2f} ({denormalized_data['Cumulative_Change_Percent'].iloc[-1]:.2f}%)")

print(f"\nFirst 5 denormalized predictions:")
print(denormalized_data[['Date', 'Predicted_Balance_Denormalized', 'Daily_Change_Denormalized']].head().to_string(index=False))

print(f"\n✓ Denormalization complete! File saved as: {output_filename}") 