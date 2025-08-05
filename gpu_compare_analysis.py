# GPU Compare Analysis: Denormalize CPU Results and Compare with Test Dataset
# Based on compare.ipynb structure

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

print("GPU Compare Analysis - Denormalizing CPU Results")
print("="*60)

# Load the datasets
print("Loading datasets...")

# Load test dataset
test_data = pd.read_csv('processed_test_dataset.csv')
test_data['Date'] = pd.to_datetime(test_data['Date'])

# Load CPU results (normalized values)
cpu_results = pd.read_csv('cpu_result.csv')
cpu_results['Date'] = pd.to_datetime(cpu_results['Date'])

# Load scaling parameters
with open('scaling_parameters.json', 'r') as f:
    scaling_params = json.load(f)

print("Dataset shapes:")
print(f"Test data: {test_data.shape}")
print(f"CPU results: {cpu_results.shape}")
print(f"\nScaling parameters: {scaling_params}")

# Display first few rows of each dataset
print("\n" + "="*50)
print("TEST DATASET (First 5 rows):")
print(test_data.head())

print("\n" + "="*50)
print("CPU RESULTS (First 5 rows):")
print(cpu_results.head())

# Function to denormalize values
def denormalize_balance(normalized_value, min_val, max_val):
    """Convert normalized value back to original scale"""
    return normalized_value * (max_val - min_val) + min_val

# Denormalize CPU forecast values
cpu_results['Predicted_Balance_Denormalized'] = denormalize_balance(
    cpu_results['Predicted_Balance'], 
    scaling_params['min_balance'], 
    scaling_params['max_balance']
)

# Also denormalize confidence intervals
confidence_columns = ['Lower_CI_95', 'Lower_CI_90', 'Lower_CI_80', 'Upper_CI_80', 'Upper_CI_90', 'Upper_CI_95']
for col in confidence_columns:
    if col in cpu_results.columns:
        cpu_results[f'{col}_Denormalized'] = denormalize_balance(
            cpu_results[col], 
            scaling_params['min_balance'], 
            scaling_params['max_balance']
        )

print("\nDenormalized CPU forecast values:")
print("Original Predicted (normalized):", cpu_results['Predicted_Balance'].head().values)
print("Denormalized Predicted:", cpu_results['Predicted_Balance_Denormalized'].head().values)

# Check date ranges
print(f"\nDate ranges:")
print(f"Test data: {test_data['Date'].min()} to {test_data['Date'].max()}")
print(f"CPU results: {cpu_results['Date'].min()} to {cpu_results['Date'].max()}")

# Find overlapping dates
cpu_dates = set(cpu_results['Date'].dt.date)
test_dates = set(test_data['Date'].dt.date)
overlapping_dates = cpu_dates.intersection(test_dates)

print(f"\nOverlapping dates: {len(overlapping_dates)}")
print(f"Overlapping date range: {min(overlapping_dates)} to {max(overlapping_dates)}")

# Merge datasets for comparison
# Rename columns for clarity
test_data_renamed = test_data.rename(columns={'Date': 'Date', 'Balance': 'Actual_Test_Balance'})
cpu_results_renamed = cpu_results.rename(columns={'Date': 'Date'})

# Merge on date
comparison_df = pd.merge(
    test_data_renamed[['Date', 'Actual_Test_Balance']], 
    cpu_results_renamed[['Date', 'Predicted_Balance_Denormalized', 'Predicted_Balance'] + 
                       [f'{col}_Denormalized' for col in confidence_columns if col in cpu_results.columns]], 
    on='Date', 
    how='inner'
)

print(f"\nMerged dataset shape: {comparison_df.shape}")
print("\nComparison dataset:")
print(comparison_df.head(10))

# Calculate differences and errors
comparison_df['Forecast_Error'] = comparison_df['Predicted_Balance_Denormalized'] - comparison_df['Actual_Test_Balance']
comparison_df['Absolute_Error'] = abs(comparison_df['Forecast_Error'])
comparison_df['Percentage_Error'] = (comparison_df['Forecast_Error'] / comparison_df['Actual_Test_Balance']) * 100
comparison_df['Absolute_Percentage_Error'] = abs(comparison_df['Percentage_Error'])

print(f"\nError Statistics:")
print(f"Mean Absolute Error (MAE): {comparison_df['Absolute_Error'].mean():.2f}")
print(f"Root Mean Square Error (RMSE): {np.sqrt((comparison_df['Forecast_Error']**2).mean()):.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {comparison_df['Absolute_Percentage_Error'].mean():.2f}%")
print(f"Mean Forecast Error (Bias): {comparison_df['Forecast_Error'].mean():.2f}")

# Additional Analysis and Summary
print("\n" + "="*60)
print("COMPREHENSIVE COMPARISON ANALYSIS")
print("="*60)

# Summary statistics
print("\n1. DATASET SUMMARY:")
print(f"   - Number of overlapping dates: {len(comparison_df)}")
print(f"   - Date range: {comparison_df['Date'].min().strftime('%Y-%m-%d')} to {comparison_df['Date'].max().strftime('%Y-%m-%d')}")

print("\n2. ACTUAL TEST BALANCE STATISTICS:")
print(f"   - Mean: {comparison_df['Actual_Test_Balance'].mean():.2f}")
print(f"   - Std Dev: {comparison_df['Actual_Test_Balance'].std():.2f}")
print(f"   - Min: {comparison_df['Actual_Test_Balance'].min():.2f}")
print(f"   - Max: {comparison_df['Actual_Test_Balance'].max():.2f}")

print("\n3. CPU PREDICTED BALANCE STATISTICS:")
print(f"   - Mean: {comparison_df['Predicted_Balance_Denormalized'].mean():.2f}")
print(f"   - Std Dev: {comparison_df['Predicted_Balance_Denormalized'].std():.2f}")
print(f"   - Min: {comparison_df['Predicted_Balance_Denormalized'].min():.2f}")
print(f"   - Max: {comparison_df['Predicted_Balance_Denormalized'].max():.2f}")

print("\n4. ERROR METRICS:")
print(f"   - Mean Absolute Error (MAE): {comparison_df['Absolute_Error'].mean():.2f}")
print(f"   - Root Mean Square Error (RMSE): {np.sqrt((comparison_df['Forecast_Error']**2).mean()):.2f}")
print(f"   - Mean Absolute Percentage Error (MAPE): {comparison_df['Absolute_Percentage_Error'].mean():.2f}%")
print(f"   - Mean Forecast Error (Bias): {comparison_df['Forecast_Error'].mean():.2f}")
print(f"   - Max Absolute Error: {comparison_df['Absolute_Error'].max():.2f}")
print(f"   - Min Absolute Error: {comparison_df['Absolute_Error'].min():.2f}")

# Correlation analysis
correlation = comparison_df['Actual_Test_Balance'].corr(comparison_df['Predicted_Balance_Denormalized'])
print(f"\n5. CORRELATION ANALYSIS:")
print(f"   - Pearson correlation coefficient: {correlation:.4f}")

# Performance categorization
high_error_threshold = comparison_df['Absolute_Percentage_Error'].quantile(0.75)
high_error_count = (comparison_df['Absolute_Percentage_Error'] > high_error_threshold).sum()

print(f"\n6. PERFORMANCE CATEGORIZATION:")
print(f"   - High error threshold (75th percentile): {high_error_threshold:.2f}%")
print(f"   - Number of high-error predictions: {high_error_count}")
print(f"   - Percentage of high-error predictions: {(high_error_count/len(comparison_df)*100):.1f}%")

# Direction accuracy
correct_direction = 0
for i in range(1, len(comparison_df)):
    actual_change = comparison_df.iloc[i]['Actual_Test_Balance'] - comparison_df.iloc[i-1]['Actual_Test_Balance']
    forecast_change = comparison_df.iloc[i]['Predicted_Balance_Denormalized'] - comparison_df.iloc[i-1]['Predicted_Balance_Denormalized']
    if (actual_change >= 0 and forecast_change >= 0) or (actual_change < 0 and forecast_change < 0):
        correct_direction += 1

direction_accuracy = (correct_direction / (len(comparison_df) - 1)) * 100 if len(comparison_df) > 1 else 0

print(f"\n7. DIRECTIONAL ACCURACY:")
print(f"   - Correct direction predictions: {correct_direction}/{len(comparison_df)-1}")
print(f"   - Directional accuracy: {direction_accuracy:.1f}%")

# Create comprehensive visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Actual vs Forecast Time Series
axes[0, 0].plot(comparison_df['Date'], comparison_df['Actual_Test_Balance'], 
                label='Actual Test Balance', marker='o', linewidth=2, markersize=6)
axes[0, 0].plot(comparison_df['Date'], comparison_df['Predicted_Balance_Denormalized'], 
                label='CPU Forecast Balance', marker='s', linewidth=2, markersize=6)
axes[0, 0].set_title('Actual vs CPU Forecast Balance Over Time', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Balance Amount')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].tick_params(axis='x', rotation=45)

# Plot 2: Forecast Errors
axes[0, 1].bar(range(len(comparison_df)), comparison_df['Forecast_Error'], 
               color=['red' if x < 0 else 'green' for x in comparison_df['Forecast_Error']])
axes[0, 1].set_title('CPU Forecast Errors (Forecast - Actual)', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Time Period')
axes[0, 1].set_ylabel('Error Amount')
axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Scatter plot - Actual vs Forecast
axes[1, 0].scatter(comparison_df['Actual_Test_Balance'], comparison_df['Predicted_Balance_Denormalized'], 
                   alpha=0.7, s=60, color='blue')
min_val = min(comparison_df['Actual_Test_Balance'].min(), comparison_df['Predicted_Balance_Denormalized'].min())
max_val = max(comparison_df['Actual_Test_Balance'].max(), comparison_df['Predicted_Balance_Denormalized'].max())
axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
axes[1, 0].set_title('Actual vs CPU Forecast Scatter Plot', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Actual Balance')
axes[1, 0].set_ylabel('CPU Forecast Balance')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Absolute Percentage Error
axes[1, 1].bar(range(len(comparison_df)), comparison_df['Absolute_Percentage_Error'], 
               color='orange', alpha=0.7)
axes[1, 1].set_title('CPU Absolute Percentage Error by Time Period', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Time Period')
axes[1, 1].set_ylabel('Absolute Percentage Error (%)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Create clean diagram: Date vs Balance (Red=Actual, Blue=Forecast)
plt.figure(figsize=(14, 8))

# Plot actual values in red
plt.plot(comparison_df['Date'], comparison_df['Actual_Test_Balance'], 
         color='red', linewidth=2.5, marker='o', markersize=6, 
         label='Actual Balance', alpha=0.8)

# Plot forecast values in blue
plt.plot(comparison_df['Date'], comparison_df['Predicted_Balance_Denormalized'], 
         color='blue', linewidth=2.5, marker='s', markersize=6, 
         label='CPU Forecast Balance', alpha=0.8)

# Customize the plot
plt.title('Actual vs CPU Forecast Balance Over Time', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Date', fontsize=14, fontweight='bold')
plt.ylabel('Balance', fontsize=14, fontweight='bold')

# Format the legend
plt.legend(fontsize=12, loc='upper right', frameon=True, fancybox=True, shadow=True)

# Format the axes
plt.xticks(rotation=45, fontsize=11)
plt.yticks(fontsize=11)

# Add grid for better readability
plt.grid(True, alpha=0.3, linestyle='--')

# Format y-axis to show values in thousands/millions
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K' if x < 1000000 else f'{x/1000000:.1f}M'))

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()

# Print summary statistics for the plot
print(f"Date Range: {comparison_df['Date'].min().strftime('%Y-%m-%d')} to {comparison_df['Date'].max().strftime('%Y-%m-%d')}")
print(f"Actual Balance Range: {comparison_df['Actual_Test_Balance'].min():.2f} to {comparison_df['Actual_Test_Balance'].max():.2f}")
print(f"CPU Forecast Balance Range: {comparison_df['Predicted_Balance_Denormalized'].min():.2f} to {comparison_df['Predicted_Balance_Denormalized'].max():.2f}")

# Save comparison results
print("\n" + "="*60)
print("SAVING COMPARISON RESULTS")
print("="*60)

# Create a comprehensive comparison dataframe with all metrics
final_comparison = comparison_df.copy()
final_comparison['Date'] = final_comparison['Date'].dt.strftime('%Y-%m-%d')

# Round numerical values for better readability
numerical_columns = ['Actual_Test_Balance', 'Predicted_Balance_Denormalized', 'Predicted_Balance',
                    'Forecast_Error', 'Absolute_Error', 'Percentage_Error', 'Absolute_Percentage_Error']
for col in numerical_columns:
    if col in final_comparison.columns:
        final_comparison[col] = final_comparison[col].round(2)

# Also round confidence interval columns
ci_columns = [col for col in final_comparison.columns if 'Denormalized' in col and col != 'Predicted_Balance_Denormalized']
for col in ci_columns:
    final_comparison[col] = final_comparison[col].round(2)

# Save to Excel with multiple sheets
output_filename = 'gpu_compare.xlsx'

with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
    # Main comparison data
    final_comparison.to_excel(writer, sheet_name='Comparison_Data', index=False)
    
    # Summary statistics
    summary_data = {
        'Metric': [
            'Analysis_Date',
            'Number_of_Data_Points',
            'Date_Range_Start',
            'Date_Range_End',
            'Mean_Absolute_Error',
            'Root_Mean_Square_Error',
            'Mean_Absolute_Percentage_Error',
            'Mean_Forecast_Bias',
            'Correlation_Coefficient',
            'Directional_Accuracy_Percent',
            'Actual_Balance_Mean',
            'Actual_Balance_Std',
            'CPU_Forecast_Balance_Mean',
            'CPU_Forecast_Balance_Std'
        ],
        'Value': [
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            len(comparison_df),
            comparison_df['Date'].min().strftime('%Y-%m-%d'),
            comparison_df['Date'].max().strftime('%Y-%m-%d'),
            round(comparison_df['Absolute_Error'].mean(), 2),
            round(np.sqrt((comparison_df['Forecast_Error']**2).mean()), 2),
            round(comparison_df['Absolute_Percentage_Error'].mean(), 2),
            round(comparison_df['Forecast_Error'].mean(), 2),
            round(correlation, 4),
            round(direction_accuracy, 1),
            round(comparison_df['Actual_Test_Balance'].mean(), 2),
            round(comparison_df['Actual_Test_Balance'].std(), 2),
            round(comparison_df['Predicted_Balance_Denormalized'].mean(), 2),
            round(comparison_df['Predicted_Balance_Denormalized'].std(), 2)
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
    
    # Error analysis
    error_analysis = comparison_df[['Date', 'Actual_Test_Balance', 'Predicted_Balance_Denormalized', 
                                   'Forecast_Error', 'Absolute_Error', 'Percentage_Error', 'Absolute_Percentage_Error']].copy()
    error_analysis['Date'] = error_analysis['Date'].dt.strftime('%Y-%m-%d')
    error_analysis.to_excel(writer, sheet_name='Error_Analysis', index=False)

print(f"✓ Detailed comparison saved to: {output_filename}")

# Display the saved comparison data
print(f"\n✓ COMPARISON COMPLETE!")
print(f"Excel file created: {output_filename}")
print(f"Contains sheets:")
print(f"  1. Comparison_Data - Full comparison with all metrics")
print(f"  2. Summary_Statistics - Key performance metrics")
print(f"  3. Error_Analysis - Detailed error breakdown")

print(f"\nFinal Comparison Table (First 10 rows):")
print(final_comparison.head(10).to_string(index=False))

print(f"\nKey Findings:")
print(f"  • The CPU forecast has a correlation of {correlation:.3f} with actual values")
print(f"  • Average absolute error is {comparison_df['Absolute_Error'].mean():.2f}")
print(f"  • Average percentage error is {comparison_df['Absolute_Percentage_Error'].mean():.1f}%")
print(f"  • Directional accuracy is {direction_accuracy:.1f}%") 