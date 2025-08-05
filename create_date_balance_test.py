import pandas as pd
from datetime import datetime, timedelta

# Read the test dataset
df = pd.read_excel('shihara_test_30days_version2.xlsx')

# Sort by date to ensure chronological order
df = df.sort_values('Date')

# Get the date range
start_date = df['Date'].min()
end_date = df['Date'].max()

# Create a complete date range
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# Create a dictionary to store the last known balance for each date
date_balance_dict = {}

# Initialize with the first balance
current_balance = df.iloc[0]['Balance']

# Fill the dictionary with balances
for date in date_range:
    # Check if this date exists in the original data
    day_data = df[df['Date'].dt.date == date.date()]
    
    if not day_data.empty:
        # If there are multiple transactions on the same day, use the last one
        current_balance = day_data.iloc[-1]['Balance']
    
    # Store the current balance for this date
    date_balance_dict[date] = current_balance

# Create the final dataframe
result_df = pd.DataFrame({
    'Date': list(date_balance_dict.keys()),
    'Balance': list(date_balance_dict.values())
})

# Format the date column
result_df['Date'] = result_df['Date'].dt.strftime('%Y-%m-%d')

# Save to Excel
output_filename = 'date_over_balance_test.xlsx'
result_df.to_excel(output_filename, index=False)

print(f"Created {output_filename} with {len(result_df)} days of balance data")
print(f"Date range: {result_df['Date'].iloc[0]} to {result_df['Date'].iloc[-1]}")
print("\nFirst 10 rows:")
print(result_df.head(10))
print("\nLast 10 rows:")
print(result_df.tail(10))

# Show some statistics
print(f"\nStatistics:")
print(f"Total days: {len(result_df)}")
print(f"Unique balance values: {result_df['Balance'].nunique()}")
print(f"Balance range: {result_df['Balance'].min():.2f} to {result_df['Balance'].max():.2f}") 