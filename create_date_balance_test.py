import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def create_date_balance_excel(input_file, output_file):
    """
    Create an Excel file with date over balance data from the input file.
    
    Args:
        input_file (str): Path to the input Excel file
        output_file (str): Path to the output Excel file
    """
    
    try:
        # Read the Excel file
        print(f"Reading file: {input_file}")
        df = pd.read_excel(input_file)
        
        print("Original data shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("\nFirst few rows:")
        print(df.head())
        
        # Check if there's a date column
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        balance_columns = [col for col in df.columns if 'balance' in col.lower() or 'amount' in col.lower()]
        
        print(f"\nDate columns found: {date_columns}")
        print(f"Balance columns found: {balance_columns}")
        
        # If no specific date/balance columns found, try to identify them
        if not date_columns:
            # Look for columns that might contain dates
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        pd.to_datetime(df[col].iloc[0])
                        date_columns.append(col)
                    except:
                        pass
        
        if not balance_columns:
            # Look for numeric columns that might be balance
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    balance_columns.append(col)
        
        print(f"Identified date columns: {date_columns}")
        print(f"Identified balance columns: {balance_columns}")
        
        # Use the first date and balance columns found
        date_col = date_columns[0] if date_columns else df.columns[0]
        balance_col = balance_columns[0] if balance_columns else df.columns[1]
        
        print(f"Using date column: {date_col}")
        print(f"Using balance column: {balance_col}")
        
        # Convert date column to datetime
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Remove rows with invalid dates
        df = df.dropna(subset=[date_col])
        
        # Sort by date
        df = df.sort_values(date_col)
        
        # Get the date range
        start_date = df[date_col].min()
        end_date = df[date_col].max()
        
        print(f"\nDate range: {start_date} to {end_date}")
        
        # Create a complete date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create a new dataframe with all dates
        date_balance_df = pd.DataFrame({'Date': date_range})
        
        # Merge with original data to get balances
        df_temp = df[[date_col, balance_col]].copy()
        df_temp.columns = ['Date', 'Balance']
        
        # Merge and forward fill missing values
        date_balance_df = date_balance_df.merge(df_temp, on='Date', how='left')
        date_balance_df['Balance'] = date_balance_df['Balance'].fillna(method='ffill')
        
        # If there are still NaN values at the beginning, backward fill them
        date_balance_df['Balance'] = date_balance_df['Balance'].fillna(method='bfill')
        
        # Calculate total balance (cumulative sum)
        date_balance_df['Total_Balance'] = date_balance_df['Balance'].cumsum()
        
        # Format the date column
        date_balance_df['Date'] = date_balance_df['Date'].dt.strftime('%Y-%m-%d')
        
        # Reorder columns
        date_balance_df = date_balance_df[['Date', 'Balance', 'Total_Balance']]
        
        print(f"\nFinal data shape: {date_balance_df.shape}")
        print("\nFirst few rows of processed data:")
        print(date_balance_df.head())
        print("\nLast few rows of processed data:")
        print(date_balance_df.tail())
        
        # Save to Excel
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            date_balance_df.to_excel(writer, sheet_name='Date_Balance_Data', index=False)
            
            # Create a summary sheet
            summary_data = {
                'Metric': ['Start Date', 'End Date', 'Total Days', 'Min Balance', 'Max Balance', 'Average Balance'],
                'Value': [
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d'),
                    len(date_balance_df),
                    date_balance_df['Balance'].min(),
                    date_balance_df['Balance'].max(),
                    date_balance_df['Balance'].mean()
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"\nExcel file created successfully: {output_file}")
        print(f"Total records: {len(date_balance_df)}")
        print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        return date_balance_df
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

if __name__ == "__main__":
    # File paths
    input_file = "37_days/shihara_test_37days_version2.xlsx"
    output_file = "date_over_balance_test.xlsx"
    
    # Create the Excel file
    result_df = create_date_balance_excel(input_file, output_file)
    
    if result_df is not None:
        print("\nProcessing completed successfully!")
        print(f"Output file: {output_file}")
    else:
        print("\nProcessing failed!") 