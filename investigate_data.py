import pandas as pd
import sys

file_path = r'D:\Zipsar-Base\anbl-model\data\Final datasets_Web develop.xlsx'

try:
    df = pd.read_excel(file_path)
    print("Column Names:")
    print(df.columns.tolist())
    
    last_col = df.columns[-1]
    print(f"\nLast Column Name: {last_col}")
    print("\nValues in the last column:")
    print(df[last_col].tolist())
    
    severe_indices = df.index[df[last_col] == 'Severe'].tolist()
    biosafe_indices = df.index[df[last_col] == 'Biosafe'].tolist()
    
    print(f"\nSevere case indices: {severe_indices[:5]} (Total: {len(severe_indices)})")
    print(f"Biosafe case indices: {biosafe_indices[:5]} (Total: {len(biosafe_indices)})")

except Exception as e:
    print(f"Error: {e}")
