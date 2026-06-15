import pandas as pd
try:
    df = pd.read_excel('data/Final datasets_Web develop.xlsx')
    print("COLUMNS:")
    print(df.columns.tolist())
    print("\nFIRST 5 ROWS OF TARGET:")
    target_col = df.columns[-1]
    print(f"Target Column Name: {target_col}")
    print(df[target_col].head())
    print("\nUNIQUE VALUES IN TARGET:")
    print(df[target_col].unique()[:20])
except Exception as e:
    print(f"ERROR: {e}")
