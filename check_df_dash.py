import sys
sys.path.append('.')
import test4
df = test4.load_dashboard_data()
import pandas as pd
print(df[['Date', 'STD', 'STA']].head())
df['Start_dt'] = pd.to_datetime(df['Date'].astype(str) + " " + df['STD'].astype(str), errors='coerce')
df['End_dt'] = pd.to_datetime(df['Date'].astype(str) + " " + df['STA'].astype(str), errors='coerce')
print(df[['Start_dt', 'End_dt']].head())
print("NaT in Start_dt:", df['Start_dt'].isna().sum())
print("NaT in End_dt:", df['End_dt'].isna().sum())
