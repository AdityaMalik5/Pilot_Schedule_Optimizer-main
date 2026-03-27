import sys
sys.path.append('.')
import test4
import json
import pandas as pd

df_dash = test4.load_dashboard_data()
df_comp = df_dash.copy()
df_comp['Start_dt'] = pd.to_datetime(df_comp['Date'].astype(str) + " " + df_comp['STD'].astype(str), errors='coerce')
df_comp['End_dt'] = pd.to_datetime(df_comp['Date'].astype(str) + " " + df_comp['STA'].astype(str), errors='coerce')
df_comp = df_comp.sort_values(['Aircraft', 'Start_dt'])
df_comp['next_Start'] = df_comp.groupby('Aircraft')['Start_dt'].shift(-1)
df_comp['prev_End'] = df_comp.groupby('Aircraft')['End_dt'].shift(1)

flight_records = []
for _, row in df_comp.iterrows():
    dep_hr = row['Start_dt'].hour + row['Start_dt'].minute / 60.0 if pd.notna(row['Start_dt']) else 0
    arr_hr = row['End_dt'].hour + row['End_dt'].minute / 60.0 if pd.notna(row['End_dt']) else 0
    if arr_hr < dep_hr:
        arr_hr += 24.0
    
    date_str = row['Date'] if pd.notna(row['Date']) else "Unknown"
    if isinstance(date_str, pd.Timestamp) or hasattr(date_str, 'strftime'):
        date_str = date_str.strftime('%Y-%m-%d')
    else:
        date_str = str(date_str)
    
    status = "normal"
    issue = ""
    basic_status = str(row['Status'])
    
    is_overlap = False
    gap_violation = False
    
    if pd.notna(row['prev_End']) and row['Start_dt'] < row['prev_End']:
        is_overlap = True
    if pd.notna(row['next_Start']) and row['End_dt'] > row['next_Start']:
        is_overlap = True
        
    if pd.notna(row['prev_End']) and 0 <= (row['Start_dt'] - row['prev_End']).total_seconds() < 7200:
        gap_violation = True
    if pd.notna(row['next_Start']) and 0 <= (row['next_Start'] - row['End_dt']).total_seconds() < 7200:
        gap_violation = True

    if basic_status == "Conflict" or is_overlap:
        status = "conflict"
        issue = "Aircraft double-booked (Overlap)" if is_overlap else "Operational Conflict"
    elif basic_status == "Delayed" or gap_violation:
        status = "warning"
        issue = "Tight turnaround (< 2h)" if gap_violation else "Delayed"
    
    flight_records.append({
        "id": str(row['Flight Number']),
        "date": date_str,
        "aircraft": str(row['Aircraft']),
        "route": f"{row['From']}→{row['To']}",
        "dep": dep_hr,
        "arr": arr_hr,
        "cap": str(row['Captain']),
        "fo": str(row['First Officer']),
        "status": status,
        "issue": issue
    })

# Read test4.py to extract board_html
with open('test4.py', 'r') as f:
    content = f.read()

start_idx = content.find('board_html = """')
end_idx = content.find('"""', start_idx + 16)
board_html = content[start_idx+16:end_idx]

board_html = board_html.replace('__FLIGHT_RECORDS_JSON__', json.dumps(flight_records))

with open('debug_timeline.html', 'w') as f:
    f.write(board_html)

print("Generated debug_timeline.html with", len(flight_records), "records.")
