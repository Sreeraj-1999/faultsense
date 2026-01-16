import json
from datetime import datetime

# Load the JSON file
with open(r'C:\Users\User\Desktop\siemens\shap_plot1.json', 'r') as f:
    data = json.load(f)

# Get anomaly detection data
anomaly_data = data['anomaly_detection_overall']

timestamps = anomaly_data['timestamps']
anomaly_flags = anomaly_data['anomaly_flags']
reconstruction_errors = anomaly_data['reconstruction_errors']

# Find all anomalies
print("=" * 80)
print("🔍 ANOMALIES FOUND:")
print("=" * 80)

anomaly_count = 0
anomaly_dates = []

for i, (ts, is_anomaly, error) in enumerate(zip(timestamps, anomaly_flags, reconstruction_errors)):
    if is_anomaly:
        anomaly_count += 1
        anomaly_dates.append(ts)
        print(f"{anomaly_count}. Timestamp: {ts}, Error: {error}")

print("=" * 80)
print(f"Total Anomalies: {anomaly_count}")
print("=" * 80)

# Find date range to test
if anomaly_dates:
    # Parse dates (handle different formats)
    parsed_dates = []
    for date_str in anomaly_dates:
        try:
            # Try parsing as datetime
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            parsed_dates.append(dt)
        except:
            # If parsing fails, just use the string
            parsed_dates.append(date_str)
    
    if parsed_dates and isinstance(parsed_dates[0], datetime):
        earliest = min(parsed_dates)
        latest = max(parsed_dates)
        
        print("\n📅 RECOMMENDED DATE RANGE FOR SHAP:")
        print("=" * 80)
        print(f"Start Date: {earliest.strftime('%Y-%m-%d')}")
        print(f"End Date: {latest.strftime('%Y-%m-%d')}")
        print("=" * 80)
        
        print("\n📋 POSTMAN REQUEST:")
        print("=" * 80)
        print(f'''{{
    "start_date": "{earliest.strftime('%Y-%m-%d')}",
    "end_date": "{latest.strftime('%Y-%m-%d')}",
    "vessel_imo": "9665669",
    "equipment": "ME_CYL"
}}''')
        print("=" * 80)
    else:
        print("\n📅 First few anomaly dates:")
        for date in anomaly_dates[:5]:
            print(f"   - {date}")
else:
    print("\n⚠️  No anomalies found in the data!")