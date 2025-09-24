import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load dataset
data = pd.read_csv('modifications.csv')

# Convert date columns
data['Clean Date'] = pd.to_datetime(data['Clean Date'], format='%d-%m-%Y')
data['Time From'] = pd.to_datetime(data['Time From'], format='mixed', errors='coerce')
data['Time End'] = pd.to_datetime(data['Time End'], format='mixed', errors='coerce')

# Convert 'Time (in mins)' to numeric
data['Time (in mins)'] = pd.to_numeric(data['Time (in mins)'], errors='coerce')

hierarchy = ['Modified Production Line', 'Modified Equipment', 'Modified Part']

def calculate_failure_frequency_hierarchical(df, group_cols, date_col, days):
    result = []
    for keys, group in df.groupby(group_cols):
        group = group.sort_values(date_col)
        for current_date in group[date_col]:
            window_start = current_date - timedelta(days=days)
            failures = len(group[(group[date_col] >= window_start) & (group[date_col] <= current_date)])
            result.append(dict(zip(group_cols, keys), **{date_col: current_date, f'Failures_Last_{days}_Days': failures}))
    return pd.DataFrame(result)

for days in [30, 60, 90]:
    freq_df = calculate_failure_frequency_hierarchical(data, hierarchy, 'Clean Date', days)
    data = data.merge(freq_df, on=hierarchy + ['Clean Date'], how='left')

avg_downtime = data.groupby('Modified Equipment')['Time (in mins)'].mean().reset_index()
avg_downtime.rename(columns={'Time (in mins)': 'Avg_Downtime_Mins'}, inplace=True)
data = data.merge(avg_downtime, on='Modified Equipment', how='left')

part_failure = data.groupby(hierarchy)['Clean Date'].count().reset_index(name='Part_Failure_Count')
data = data.merge(part_failure, on=hierarchy, how='left')

data = data.sort_values(hierarchy + ['Clean Date'])
data['Days_Since_Last_Failure'] = data.groupby(hierarchy)['Clean Date'].diff().dt.days.fillna(0)

first_failure = data.groupby('Modified Equipment')['Clean Date'].min().reset_index(name='First_Failure_Date')
data = data.merge(first_failure, on='Modified Equipment', how='left')
data['Equipment_Age_Days'] = (data['Clean Date'] - data['First_Failure_Date']).dt.days

shift_failure = data.groupby(['Modified Equipment', 'Shift'])['Clean Date'].count().reset_index(name='Shift_Failure_Count')
data = data.merge(shift_failure, on=['Modified Equipment', 'Shift'], how='left')

line_failure = data.groupby(['Modified Production Line'])['Clean Date'].count().reset_index(name='Line_Failure_Count')
data = data.merge(line_failure, on=['Modified Production Line'], how='left')

data['Hour_of_Day'] = data['Time From'].dt.hour
data['Day_of_Week'] = data['Clean Date'].dt.dayofweek
data['Month'] = data['Clean Date'].dt.month
data['Quarter'] = data['Clean Date'].dt.quarter

def calculate_rolling_stats_hierarchical(df, group_cols, date_col, value_col, window_days):
    result = []
    for keys, group in df.groupby(group_cols):
        group = group.dropna(subset=[date_col, value_col])
        group = group.sort_values(date_col).drop_duplicates(subset=[date_col])

        if group.empty:
            continue

        group = group.set_index(date_col)
        rolling = group[value_col].rolling(f'{window_days}D', min_periods=1)
        rolling_mean = rolling.mean()
        rolling_std = rolling.std()

        temp = pd.DataFrame({
            date_col: rolling_mean.index,
            f'Rolling_Mean_{window_days}d': rolling_mean.values,
            f'Rolling_Std_{window_days}d': rolling_std.values
        })
        for col, key in zip(group_cols, keys):
            temp[col] = key

        result.append(temp)

    return pd.concat(result, ignore_index=True) if result else pd.DataFrame()

rolling_30 = calculate_rolling_stats_hierarchical(data, hierarchy, 'Clean Date', 'Time (in mins)', 30)
rolling_30 = rolling_30.drop_duplicates(subset=hierarchy + ['Clean Date'])
data = data.merge(rolling_30, on=hierarchy + ['Clean Date'], how='left')

rolling_90 = calculate_rolling_stats_hierarchical(data, hierarchy, 'Clean Date', 'Time (in mins)', 90)
rolling_90 = rolling_90.drop_duplicates(subset=hierarchy + ['Clean Date'])
data = data.merge(rolling_90, on=hierarchy + ['Clean Date'], how='left')

def calculate_health_score(row):
    score = 100
    score -= row['Failures_Last_30_Days'] * 10
    score -= row['Failures_Last_60_Days'] * 5
    score -= row['Failures_Last_90_Days'] * 3
    if pd.notna(row['Time (in mins)']):
        score -= min(row['Time (in mins)'] / 60, 20)
    score += min(row['Days_Since_Last_Failure'] / 2, 20)
    return max(0, min(100, score))

data['Equipment_Health_Score'] = data.apply(calculate_health_score, axis=1)

data['Maintenance_Count'] = data.groupby('Modified Equipment').cumcount() + 1

data['Is_Peak_Hour'] = ((data['Hour_of_Day'] >= 9) & (data['Hour_of_Day'] <= 17)).astype(int)
data['Is_Weekend'] = (data['Day_of_Week'] >= 5).astype(int)

equip_part_failures = data.groupby(hierarchy).agg({
    'Clean Date': 'count',
    'Time (in mins)': 'mean'
}).reset_index()
equip_part_failures.columns = hierarchy + ['Part_Failure_Frequency', 'Avg_Part_Downtime']
data = data.merge(equip_part_failures, on=hierarchy, how='left')

# ✅ Lag Feature: Previous Downtime (optional for predictive modeling)
data['Prev_Downtime'] = data.groupby(hierarchy)['Time (in mins)'].shift(1)

# ✅ EMA (Exponential Moving Average) of Downtime for smoother trends
data['EMA_Downtime'] = data.groupby(hierarchy)['Time (in mins)'].transform(lambda x: x.ewm(span=3, adjust=False).mean())

# Final save
data.to_csv('processed_maintenance_data_cleaned.csv', index=False)
