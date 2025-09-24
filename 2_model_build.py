# import pandas as pd
# import numpy as np
# from sklearn.model_selection import GroupShuffleSplit
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import joblib

# # 1️⃣ Load processed dataset
# data = pd.read_csv("processed_maintenance_data_cleaned.csv")

# # 2️⃣ Generate Target Variable: Days Until Next Failure
# data = data.sort_values(['Modified Production Line', 'Modified Equipment', 'Modified Part', 'Clean Date'])
# data['Clean Date'] = pd.to_datetime(data['Clean Date'], errors='coerce')

# data['Days_Until_Next_Failure'] = data.groupby(
#     ['Modified Production Line', 'Modified Equipment', 'Modified Part']
# )['Clean Date'].shift(-1) - data['Clean Date']
# data['Days_Until_Next_Failure'] = data['Days_Until_Next_Failure'].dt.days

# # Drop rows where target is missing
# data = data.dropna(subset=['Days_Until_Next_Failure'])

# # 3️⃣ Feature Selection
# features = [
#     'Failures_Last_30_Days', 'Failures_Last_60_Days', 'Failures_Last_90_Days',
#     'Avg_Downtime_Mins', 'Days_Since_Last_Failure', 'Equipment_Age_Days',
#     'Shift_Failure_Count', 'Line_Failure_Count', 'Hour_of_Day', 'Day_of_Week',
#     'Month', 'Quarter', 'Rolling_Mean_30d', 'Rolling_Std_30d',
#     'Rolling_Mean_90d', 'Rolling_Std_90d', 'Equipment_Health_Score',
#     'Maintenance_Count', 'Is_Peak_Hour', 'Is_Weekend',
#     'Part_Failure_Frequency', 'Avg_Part_Downtime',
#     'Prev_Downtime', 'EMA_Downtime'
# ]

# # Fill missing feature values (optional: consider smarter imputation if needed)
# X = data[features].fillna(0)
# y = data['Days_Until_Next_Failure']

# # 4️⃣ Group-based Train-Test Split to prevent data leakage
# groups = data['Modified Equipment']  # OR ['Modified Equipment', 'Modified Part'] for stricter grouping

# split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# train_idx, test_idx = next(split.split(X, y, groups=groups))

# X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
# y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# # 5️⃣ Train XGBoost Model
# model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
# model.fit(X_train, y_train)

# # 6️⃣ Evaluate Model
# y_pred = model.predict(X_test)
# mae = mean_absolute_error(y_test, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# print(f"MAE: {mae:.2f} days")
# print(f"RMSE: {rmse:.2f} days")

# # 7️⃣ Save the model
# joblib.dump(model, 'equipment_rul_predictor_grouped.pkl')
# print("✅ Model saved to 'equipment_rul_predictor_grouped.pkl'")

# without grouping
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# 1. Load processed dataset
data = pd.read_csv("processed_maintenance_data_cleaned.csv")

# 2. Generate Target Variable: Days Until Next Failure
data = data.sort_values(['Modified Production Line', 'Modified Equipment', 'Modified Part', 'Clean Date'])
data['Clean Date'] = pd.to_datetime(data['Clean Date'], errors='coerce')

# Group by equipment-part and compute time difference to next failure
data['Days_Until_Next_Failure'] = data.groupby(['Modified Production Line', 'Modified Equipment', 'Modified Part'])['Clean Date'].shift(-1) - data['Clean Date']
data['Days_Until_Next_Failure'] = data['Days_Until_Next_Failure'].dt.days

# Drop rows where target is missing (e.g. last record of each group)
data = data.dropna(subset=['Days_Until_Next_Failure'])

# 3. Feature Selection
features = [
    'Failures_Last_30_Days', 'Failures_Last_60_Days', 'Failures_Last_90_Days',
    'Avg_Downtime_Mins', 'Days_Since_Last_Failure', 'Equipment_Age_Days',
    'Shift_Failure_Count', 'Line_Failure_Count', 'Hour_of_Day', 'Day_of_Week',
    'Month', 'Quarter', 'Rolling_Mean_30d', 'Rolling_Std_30d',
    'Rolling_Mean_90d', 'Rolling_Std_90d', 'Equipment_Health_Score',
    'Maintenance_Count', 'Is_Peak_Hour', 'Is_Weekend',
    'Part_Failure_Frequency', 'Avg_Part_Downtime',
    'Prev_Downtime', 'EMA_Downtime'
]

# Remove any features with too many missing values
X = data[features].fillna(0)
y = data['Days_Until_Next_Failure']

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train XGBoost Model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate Model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"MAE: {mae:.2f} days")
print(f"RMSE: {rmse:.2f} days")

# 7. Save model
joblib.dump(model, 'equipment_rul_predictor.pkl')
print("Model saved to 'equipment_rul_predictor.pkl'")