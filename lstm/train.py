import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib

# Load dataset
historical_data = pd.read_csv('data/Dataset.csv')

# Feature engineering for historical data
historical_data['Date'] = pd.to_datetime(historical_data['Date'])
historical_data['Month'] = historical_data['Date'].dt.month
historical_data['Day'] = historical_data['Date'].dt.day
historical_data['Year'] = historical_data['Date'].dt.year
historical_data['dayofweek'] = historical_data['Date'].dt.dayofweek

# Define features and target
features = ['dayofweek', 'tempmax', 'tempmin', 'temp', 'dew', 'humidity', 'solarenergy', 'Month', 'Day', 'Year']
target = 'Max Demand Met (MW)'

X = historical_data[features].values
y = historical_data[target].values

# Normalize features
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Reshape input for LSTM [samples, time steps, features]
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Split data into training and validation sets
split_idx = int(0.8 * len(X_reshaped))
X_train, X_val = X_reshaped[:split_idx], X_reshaped[split_idx:]
y_train, y_val = y_scaled[:split_idx], y_scaled[split_idx:]

# Build LSTM model
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(1, X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val), verbose=1)

# Validate the model
y_pred_scaled = model.predict(X_val)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_val_actual = scaler_y.inverse_transform(y_val)

# Compute error metrics
rmse = np.sqrt(mean_squared_error(y_val_actual, y_pred))
mae = mean_absolute_error(y_val_actual, y_pred)
r2 = r2_score(y_val_actual, y_pred)

print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'R²: {r2}')

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_val_actual, label='Actual Values', linestyle='-', marker='o', color='blue')
plt.plot(y_pred, label='Predicted Values', linestyle='-', marker='x', color='red')
plt.xlabel('Data Points')
plt.ylabel('Electricity Demand (MW)')
plt.title('Actual vs Predicted Electricity Demand')
plt.legend()
plt.grid()
plt.show()

# Save the model
model.save('electric_demand_lstm.h5')
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')
