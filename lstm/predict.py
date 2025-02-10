import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# Load the trained LSTM model and scalers
import tensorflow as tf

# Custom object dictionary
custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}

# Load the model with custom objects
model = tf.keras.models.load_model(
    "C://Users//Ramamuthukumaran s//OneDrive//Desktop//Electricity_Demand_analysis//electric_demand_lstm.h5",
    custom_objects=custom_objects
)


scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

# Load future dataset
future_data = pd.read_csv('C:\\Users\\Ramamuthukumaran s\\OneDrive\\Desktop\\Electricity_Demand_analysis\\data\\Final_Data.csv')

# Feature engineering for 2025 data
future_data['Date'] = pd.to_datetime(future_data['Date'])
future_data['Month'] = future_data['Date'].dt.month
future_data['Day'] = future_data['Date'].dt.day
future_data['Year'] = future_data['Date'].dt.year
future_data['dayofweek'] = future_data['Date'].dt.dayofweek

# Define features
features = ['dayofweek', 'tempmax', 'tempmin', 'temp', 'cloudcover', 'dew', 'humidity', 'solarenergy', 'Month', 'Day', 'Year']

# Preprocess and reshape data for LSTM
X_future = future_data[features].values
X_future_scaled = scaler_X.transform(X_future)
X_future_reshaped = X_future_scaled.reshape((X_future_scaled.shape[0], 1, X_future_scaled.shape[1]))

# Predict for 2025
y_pred_scaled = model.predict(X_future_reshaped)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Save predictions to a CSV
future_data['Predicted Max Demand (MW)'] = y_pred
future_data.to_csv('2025_demand_prediction_lstm.csv', index=False)
