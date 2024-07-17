# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:49:55 2024

@author: Muneer A
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.preprocessing import minmax_scale
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_absolute_percentage_error

# Load data
data = pd.read_csv(r"C:\Users\Archana Raj\Desktop\360 PYTHON CODES\360 Project 2\DATASET\Cleaned Data1.csv")

# Convert 'Dateofbill' column to datetime format if not already done
data['Dateofbill'] = pd.to_datetime(data['Dateofbill'])

# Extract features from Dateofbill
data['WeekofYear'] = data['Dateofbill'].dt.isocalendar().week
data['month'] = data['Dateofbill'].dt.month
data['year'] = data['Dateofbill'].dt.year

# Clean the data and drop NaN values
data_clean = data[['Dateofbill', 'DrugName', 'Quantity', 'WeekofYear']].dropna()

# Identify the top 20 drugs based on their frequency in the data
top_drugs = data['DrugName'].value_counts().head(20).index

# Filter the data to include only the top 20 drugs 
data_top_drugs = data_clean[data_clean['DrugName'].isin(top_drugs)]

# Pivot the DataFrame to aggregate by week of year for each drug
pivot_data = data_top_drugs.pivot_table(
    index='WeekofYear',
    columns='DrugName',
    values='Quantity',
    aggfunc='sum'
)

# Fill NaN values with 0
pivot_data = pivot_data.fillna(0)

# Convert pivot table to DataFrame
data = pd.DataFrame(pivot_data)

# Plot each time series
for column in data.columns:
    plt.figure(figsize=(10, 6))
    plt.plot(data.index.values, data[column].values, marker='o', linestyle='-', label=column)
    plt.title(f'Time Series of {column}')
    plt.xlabel('Week of Year')
    plt.ylabel('Quantity')
    plt.legend()
    plt.show()

# Function to check stationarity
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    for key, value in result[4].items():
        print('Critical Values:')
        print(f'   {key}, {value}')

# Making series stationary if needed
def make_stationary(timeseries):
    if adfuller(timeseries)[1] > 0.05:
        timeseries_diff = timeseries.diff().dropna()
        return timeseries_diff
    else:
        return timeseries

train_size = 0.8

# Fit and forecast using SARIMAX
def fit_sarimax(timeseries, forecast_steps=4, train_size=0.8):
    train_size = int(len(timeseries) * train_size)
    train, test = timeseries[:train_size], timeseries[train_size:]
    train_stationary = make_stationary(train)
    
    model = SARIMAX(train_stationary, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))  # Adjust orders as necessary
    model_fit = model.fit(disp=False)
    
    # Forecast
    forecast = model_fit.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean
    
    # Calculate MAPE
    mape = mean_absolute_percentage_error(test.values[:forecast_steps], forecast_mean)
    
    return model_fit, forecast_mean, mape

# Plot SARIMAX forecasts with MAPE for the next 4 weeks only
forecast_steps = 4

for column in data.columns:
    plt.figure(figsize=(12, 8))
    
    # Plot original series
    plt.plot(data.index.values, data[column].values, marker='o', linestyle='-', label='Original')
    
    # Check stationarity
    print(f'Stationarity Check for {column}')
    check_stationarity(data[column])
    
    # Fit and forecast using SARIMAX for the next 4 weeks only
    print(f'SARIMAX Model for {column}')
    sarimax_fit, sarimax_forecast, sarimax_mape = fit_sarimax(data[column], forecast_steps=forecast_steps, train_size=train_size)
    
    # Create forecast index
    sarimax_forecast_index = np.arange(len(data), len(data) + forecast_steps)
    
    # Plot forecast for the next 4 weeks only
    plt.plot(sarimax_forecast_index, sarimax_forecast.values, marker='x', linestyle='--', label='SARIMAX Forecast (Next 4 Weeks)')
    
    # Plot settings
    plt.title(f'SARIMAX Forecast of {column} for Next 4 Weeks')
    plt.xlabel('Week of Year')
    plt.ylabel('Quantity')
    plt.legend()
    plt.show()
    
    # Print MAPE value
    print(f'SARIMAX MAPE for {column}: {sarimax_mape}')
    
    # Plot separately for the next 4 weeks
    plt.figure(figsize=(12, 8))
    
    # Plot original series
    plt.plot(data.index.values[-forecast_steps:], data[column].values[-forecast_steps:], marker='o', linestyle='-', label='Original')
    
    # Plot forecast for the next 4 weeks only
    plt.plot(sarimax_forecast_index, sarimax_forecast.values, marker='x', linestyle='--', label='SARIMAX Forecast (Next 4 Weeks)')
    
    # Plot settings
    plt.title(f'SARIMAX Forecast of {column} for Next 4 Weeks (Separate Plot)')
    plt.xlabel('Week of Year')
    plt.ylabel('Quantity')
    plt.legend()
    plt.show()


    
