# -*- coding: utf-8 -*-
"""
Created on Thu May 30 21:54:58 2024

@author: Muneer A
"""

#sarimax deployement :
    
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_percentage_error

# Function to check stationarity
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    st.write('ADF Statistic:', result[0])
    st.write('p-value:', result[1])
    for key, value in result[4].items():
        st.write('Critical Values:')
        st.write(f'   {key}, {value}')

# Making series stationary if needed
def make_stationary(timeseries):
    if adfuller(timeseries)[1] > 0.05:
        timeseries_diff = timeseries.diff().dropna()
        return timeseries_diff
    else:
        return timeseries

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
    forecast_conf_int = forecast.conf_int()
    
    # Calculate MAPE
    mape = mean_absolute_percentage_error(test.values[:forecast_steps], forecast_mean)
    
    return model_fit, forecast_mean, forecast_conf_int, mape

# Streamlit app for uploading new data
st.title('Drug Sales Forecasting')
uploaded_file = st.file_uploader("Upload new drug sales data", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head())  # Display uploaded data

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

    # Allow user to select a drug
    selected_drug = st.selectbox("Select Drug", data.columns)

    if selected_drug:
        # Plot the selected drug's time series
        st.write(f"Time Series of {selected_drug}")
        plt.figure(figsize=(10, 6))
        plt.plot(data.index.values, data[selected_drug].values, marker='o', linestyle='-', label=selected_drug)
        plt.title(f'Time Series of {selected_drug}')
        plt.xlabel('Week of Year')
        plt.ylabel('Quantity')
        plt.legend()
        st.pyplot(plt)

        # Check stationarity
        st.write(f'Stationarity Check for {selected_drug}')
        check_stationarity(data[selected_drug])
        
        # Fit and forecast using SARIMAX
        st.write(f'SARIMAX Model for {selected_drug}')
        sarimax_fit, sarimax_forecast, sarimax_conf_int, sarimax_mape = fit_sarimax(data[selected_drug])

        # Create forecast index
        sarimax_forecast_index = np.arange(len(data), len(data) + len(sarimax_forecast))

        # Display forecasted quantities and confidence intervals
        st.write(f"Forecasted Quantities for next 4 weeks: {sarimax_forecast.values}")
        st.write(f"Confidence Intervals for next 4 weeks: {sarimax_conf_int.values}")

        # Plot forecast
        plt.figure(figsize=(12, 8))
        plt.plot(data.index.values, data[selected_drug].values, marker='o', linestyle='-', label='Historical Data')
        plt.plot(sarimax_forecast_index, sarimax_forecast.values, marker='x', linestyle='--', label='SARIMAX Forecast')
        plt.fill_between(sarimax_forecast_index, sarimax_conf_int.iloc[:, 0], sarimax_conf_int.iloc[:, 1], color='k', alpha=0.1)
        plt.title(f'SARIMAX Forecast of {selected_drug} for Next 4 Weeks')
        plt.xlabel('Week of Year')
        plt.ylabel('Quantity')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)
        
        # Print MAPE value
        st.write(f'SARIMAX MAPE for {selected_drug}: {sarimax_mape}')

        # Plot separately for the next 4 weeks
        st.write(f'Forecast for next 4 weeks separately for {selected_drug}')
        plt.figure(figsize=(12, 8))
        plt.plot(sarimax_forecast_index, sarimax_forecast.values, marker='x', linestyle='--', label='SARIMAX Forecast')
        plt.fill_between(sarimax_forecast_index, sarimax_conf_int.iloc[:, 0], sarimax_conf_int.iloc[:, 1], color='k', alpha=0.1)
        plt.title(f'SARIMAX Forecast of {selected_drug} for Next 4 Weeks (Separate Plot)')
        plt.xlabel('Week of Year')
        plt.ylabel('Quantity')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

    
