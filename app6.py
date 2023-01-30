#from tkinter import font
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from math import sqrt
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot

st.title('Forecast of Cement Sales')
st.markdown('Sales Graph')

uploaded_file = st.file_uploader(" ", type=['xlsx']) #Only accepts excel file format

if uploaded_file is not None:     
    cement = pd.read_excel(uploaded_file)
    
    Train = cement.head(59)
    Test = cement.tail(12)

    ar_model = pm.auto_arima(Train.Sales, start_p=0, start_q=0,
                          max_p=12, max_q=12, # maximum p and q
                          m=1,              # frequency of series
                          d=None,           # let model determine 'd'
                          seasonal=True,   # No Seasonality
                          start_P=0, trace=True,
                          error_action='warn', stepwise=True)

    # Best Parameters ARIMA
    # ARIMA with AR = 4, I = 0, MA = 0
    model = ARIMA(Train.Sales, order = (4, 0, 0))
    res = model.fit()
    print(res.summary())

    # Forecast for next 12 months
    start_index = len(Train)
    end_index = start_index + 11
    forecast_best = res.predict(start=start_index, end=end_index)

    print(forecast_best)

    # Evaluate forecasts
    rmse_best = sqrt(mean_squared_error(Test.Sales, forecast_best))
    print('Test RMSE: %.3f' % rmse_best)

    # plot forecasts against actual outcomes
    pyplot.plot(Test.Sales)
    pyplot.plot(forecast_best, color='red')
    pyplot.show()

    # Forecast for future 12 months
    start_index = len(cement)
    end_index = start_index + 11
    forecast = res.predict(start = start_index, end = end_index)
    #########################################
    st.info("Forecasted Value")

    

    st.subheader("For Arima Model")
   
    st.write("Sales Forecast: ", forecast)
   
    
    st.subheader("Thanks for visit.")
