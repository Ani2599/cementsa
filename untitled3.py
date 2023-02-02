import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.title('Forecast of Cement Sales')
uploaded_file = st.file_uploader(" ", type=['xlsx'])

if uploaded_file is not None:     
    cement = pd.read_excel(uploaded_file)
    cement['Month'] = cement['Month'].apply(lambda x: x.strftime('%B-%Y'))
    
    hwe_model_mul_add = ExponentialSmoothing(cement["Sales"][:71], seasonal = "mul", trend = "add", seasonal_periods = 12).fit()
    
    newdata_pred = hwe_model_mul_add.predict(start = cement.index[0], end = cement.index[-1])
    
    st.subheader("For exponential model")
   
    st.write("Sales Forecast: ", newdata_pred)
   
    
    st.subheader("Thanks for visit.")