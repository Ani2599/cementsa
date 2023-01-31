import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # Holt Winter's Exponential Smoothing

st.title('Forecast of Cement Sales')
st.markdown('Sales Graph')

uploaded_file = st.file_uploader(" ", type=['xlsx']) #Only accepts excel file format
new_data = st.file_uploader("", type = ['xlsx'])

if uploaded_file is not None:     
    cement = pd.read_excel(uploaded_file)
    cement['Month'] = cement['Month'].apply(lambda x: x.strftime('%B-%Y'))
    
    hwe_model_mul_add = ExponentialSmoothing(cement["Sales"], seasonal = "mul", trend = "add", seasonal_periods = 12).fit()

    
    

    newdata_pred = hwe_model_mul_add.predict(start = new_data.index[0], end = new_data.index[-1])
    newdata_pred
    
    
  
    

    st.subheader("For exponential model")
   
    st.write("Sales Forecast: ", newdata_pred)
   
    
    st.subheader("Thanks for visit.")

