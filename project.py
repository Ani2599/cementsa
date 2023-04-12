import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot

cement = pd.read_excel(r"D:/Aniket Project/cementfinal.xlsx")
cement['Month'] = cement['Month'].apply(lambda x: x.strftime('%B-%Y'))

cement['Sales'].plot()
sns.boxplot(cement['Sales'])
plt.hist(cement['Sales'])
cement.Sales.skew()
cement.Sales.kurtosis()


cement["t"] = np.arange(1,72)
cement["t_square"] = cement["t"] * cement["t"]
cement["log_sales"] = np.log(cement["Sales"])
cement.columns

p = cement["Month"][0]
p[0:3]

cement['months']= 0

for i in range(71):
    p = cement["Month"][i]
    cement['months'][i]= p[0:3]
    
month_dummies = pd.DataFrame(pd.get_dummies(cement['months']))
cement1 = pd.concat([cement, month_dummies], axis = 1)

Train = cement1.head(59)
Test = cement1.tail(12)
####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('Sales ~ t', data = Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales']) - np.array(pred_linear))**2))
rmse_linear

##################### Exponential ##############################

Exp = smf.ols('Sales ~ t', data = Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales']) - np.array(np.exp(pred_Exp)))**2))
rmse_Exp

#################### Quadratic ###############################

Quad = smf.ols('Sales ~ t + t_square', data = Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t", "t_square"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales']) - np.array(pred_Quad))**2))
rmse_Quad

################### Additive seasonality ########################

add_sea = smf.ols('Sales ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov', data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales']) - np.array(pred_add_sea))**2))
rmse_add_sea

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_sales ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales']) - np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

################## Additive Seasonality Quadratic Trend ############################

add_sea_Quad = smf.ols('Sales ~ t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov', data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 

################## Multiplicative Seasonality Linear Trend  ###########

Mul_Add_sea = smf.ols('log_sales ~ t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales']) - np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_Mult_sea","rmse_add_sea_quad","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_Mult_sea,rmse_add_sea_quad,rmse_Mult_add_sea])}
table_rmse = pd.DataFrame(data)
table_rmse

model_full = smf.ols('Sales ~ t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov', data = cement1).fit()

predict_data = pd.read_excel(r"D:/Aniket Project/predict.xlsx")

pred_new = pd.Series(model_full.predict(predict_data))
pred_new

predict_data["forecasted_Sales"] = pd.Series(pred_new)

# Autoregression Model (AR)
# Calculating Residuals from best model applied on full data
# AV - FV
full_res = cement1.Sales - model_full.predict(cement1)

# ACF plot on residuals
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(full_res, lags = 12)
# ACF is an (complete) auto-correlation function gives values 
# of auto-correlation of any time series with its lagged values.

# PACF is a partial auto-correlation function. 
# It finds correlations of present with lags of the residuals of the time series 
tsa_plots.plot_pacf(full_res, lags=12)

# Alternative approach for ACF plot
# from pandas.plotting import autocorrelation_plot
# autocorrelation_ppyplot.show()

# AR model
from statsmodels.tsa.ar_model import AutoReg
model_ar = AutoReg(full_res, lags=[1])
# model_ar = AutoReg(Train_res, lags=5)
model_fit = model_ar.fit()

print('Coefficients: %s' % model_fit.params)

pred_res = model_fit.predict(start=len(full_res), end=len(full_res)+len(predict_data)-1, dynamic=False)
pred_res.reset_index(drop=True, inplace=True)

# The Final Predictions using ASQT and AR(1) Model
final_pred = pred_new + pred_res
final_pred
############################## ARIMA ##############
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing 

cement = pd.read_excel(r"D:/Aniket Project/cementfinal.xlsx")
Train = cement.head(59)
Test = cement.tail(12)

tsa_plots.plot_acf(cement.Sales, lags = 12)
tsa_plots.plot_pacf(cement.Sales,lags = 12)

# ARIMA with AR=1, MA = 12(1,1,12),(4,1,12)
model1 = ARIMA(Train.Sales, order = (4,1,12))
res1 = model1.fit()
print(res1.summary())

# Forecast for next 12 months
start_index = len(Train)
end_index = start_index + 11
forecast_test = res1.predict(start = start_index, end = end_index)

print(forecast_test)

# Evaluate forecasts
rmse_test = sqrt(mean_squared_error(Test.Sales, forecast_test))
print('Test RMSE: %.3f' % rmse_test)

# plot forecasts against actual outcomes
pyplot.plot(Test.Sales)
pyplot.plot(forecast_test, color='red')
pyplot.show()

# Auto-ARIMA - Automatically discover the optimal order for an ARIMA model.
# pip install pmdarima --user
import pmdarima as pm

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

print(forecast)

######################## Data Driven ###############
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # Holt Winter's Exponential Smoothing

cement = pd.read_excel(r"D:/Aniket Project/cementfinal.xlsx")
cement.Sales.plot()

Train = cement.head(59)
Test = cement.tail(12)

decompose_ts_add = seasonal_decompose(cement.Sales, model = "additive", period = 12)
print(decompose_ts_add.trend)
print(decompose_ts_add.seasonal)
print(decompose_ts_add.resid)
print(decompose_ts_add.observed)
decompose_ts_add.plot()

decompose_ts_mul = seasonal_decompose(cement.Sales, model = "multiplicative", period = 12)
decompose_ts_mul.plot()

# Creating a function to calculate the MAPE value for test data 
def MAPE(pred, org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)

mv_pred = cement["Sales"].rolling(4).mean()
mv_pred.tail(4)
MAPE(mv_pred.tail(4), Test.Sales)

cement.Sales.plot(label = "org")
for i in range(2, 9, 2):
    cement["Sales"].rolling(i).mean().plot(label = str(i))
plt.legend(loc = 3)


import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(cement.Sales, lags = 12)
tsa_plots.plot_pacf(cement.Sales, lags=12)

# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["Sales"]).fit()
pred_ses = ses_model.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_ses, Test.Sales) 

# Holt method 
hw_model = Holt(Train["Sales"]).fit()
pred_hw = hw_model.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hw, Test.Sales) 

# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["Sales"], seasonal = "add", trend = "add", seasonal_periods = 12).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hwe_add_add, Test.Sales) 

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Sales"], seasonal = "mul", trend = "add", seasonal_periods = 12).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hwe_mul_add, Test.Sales) 

# Final Model on 100% Data
hwe_model_mul_add = ExponentialSmoothing(cement["Sales"], seasonal = "mul", trend = "add", seasonal_periods = 12).fit()

# Load the new data which includes the entry for future 4 values
new_data = pd.read_excel(r"D:/Aniket Project/datadriven.xlsx")

newdata_pred = hwe_model_mul_add.predict(start = new_data.index[0], end = new_data.index[-1])
newdata_pred


#####################################
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
import pandas as pd
from fbprophet import prophet
pip install pystan
conda install -c conda-forge fbprophet
cement = pd.read_excel(r"C:/Users/Aniket/Downloads/All India_Features_07012023 (1).xlsx")
Train = cement.head(59)
Test = cement.tail(12)


pip install prophet


























