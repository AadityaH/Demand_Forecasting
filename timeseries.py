# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 19:33:30 2022

@author: Kunal
"""

import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf 
from statsmodels.tsa.seasonal import seasonal_decompose      
from pmdarima import auto_arima                              
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse
import warnings
warnings.filterwarnings("ignore")
import streamlit.components.v1 as components
import matplotlib as plt

st.title("Demand Forecasting in Minutes")
st.caption("a web based app to Forecast your future demands ")


df=pd.read_csv('timeseries.csv',index_col='Date',parse_dates=True)
df.dropna(axis=0,inplace=True)
df.index = pd.DatetimeIndex(df.index).to_period('M')   ## Convert to Frequency M

df2=df.groupby(['Date','Model']).agg(sales=('Count',np.sum)).reset_index()
options=['M119']
dfmodel=df2.loc[df2['Model'].isin(options)]
dfmodel=dfmodel.set_index('Date')
dfmodel['Date'].isna().sum()
dfmodel2=dfmodel.groupby([dfmodel['Date'].dt.month,'Model']).sum().reset_index()
dfmodel2['Year']=dfmodel2['Date'].dt.year
dfmodel2['Month']=dfmodel2['Date'].dt.month
dfmodel3=df.loc[df['Model'].isin(options)]
## Defining ARIMA Model

#auto_arima(dfmodel2['sales'], seasonal=True,m=1).summary()
auto_arima(dfmodel3['Count'],seasonal=True,m=12).summary()

## Train Test Split
train_len=int(len(dfmodel3['Count'])*0.8)      ## Defining training data length in Integer           
train= dfmodel3.iloc[:train_len]  ## First 80% values
test=dfmodel3.iloc[train_len:] ## Last 20% Values

## Fitting the model

model=SARIMAX(train['Count'],order=(0,1,1),seasonal_order=(1,0,1,12))  
results=model.fit()
results.summary()



## Obtain Predicted Values

start=len(train)
end=len(train)+len(test)-1
predictions=results.predict(start=start,end=end,dynamic=False,typ='Levels')


## Plot details

ax=test['Count'].plot(legend=True,figsize=(18,6)) 
predictions.plot()                