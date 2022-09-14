# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 09:40:14 2022

@author: Kunal
"""

import pandas as pd
import streamlit as st
from pycaret.time_series import *
from pycaret.datasets import get_data
import codecs
import streamlit.components.v1 as components
from streamlit_lottie import st_lottie
import json
import requests
import numpy as np
from  PIL import Image
## Define Functions to call lottie Animations
def load_lottieurl(url:str):
    r = requests.get(url)
    if r.status_code!=200:
        return None
    return r.json()

## Define function for Button clicked

if "button_clicked" not in st.session_state:
    st.session_state.button_clicked=False
    
def callback():
    #button was clicked
    st.session_state.button_clicked=True


data=get_data('pycaret_downloads')

data['Date']=pd.to_datetime(data['Date'])
data.set_index('Date',inplace=True)
train=int(len(data)*0.8)   ## Defining Training Data for example dataset
train2=data.iloc[:train]  ## Passing Training Data into a variable , First 80% values as training data
test=data.iloc[train:]    ## Passing Tessting Data into a variable , last 20% values as testing data

@st.cache
def load_data():
    data=get_data('pycaret_downloads')
    data['Date']=pd.to_datetime(data['Date'])
    data.set_index('Date',inplace=True)
    return data



## Define Streamlit Components

st.title("Web based Demand Forecasting Application")
st.caption("Use this web based application to forecast your demands - upload a file in CSV format")
lottie_hello=load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_49rdyysj.json")
st_lottie(lottie_hello,speed=1,reverse=True,loop=True,quality="high",height=250,width=1200,key=None)
st.sidebar.title(body="Prediction Application")
logo = Image.open(r'image3.jpg')
st.sidebar.image(logo,  width=320)
linkedIN=("https://www.linkedin.com/in/aadityahamine")
st.sidebar.write(linkedIN)
    
data_file=st.file_uploader("Want to predict what's your next demands are - upload a csv file here",type=['csv']) 
st.caption("CSV file should have first column named as Date followed by target figures in next column")

    

if data_file is not None:

    
        data_file=pd.read_csv(data_file)
        data_file['Date']=pd.to_datetime(data_file['Date'])
        data_file.set_index('Date',inplace=True)
        train_data=int(len(data_file)*0.8)
        train_data2=data_file.iloc[:train_data]   ## First 80% values as training data
        test_data2=data_file.iloc[train_data:]    ## Last 20% values as testing data
        s=setup(data_file,fold=3,fh=len(test_data2))  ## Pycaret function
        best=compare_models()
        final_best=finalize_model(best)
        st.caption("Model is ready , please select the time horizon from slider below to forecast the results")
        fh=st.slider('Select Forecast Period in Days',value=50,min_value=10,max_value=180,step=10)
        if st.button("Click here to predict the results"):
            st.write("Predicting Results for Days:",fh)
            predict_model(final_best,fh=fh)    ## fh figure in predict function tells how many time horizons you want to see ahead
            plot_model(best,plot='forecast',data_kwargs={'fh':fh})


else:
    
        st.caption("Don't have a CSV file to upload ? Try with our example dataset")
        
    
        if (st.button("Try with our example dataset",on_click=callback) or st.session_state.button_clicked):
            s=setup(data,fold=3,fh=len(test['Total']))  ## Pycaret function
            best=compare_models()
            final_best=finalize_model(best)
                    
            st.caption("Model is ready , please select the time horizon from slider below to forecast the results")
            fh=st.slider('Select Forecast Period in Days',value=50,min_value=10,max_value=180,step=10)

            if st.button("Click here to predict the results"):
                
                    st.write("Predicting Results for Days:",fh)
                    
                    predict_model(final_best,fh=fh)    ## fh figure in predict function tells how many time horizons you want to see ahead
                    plot=plot_model(best,plot='forecast',data_kwargs={'fh':fh})
                        
                                       


