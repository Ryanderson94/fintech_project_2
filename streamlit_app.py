"""This is the streamlit input application"""

# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np

st.title("welcome to the fuckin' show")

ticker  = st.text_input(
    'Enter the ticker you would like to review', 
    )

st.title('Did you know that {ticker}? How crazy!')

# Input ticker name
ticker_or_stock_name = input("What is the Ticker or name of the Company you would like to analyze:  ")

if ticker_or_stock_name != '':
    print("Please wait a moment as we retrieve the top 10 recent articles for {} to analyze".format(ticker_or_stock_name))
    print("")