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
