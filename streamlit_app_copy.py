"""This is the streamlit input application"""

# Import required libraries
from datetime import date
import streamlit as st
import pandas as pd
import numpy as np # np mean, np random 
import yfinance as yf
from sentiment_analysis_copy import date_formatter, news_scraper, percentage, pie_chart_data, word_cloud

st.title("**Welcome to the AutoTrader Pro**")
st.header("A Powerful Tool for Empowering Investors")
st.subheader("*Creation of Team 1*")
 
prescreen = st.radio(
     "Are you interested in reviewing recent trend data on a potential stock?",
     ('Yes', 'No'))

if prescreen == 'Yes':
     ticker  = st.text_input('Enter the ticker you would like to review', "")
else:
     st.write("You chose not to review recent trend data")

if ticker != "":
    st.markdown(f'You entered the stock ticker : {ticker}')

now = date_formatter()[0]
yesterday = date_formatter()[1]
config = date_formatter()[2]

news_scraper(now, yesterday, ticker, config)
# This function should return a dataframe and we need to figure out how to pass it into the function below
percentage(part=.5, whole=1, '''This is where the dataframe from the above needs to go''')

st.display(percentage()[1])

postscreen = st.radio(
     "Would you like to review an additional stock?",
     ('Yes', 'No'))

if postscreen == 'Yes':
     ticker2  = st.text_input('Enter the new ticker you would like to review', "")
else:
     st.write("You chose not to review an additional stock's recent trend data")

if ticker != "":
    st.markdown(f'You entered the stock ticker : {ticker2}')


def load_data():
    components = pd.read_html('https://en.wikipedia.org/wiki/List_of_S&P_500_companies')[0]
    return components.drop("SEC filings", axis=1).set_index("Symbol")

def main():
    components = load_data()
    title = st.empty()
    st.sidebar.title("Please select a company to analyze")

    def label(symbol):
        a = components.loc[symbol]
        return symbol + ' - ' + a.Security

    asset = st.sidebar.selectbox('Fortune 500 Companies',
                                 components.index.sort_values(), index=3,
                                 format_func=label)
    title.title(components.loc[asset].Security)
    if st.sidebar.checkbox('Would you like to view more information about the options?'):
        st.dataframe(components[['Security',
                                 'GICS Sector',
                                 'Date first added',
                                 'Founded']])
    if st.sidebar.checkbox('Would you like to review additional company data?'):
        start = st.date_input("Please enter the date you would like to begin your analysis")
        end = st.date_input('Please select the date you would like to end your analysis window', value = pd.to_datetime('today'))
        df = yf.download(asset,start,end)['Adj Close']
        st.line_chart(df)
        st.table(components.loc[asset])

if __name__ == '__main__':
    main()

amount  = st.number_input('Enter the amount you would like to trade', min_value=0, value=0)

if amount != 0:
    st.markdown(f'You entered the desired trade amount : ${amount}')
