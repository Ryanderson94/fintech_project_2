"""This is the streamlit input application"""

# Import required libraries
from datetime import date
from matplotlib import patches
import streamlit as st
import pandas as pd
import numpy as np # np mean, np random 
import yfinance as yf
from sentiment_analysis import date_formatter, news_scraper, percentage, pie_chart_data, word_cloud
from svm_algo import define_hyper_parameters, run_trading_bot

st.title("**Welcome to the AutoTrader Pro**")
st.header("A Powerful Tool for Empowering Investors")
st.subheader("*Creation of Team 1*")

now = date_formatter()[0]
yesterday = date_formatter()[1]
config = date_formatter()[2]

# Load data in for S&P 500 Companies
def load_data():
    components = pd.read_html('https://en.wikipedia.org/wiki/List_of_S&P_500_companies')[0]
    return components.drop("SEC filings", axis=1).set_index("Symbol")

# Define the Main function
def main():
    components = load_data()                                # Load in components
    title = st.empty()                                      # Set title
    st.sidebar.title("Please select a company to analyze")  # Set sidebar title

    def label(symbol):
        a = components.loc[symbol]
        return symbol + ' - ' + a.Security

    asset = st.sidebar.selectbox('Fortune 500 Companies',
                                 components.index.sort_values(), index=3,
                                 format_func=label)
    if st.sidebar.checkbox('Would you like to view more information about potential stocks?'):
        st.dataframe(components[['Security',
                                 'GICS Sector',
                                 'Date first added',
                                 'Founded']])
    test = components.astype(str)
    st.dataframe(test.loc[asset])
    
    # Define DF1
    df1 = news_scraper(now, yesterday, asset, config)
    
    positive_list, neutral_list, negative_list, positive_count, neutral_count, negative_count = percentage(df1)
    
    # Conditionally display news article tables
    if positive_count != 0:
        st.write(positive_list)

    if neutral_count != 0:
        st.write(neutral_list)
    
    if negative_count !=0:
        st.write(negative_list)
    
    # Display Pie Chart
    pie = pie_chart_data((len(positive_list)),(len(negative_list)),(len(neutral_list)))
    st.subheader(f"News Article Sentiment Analysis for {asset}")
    st.pyplot(pie)

    # Display WordCloud
    st.subheader(f'WordCloud for Pulled Articles Related to {asset}')
    wordcloud_diagram = word_cloud(df1['Summary'])
    st.image(wordcloud_diagram.to_array())

    title.title(components.loc[asset].Security)
    if st.sidebar.checkbox('Would you like to review historical stock pricing data about this firm?'):
        start = st.date_input("Please enter the date you would like to begin your analysis")
        end = st.date_input('Please select the date you would like to end your analysis window', value = pd.to_datetime('today'))
        df = yf.download(asset,start,end)['Adj Close']
        st.subheader(f'Historical Adjusted Close Price for {asset} from {start} to {end}')
        st.line_chart(df)
    
    amount  = st.number_input('Enter the amount you would like to trade', min_value=0, value=0)

    if amount != 0:
        st.markdown(f'We will begin trading ${amount} worth of {asset}.')
        stock, shift_time, start_date_historical, end_date_historical, periods_for_training_data, sma_short_window, sma_long_window, RSI_time_period, EMA_time_period = define_hyper_parameters(asset)
        run_trading_bot(stock, shift_time, start_date_historical, end_date_historical, periods_for_training_data, sma_short_window, sma_long_window, RSI_time_period, EMA_time_period)
    

if __name__ == '__main__':
    main()

