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

# Review of article data
review_articles = st.text_input("Would you like to review the data used for the analysis: Yes or No  ").lower()
    
if review_articles == "yes":
    print(news_df)
    print("")

    advise = input("Would you like advice on how this Sentiment Analysis can be applied to your trading?: Yes or No  ").lower()
    
    if advise == "yes":
        ticker = st.text_input("Please input ticker symbol:  ").lower()

        # Need to figure this part out
        analyst = si.get_quote_table(ticker)

        analyst_df = pd.DataFrame.from_dict(analyst, orient = "index")

        print("")
        print("""FOR POSITIVE SENTIMENT: Look for current price near support levels; For example near the 52 Week Range lows or the Day's Range lower bounds for bounces 
                    or Breakout approve those levels supported by volume for short term long trading oportunites.   """)
        print("")
        print("""FOR NEGATIVE SENTIMENT: Look for current price near resistance levels; For example near the 52 Week Range high or the Day's Range upper bounds for drops 
                    or Breakout below those levels supported by volume for short term short trading oportunites.   """)
        print("")
        print (f"Current Financial Metrics for your reference: {analyst_df}") 

        print("")
        print(""" PLEASE NOTE, Our Proprietary Trading Algorithm can trade all these short term trading opportunites based on the Stock News Sentiment Analysis on your behalf. 
                                                        BEST Of LUCK TRADING! """)   
        print("")
        
    # Request to look at another ticker or Company    
        
    next_step = input("Would you like to analyze another Ticker or Company?: Yes or No  ").lower()

trade = st.text_input("Do you want to export this ticker to the Trading Bot to trade:  Yes or No  ").lower()
ticker = st.text_input("Please input ticker symbol:  ").lower()
        
if trade == "yes":
    print(f"Exporting {ticker} to Trading Bot")
else:
    print("no")