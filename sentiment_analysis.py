""" This is a Stock Sentiment Analyzer on recent news and Investment advise program """

import pandas as pd
import matplotlib.pyplot as plt
import hvplot.pandas
import datetime as dt
import nltk
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from GoogleNews import GoogleNews
from newspaper import Article
from newspaper import Config
from sympy import true
from wordcloud import WordCloud, STOPWORDS
import yahoo_fin.stock_info as si
nltk.download("vader_lexicon")
    
# A- Formatting and Extracting (scraping) the News data for 1 day
def date_formatter():
    now = dt.date.today()
    now = now.strftime("%m-%d-%Y")
    yesterday = dt.date.today() - dt.timedelta(days = 1)
    yesterday = yesterday.strftime("%m-%d-%Y")

    nltk.download("punkt")
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0'
    config = Config()
    config.browser_user_agent = user_agent
    config.request_timeout = 10

    return now, yesterday, config


# Extract News with Google News
@st.cache
def news_scraper(yesterday, now, ticker_or_stock_name, config):
    googlenews = GoogleNews(start = yesterday, end = now)
    googlenews.search(ticker_or_stock_name)
    googlenews.get_page(2)
    result = googlenews.result()

    # Store the results
    df = pd.DataFrame(result)
    
    # B- Summarizing the extracted News data - by downloading, parsing and performing natural language processing on the articles    
    try:
        list =[] #creating an empty list 
        for i in df.index:
            dict = {} #creating an empty dictionary to append an article in every single iteration
            article = Article(df['link'][i],config=config) #providing the link
            try:
                article.download() #downloading the article 
                article.parse() #parsing the article
                article.nlp() #performing natural language processing (nlp)
            except:
                pass 
        #storing results in our empty dictionary
            dict['Date']=df['date'][i] 
            dict['Media']=df['media'][i]
            dict['Title']=article.title
            dict['Article']=article.text
            dict['Summary']=article.summary
            dict['Key_words']=article.keywords
            list.append(dict)
        check_empty = not any(list)
        # print(check_empty)
        if check_empty == False:
            news_df=pd.DataFrame(list) #creating dataframe
    

    except Exception as e:
    #exception handling
        print("exception occurred:" + str(e))
        print("Looks like, there is some error in retrieving the data, Please try again or try with a different ticker." )

    return news_df


# C- Sentiment Analysis categorization of selected News articles into sentiment buckets
def percentage(news_df):

    # Assigning Initial Values
    positive = 0
    negative = 0
    neutral = 0
    # Creating empty lists
    news_list = []
    neutral_list = []
    negative_list = []
    positive_list = []

    # Iterating over the tweets in the dataframe
    for news in news_df['Summary']:
        news_list.append(news)
        analyzer = SentimentIntensityAnalyzer().polarity_scores(news)
        neg = analyzer['neg']
        neu = analyzer['neu'] # Maybe we can remove this
        pos = analyzer['pos']
        comp = analyzer['compound']  # Maybe we can remove this

        if neg > pos:
            negative_list.append(news)  # appending the news that satisfies this condition
            negative += 1  # increasing the count by 1
        elif pos > neg:
            positive_list.append(news)  # appending the news that satisfies this condition
            positive += 1  # increasing the count by 1
        elif pos == neg:
            neutral_list.append(news)  # appending the news that satisfies this condition
            neutral += 1  # increasing the count by 1

    # Converting lists to pandas dataframe
    news_list = pd.DataFrame(news_list)
    neutral_list = pd.DataFrame(neutral_list, columns=['Neutral News Articles'])
    negative_list = pd.DataFrame(negative_list, columns=['Negative News Articles'])
    positive_list = pd.DataFrame(positive_list, columns=['Positive News Articles'])

    # using len(length) function for counting
    print("Positive Sentiment:", '%.2f' % len(positive_list), end='\n')
    print("Neutral Sentiment:", '%.2f' % len(neutral_list), end='\n')
    print("Negative Sentiment:", '%.2f' % len(negative_list), end='\n')

    return positive_list, neutral_list, negative_list, positive, neutral, negative
    
# D- PieChart creation & word cloud visualiztion
def pie_chart_data(positive, negative, neutral):
    if neutral !=0:
        labels = ['Positive', 'Neutral', 'Negative']
        explode = (0.1, 0, 0)
        sizes = [positive, neutral, negative]
        colors = ['yellowgreen', 'blue', 'red']
    else:
        labels = ['Positive', 'Negative']
        explode = (0.1, 0)
        sizes = [positive, negative]
        colors = ['yellowgreen', 'red']
    patches, ax1 = plt.subplots()
    ax1.pie(
        sizes,
        labels = labels,
        colors=colors,
        explode=explode,
        shadow=True,
        autopct='%1.1f%%', 
        startangle=90
        )
    ax1.axis('equal')

    return patches

# word cloud visualization
def word_cloud(text):
    stopwords = set(STOPWORDS)
    allWords = ' '.join([nws for nws in text])
    wordCloud = WordCloud(background_color='black', width=1000, height=500, stopwords=stopwords, min_font_size=20,
                            max_font_size=150, colormap='prism').generate(allWords)

    return wordCloud
