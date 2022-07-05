""" This is a Stock Sentiment Analyzer on recent news and Investment advise program """

import pandas as pd
import matplotlib.pyplot as plt
import hvplot.pandas
import datetime as dt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from GoogleNews import GoogleNews
from newspaper import Article
from newspaper import Config
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
def news_scraper(yesterday, now, ticker_or_stock_name, config):
    googlenews = GoogleNews(start = yesterday, end = now)
    googlenews.search(ticker_or_stock_name)
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

    return df, news_df


# C- Sentiment Analysis categorization of selected News articles into sentiment buckets

def percentage(part, whole, news_df):

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

    positive = percentage(positive, len(news_df))  # percentage is the function defined above
    negative = percentage(negative, len(news_df))
    neutral = percentage(neutral, len(news_df))

    # Converting lists to pandas dataframe
    news_list = pd.DataFrame(news_list)
    neutral_list = pd.DataFrame(neutral_list)
    negative_list = pd.DataFrame(negative_list)
    positive_list = pd.DataFrame(positive_list)

    # using len(length) function for counting
    print("Positive Sentiment:", '%.2f' % len(positive_list), end='\n')
    print("Neutral Sentiment:", '%.2f' % len(neutral_list), end='\n')
    print("Negative Sentiment:", '%.2f' % len(negative_list), end='\n')

    return 100 * float(part) / float(whole), positive_list, negative_list, neutral_list
    
# CAN THIS BE A GRAPHICAL OUTPUT IN THE STREAMLIT APPLICATION?
# D- PieChart creation & word cloud visualiztion
def pie_chart_data(positive, negative, neutral):
    labels = ['Positive [' + str(round(positive)) + '%]', 'Neutral [' + str(round(neutral)) + '%]',
                'Negative [' + str(round(negative)) + '%]']
    sizes = [positive, neutral, negative]
    colors = ['yellowgreen', 'blue', 'red']
    patches, texts = plt.pie(sizes, colors=colors, startangle=90)
    plt.style.use('default')
    plt.legend(labels)
    plt.title("Sentiment Analysis Result for stock= " + ticker_or_stock_name + "")
    plt.axis('equal')
    plt.show()

# word cloud visualization
def word_cloud(text, news_df):
    stopwords = set(STOPWORDS)
    allWords = ' '.join([nws for nws in text])
    wordCloud = WordCloud(background_color='black', width=1000, height=500, stopwords=stopwords, min_font_size=20,
                            max_font_size=150, colormap='prism').generate(allWords)
    fig, ax = plt.subplots(figsize=(20, 10), facecolor='k')
    plt.imshow(wordCloud)
    ax.axis("off")
    fig.tight_layout(pad=0)
    plt.show()

    print('Wordcloud for ' + ticker_or_stock_name)
    word_cloud(news_df['Summary'].values)

# Ryan put this into the streamlit
    #review of article data

    review_articles = input("Would you like to review the data used for the analysis: Yes or No  ").lower()
    

    if review_articles == "yes":
        print(news_df)
        print("")
        
    #Investment Advise utilzing the Sentiment Analysis
    
    advise = input("Would you like advice on how this Sentiment Analysis can be applied to your trading?: Yes or No  ").lower()
    
    if advise == "yes":
        ticker = input("Please input ticker symbol:  ").lower()
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