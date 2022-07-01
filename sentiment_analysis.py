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

next_step = "yes"
while next_step == "yes":
    
# A- Formatting and Extracting (scraping) the News data for 1 day
    now = dt.date.today()
    now = now.strftime("%m-%d-%Y")
    yesterday = dt.date.today() - dt.timedelta(days = 1)
    yesterday = yesterday.strftime("%m-%d-%Y")

    nltk.download("punkt")
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0'
    config = Config()
    config.browser_user_agent = user_agent
    config.request_timeout = 10

# save the company name in a variable
    ticker_or_stock_name = input("What is the Ticker or name of the Company you would like to analyze:  ")

# As long as the company name is valid not empty...
    if ticker_or_stock_name != '':
        print("Please wait a moment as we retrieve the top 10 recent articles for {} to analyze".format(ticker_or_stock_name))
        print("")

    # Extract News with Google News
        googlenews = GoogleNews(start=yesterday,end=now)
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


# C- Sentiment Analysis categorization of selected News articles into sentiment buckets

        def percentage(part, whole):
            return 100 * float(part) / float(whole)


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
            neu = analyzer['neu']
            pos = analyzer['pos']
            comp = analyzer['compound']

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
        
    

# D- PieChart creation & word cloud visualiztion

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
        def word_cloud(text):
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
            
        # request to look at another ticker or Company    
            
        next_step = input("Would you like to analyze another Ticker or Company?: Yes or No  ").lower()
    
        if next_step == "yes":
            continue
        else:
            break
        
trade = input("Do you want to export this ticker to the Trading Bot to trade:  Yes or No  ").lower()
ticker = input("Please input ticker symbol:  ").lower()
        
if trade == "yes":
    print(f"Exporting {ticker} to Trading Bot")
else:
    print("no")