# The AutoTrader Pro

![AutoTrader](https://github.com/Ryanderson94/fintech_project_1/blob/main/Readme%20Resources/Capture.PNG)

## Overview & Features
Fueled by our success following the deployment of the AutoTrader, we have engineered a scalable, adaptable, and succesfull iteration of the orginal AutoTrader dubbed: "The AutoTrader Pro". The Pro version is for users with more knowledge about trading and who have more time than those who use the regular AutoTrader. But what makes the AutoTrader Pro truly different from the original?

This Application features the following upgrades:

1. Sentiment Analysis of Fortune 500 stocks using the GoogleNews API.
2. State-of-the-art engine powered by neural networks and eight other Machine Learning activations.
3. Intuitive, scalable, and adaptive UI for users to easily select their preferred stock and begin trading.

### NOTE
* A subscription to utilize Alpaca's Api is needed to run the AutoTrader Pro.
* The AutoTrader Pro can only execute trades during market hours (9:30 AM - 4 PM).

![Market](https://github.com/Ryanderson94/fintech_project_1/blob/main/Readme%20Resources/nick-chong-N__BnvQ_w18-unsplash.jpg)

---

## Technologies

The AutoTrader Pro utilizes **Python (v 3.9.7)** and the following libraries (in no specific order):

1. `os`
2. `alpaca_trade_api as tradeapi`
3. `ast`
4. `time`
5. `json`
6. `pandas as pd`
7. `numpy as np`
8. `numpy.random as np`
9. `load_dotenv from dotenv`
10. `requests`
11. `Path from pathlib`
12. `hvplot.pandas`
13. `sqlalchemy`
14. `tensorflow as tf`
15. `Dense from tensorflow.keras.layers`
16. `Sequential from tensorflow.keras.models`
17. `train_test_split from sklearn.model_selection`
18. `StandardScaler, OneHotEncoder from sklearn.preprocessing`
19. `REST, TimeFrame from alpaca_trade_api`
20. `talib as ta`
21. `websocket`
22. `plotly.graph_objects as go`
23. `plotly.express as px`
24. `datetime, timedelta from datetime`
25. `matplotlib.pyplot as plt`
26. `nltk`
27. `SentimentIntensityAnalyzer from nltk.sentiment.vader`
28. `GoogleNews from GoogleNews`
29. `Article, Config from newspaper`
30. `WordCloud, STOPWORDS from wordcloud`
31. `yahoo_fin.stock_info as si`

---

## Installation Guide

A large majority of the libraries are included with the Python version above. All other libraries can be installed using the Pip Package Manager or using the Conda Forge method.

[PIP Install Support Web Site](https://packaging.python.org/en/latest/tutorials/installing-packages/#ensure-you-can-run-python-from-the-command-line)

---

## AutoTrader Pro Table of Contents

Please use the following table of contents for reference:

1. [Sentiment Analysis](./sentiment_analysis.ipynb)
2. [Neural Network](./Trading_Bot_Deep_nn_loop_bot_connected.ipynb)
3. [SVM](./Trading_Bot_SVM_TA_v2_loop.ipynb)
4. [Streamlit](./streamlit_app.py)
5. [Streamlit Sentiment Analysis](./sentiment_analysis_copy.py)
6. [Streamlit Neural Network Algo](./nn_algo.py)
    
---

### Sentiment Analysis

A new feature of the AutoTrader Bot is the Sentiment Analysis. Our sentiment analysis, powered by MLTK Vader, gauges the sentiment of the market regarding the particular stock the user has selected using NLP analysis. With this information, the user will be better informed before placing a trade.

The data collection utilizes the GoogleNews API to fetch the first 10 articles relating to the ticker. The data is parsed and categorized into the applicable sentiment categories which are Positive, Negative and Neutral and visualized for the user in a pie chart and a word cloud as a reference on the current sentiment of the ticker.

With this information, the user can decide if they would like the AutoTrader to execute trades. If the selection is yes, the ticker is passed on to the Trading Bot where the magic starts.

---

### Neural Network

Utilizing Neural Networks creates a data-driven trading signal and a probability/confidence value. We use this confidence value to help the AutoTrader Pro determine the following:

1. Execute the trade
2. Hold the position
3. Short-sell the trade
4. Buy the trade
    
Across nine compounding TA's, a multitude of hidden layers and neurons, and 500 epochs, our model has achieved a significant predicted classification value of 68% accuracy and 60% annualized returns.

![spread](https://github.com/Ryanderson94/fintech_project_1/blob/main/Readme%20Resources/500bt1.PNG)

Bear Market                              | Bull Market                    | Sideways Market
:---------------------------------------:|:------------------------------------:|:------------------------------|
![bear](https://github.com/Ryanderson94/fintech_project_1/blob/main/Readme%20Resources/500bt2.PNG)     | ![bull](https://github.com/Ryanderson94/fintech_project_1/blob/main/Readme%20Resources/500bt3.PNG)    | ![side](https://github.com/Ryanderson94/fintech_project_1/blob/main/Readme%20Resources/500bt4.PNG)

---

### Support Vector Machines

Due to its robust and reliable nature as a model with one of the highest prediction accuracy values that our classification could predict, SVM was a natural path for us to guide our model. Further backtesting provided the following results:

1. Annualized return of 49%
2. Sharpe Ratio of 2.7
    
---

### Streamlit

To offer our users an intuitive and simple UI, we opted to utilize Streamlit and created a dashboard where the user selects a stock from a dropdown that would then offer sentiment analysis and further trading information based a collection of streamed data.

---

## Development Pipeline

Upon succesful deployment of the AutoTrader Pro, we aim to create a Smart Contract system to not only allow trading of CryptoCurrencies, but to create a verifiable Block Chain that the AutoTrader ecosystem will be fueled by.

Final deployment onto a real trading account will be the penultimate venture before our initial public offering.

---

## Contributors

Contributors for the development and deployment of the AutoTrader Pro include:

1. Ryan Anderson:
    * Repository Administrator
    * Streamlit Engineer
2. Tao Chen:
    * Neural Network Model
    * SVM Model
3. James Handral:
    * Sentiment Analysis
    * README
4. Colton Mayes:
    * Streamlit Engineer
    * Final Presentation
5. Anton Maliksi:
    * Final Presentation
    * README
    
---

## Licenses

No licenses were used for the AutoTrader Pro.