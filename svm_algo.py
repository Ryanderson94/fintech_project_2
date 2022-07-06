"""This is the algorithm leveraging the Support Vector Machine supervised learning method"""

# Install required libraries
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
from alpaca_trade_api import TimeFrame
import talib as ta
from datetime import datetime, timedelta
from pandas.tseries.offsets import DateOffset

from utils.helper import get_alpacas_info

# Set API information from alpacas
api = get_alpacas_info()[0]
alpaca_api_key = get_alpacas_info()[1]
alpaca_secret_key = get_alpacas_info()[2]

# use timedelta from Datetime to generate earlier date so we can extract historical data from alpaca later

def define_hyper_parameters(asset):
    days_to_subtract = 7
    today = (datetime.today()).strftime('%Y-%m-%d')

    earlier_date_to_compare = (datetime.today()-timedelta(days=days_to_subtract)).strftime('%Y-%m-%d')

    # Define parameters for the algorithm
    stock = asset                                     # Stock ticker
    shift_time = 1                                      # Offset shift
    start_date_historical = earlier_date_to_compare     # Historical start date
    end_date_historical = today                         # Historical end date
    periods_for_training_data = 800                     # Period count for training data
    sma_short_window = 20                               # Simple Moving Average short window length
    sma_long_window = 50                                # Simple Moving Average long window length
    RSI_time_period = 3                                 # RSI Time Period
    EMA_time_period = 50                                # EMA Time Period

    return stock, shift_time, start_date_historical, end_date_historical, periods_for_training_data, sma_short_window, sma_long_window, RSI_time_period, EMA_time_period

# Run the signal function
def run_trading_bot(
    stock, 
    shift_time, 
    start_date_historical, 
    end_date_historical, 
    periods_for_training_data, 
    sma_short_window, 
    sma_long_window, 
    RSI_time_period, 
    EMA_time_period
    ):

    while True:
        bars = api.get_bars(stock, TimeFrame.Minute, start_date_historical, end_date_historical).df
        bars['actual_returns'] = bars['close'].pct_change()
        bars = bars.dropna()
        
        # Use ta-lib to calcualte the simple moving average (SMA)
        bars['sma_short_window'] = ta.SMA(bars['close'],timeperiod=sma_short_window)
        bars['sma_long_window'] = ta.SMA(bars['close'],timeperiod=sma_long_window)

        bars['RSI'] = ta.RSI(bars['close'],timeperiod=RSI_time_period)
        bars['EMA'] = ta.RSI(bars['close'],timeperiod=EMA_time_period)

        # Drop NaN values in columns
        bars = bars.dropna()

        # Assign a copy of the technical variable columns to the Features dataframe and lag it
        X = bars[['sma_short_window', 'sma_long_window', 'RSI', 'EMA', ]].shift(shift_time).dropna().copy()

        # Create a new signal column and default value to '0'
        bars['signal'] = 0.0

        # Generate signal to buy stock long
        bars.loc[(bars['actual_returns'] >= 0), 'signal'] = 1

        # Generate signal to sell stock short
        bars.loc[(bars['actual_returns'] < 0), 'signal'] = -1
        
        # Set contrarian signal as inverse of signal generated based on actual returns
        bars['contrarian_signal']= bars['signal']*(-1)
        
        # Calculate the strategy returns and add them to the signals_df DataFrame
        # add shift after signal is because the buying/selling will be one day delay after the reutrn
        bars['Strategy_Returns'] = bars['actual_returns'] * bars['signal'].shift(shift_time)
        bars['contrarian_Strategy_Returns'] = bars['actual_returns'] * bars['contrarian_signal'].shift(shift_time)
        
        # Copy the "signal" column to the target series called `y`
        y = bars['signal']

        # Set the start and end of the training period
        training_begin = X.index.min()
        training_end = X.index.min() + DateOffset(minutes=periods_for_training_data)
        
        # Generate the X_train and y_train DataFrames
        X_train = X.loc[training_begin:training_end]
        y_train = y.loc[training_begin:training_end]

        # Generate the X_test and y_test DataFrames
        X_test = X.loc[training_end:]
        y_test = y.loc[training_end:]
        
        # Scale the training and testing data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        
        X_scaler = scaler.fit(X_train)
        
        X_train_scaled = X_scaler.transform(X_train)
        
        X_test_scaled = X_scaler.transform(X_test)

        from imblearn.over_sampling import RandomOverSampler
        
        # Resample the data
        ros = RandomOverSampler(random_state=1)
        X_resampled, y_resampled = ros.fit_resample(X_train_scaled, y_train)

        # Create the classifier model.
        from sklearn.svm import SVC
        model = SVC()

        # Fit the model to the data using X_train_scaled and y_train
        model = model.fit(X_resampled, y_resampled)
        
        # Use the trained model to predict the trading signals for the training data.
        training_signal_predictions = model.predict(X_resampled)

        # Evaluate the model using a classification report
        from sklearn.metrics import classification_report
        training_report = classification_report(y_resampled, training_signal_predictions)
        print(training_report)

        # Use the trained model to predict the trading signals for the training data.
        training_signal_predictions = model.predict(X_test_scaled)

        # NEED COMMENT HERE
        contrarian_training_signal_predictions = training_signal_predictions * (-1)

        # Create a predictions DataFrame
        predictions_df = pd.DataFrame(index=X_test.index)

        # Add the SVM model predictions to the DataFrame
        predictions_df['Predicted'] = training_signal_predictions

        # NEED COMMENT HERE
        predictions_df['Contrarian_Predicted'] = contrarian_training_signal_predictions

        # Add the actual returns to the DataFrame
        predictions_df['actual_returns'] = bars['actual_returns'] 

        # Add the strategy returns to the DataFrame
        predictions_df['Strategy_Returns'] = (
            predictions_df["actual_returns"] * predictions_df["Predicted"]
        )

        # COMMENT
        predictions_df['Strategy_Returns'] = (
            predictions_df["actual_returns"] * predictions_df["Predicted"]
        )

        # COMMENT
        predictions_df['Contrarian_Strategy_Returns'] = (
            predictions_df["actual_returns"] * predictions_df["Contrarian_Predicted"]
        )

        # Extract trading signal from the last min in the df
        trading_execute_signal = predictions_df.iloc[-1,0]
        trading_execute_signal
        if trading_execute_signal >0:
            side = 'buy'
            side_reverse = 'sell'
        else: 
            side = 'sell'
            side_reverse = 'buy'

        print('side is',side)
        print('side_reverse is',side_reverse)
        
        """This is the Trading Bot Piece"""
        api.submit_order(symbol=stock,qty=1,side=side,type='market',time_in_force='gtc')
        time.sleep(60)
        api.submit_order(symbol=stock,qty=1,side=side_reverse,type='market',time_in_force='gtc')
