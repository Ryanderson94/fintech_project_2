"""This is the algorithm leveraging the Neural Network supervised learning method"""

# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from alpaca_trade_api import TimeFrame
import talib as ta
import time
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from utils.helper import get_alpacas_info

# Set API information from alpacas
api = get_alpacas_info()[0]
alpaca_api_key = get_alpacas_info()[1]
alpaca_secret_key = get_alpacas_info()[2]

# define bunch of variables that can be easiy changed to fine turn the algo
def define_hyper_parameters():
    stock = 'SPY' ### THIS IS WHERE WE NEED TO INPUT THE STOCK ###
    max_shares_to_trade = 10
    shift_time = 1
    start_date_historical = '2022-06-07'
    end_date_historical = '2022-06-10'
    periods_for_training_data = 800
    sma_short_window =20
    sma_long_window=50
    RSI_time_period = 3
    EMA_time_period = 50

    return stock, max_shares_to_trade, shift_time, start_date_historical, end_date_historical, periods_for_training_data, sma_short_window, sma_long_window, RSI_time_period, EMA_time_period

def run_trading_bot(
    stock, 
    max_shares_to_trade,
    shift_time, 
    start_date_historical, 
    end_date_historical, 
    periods_for_training_data, 
    sma_short_window, 
    sma_long_window, 
    RSI_time_period, 
    EMA_time_period
    ):
    
    # Define alpaca api object 
    bars = api.get_bars(stock, TimeFrame.Minute, start_date_historical, end_date_historical).df

    bars["actual_returns"] = bars["close"].pct_change()

    # Drop all NaN values from the DataFrame
    bars = bars.dropna()

    # use talib to calcualte SMA (Simple moving avera
    bars['sma_short_window'] = ta.SMA(bars['close'],timeperiod=sma_short_window)
    bars['sma_long_window'] = ta.SMA(bars['close'],timeperiod=sma_long_window)

    bars['RSI'] = ta.RSI(bars['close'],timeperiod=RSI_time_period)
    bars['EMA'] = ta.RSI(bars['close'],timeperiod=EMA_time_period)

    # Average Directional Movement Index(Momentum Indicator)， 
    # ADX can be used to help measure the overall strength of a trend. 
    # The ADX indicator is an average of expanding price range values.
    bars['ADX'] = ta.ADX(bars['high'],bars['low'], bars['close'], timeperiod=20)

    # Bollinger Bands are a type of statistical chart characterizing the prices and 
    # volatility over time of a financial instrument or commodity, using a formulaic method propounded by John Bollinger.
    bars['Bollinger_up_band'], bars['Bollinger_mid_band'], bars['Bollinger_low_band'] =   ta.BBANDS(bars['close'], timeperiod =20)

    bars = bars.dropna()

    # Assign a copy of the technical variable columns to a new DataFrame called `X` and lag it.
    X_time_series = bars[['sma_short_window', 'sma_long_window', 'RSI', 'EMA', 'volume','ADX','Bollinger_up_band','Bollinger_mid_band','Bollinger_low_band']].shift(shift_time).dropna().copy()

    # Initialize the new `Signal` column
    bars['signal'] = 0.0

    # Generate signal to buy stock long
    bars.loc[(bars['actual_returns'] >= 0), 'signal'] = 1
    # Generate signal to sell stock short
    bars.loc[(bars['actual_returns'] < 0), 'signal'] = 0

    bars.loc[(bars['signal'] == 1), 'signal_1_or_-1'] = 1
    # Generate signal to sell stock short
    bars.loc[(bars['signal'] == 0), 'signal_1_or_-1'] = -1

    bars['contrarian_signal']= bars['signal_1_or_-1']*(-1)

    # Calculate the strategy returns and add them to the signals_df DataFrame
    # add shift after signal is because the buying/selling will be one day delay after the reutrn
    bars['Strategy_Returns'] = bars['actual_returns'] * bars['signal_1_or_-1'].shift(shift_time)
    bars['contrarian_Strategy_Returns'] = bars['actual_returns'] * bars['contrarian_signal'].shift(shift_time)


    # Copy the new "signal" column to a new Series called `y`.
    y_time_series = bars['signal']

    # Import the neccessary Date function
    from pandas.tseries.offsets import DateOffset

    # Use the following code to select the start of the training period: `training_begin = X.index.min()`
    training_begin = X_time_series.index.min()

    # Use the following code to select the ending period for the training data: `training_end = X.index.min() + DateOffset(months=3)`
    training_end = X_time_series.index.min() + DateOffset(minutes=periods_for_training_data)

    # Generate the X_train and y_train DataFrames using loc to select the rows from `training_begin` up to `training_end`
    X_train_time_series = X_time_series[training_begin:training_end]
    y_train_time_series = y_time_series[training_begin:training_end]
    X_train = X_train_time_series.values

    # Generate the X_test and y_test DataFrames using loc to select from `training_end` to the last row in the DataFrame.
    X_test_time_series = X_time_series.loc[training_end:]
    y_test_time_series = y_time_series.loc[training_end:]
    X_test = X_test_time_series.values

    # Use StandardScaler to scale the X_train and X_test data.
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaler = scaler.fit(X_train)
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    X_test_scaled.shape

    # Define the the number of inputs
    number_input_features = 9

    # Define the number of neurons in the output layer
    number_output_neurons = 1

    # Define the number of hidden nodes for the first hidden layer
    hidden_nodes_layer1 =  10

    # Define the number of hidden nodes for the second hidden layer
    hidden_nodes_layer2 =  5

    # Define the number of hidden nodes for the third hidden layer
    hidden_nodes_layer3 =  10

    nn = Sequential()

    # First hidden layer
    nn.add(Dense(units=hidden_nodes_layer1,input_dim = number_input_features,activation='relu'))

    # Second hidden layer
    nn.add(Dense(units=hidden_nodes_layer2,activation='relu'))

    # Third hidden layer
    nn.add(Dense(units=hidden_nodes_layer3,activation='relu'))
 
    # Output layer
    nn.add(Dense(units=number_output_neurons, activation='sigmoid'))

    # Compile the model
    nn.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    # Use the trained model to predict the trading signals for the training data.
    training_signal_predictions = nn.predict(X_test_scaled)

    # Create a predictions DataFrame
    predictions_df = pd.DataFrame(index=X_test_time_series.index)

    # Add the SVM model predictions to the DataFrame
    predictions_df['Predicted'] = training_signal_predictions
    
    # Convert predicated prob (from 0 to 1) to trading signal 1 (buy) or -1 (sell)
    predictions_df.loc[(predictions_df['Predicted'] >= 0.5), 'Predicted_1_or_-1'] = 1
    predictions_df.loc[(predictions_df['Predicted'] < 0.5), 'Predicted_1_or_-1'] = -1

    # flip the signal to contrarian
    predictions_df['Contrarian_Predicted'] = -1* predictions_df['Predicted_1_or_-1']

    predictions_df['confidence_level_stock_going_up_or_down'] = abs((predictions_df['Predicted']-0.5)/0.5)

    predictions_df['shares_to_trade'] = round(predictions_df['confidence_level_stock_going_up_or_down'] * max_shares_to_trade)

    # Add the actual returns to the DataFrame
    predictions_df['close'] = bars['close'] 
    predictions_df['signal'] = bars['signal'] 
    predictions_df['actual_returns'] = bars['actual_returns'] 

    # Add the strategy returns to the DataFrame
    predictions_df['Strategy_Returns'] = (
        predictions_df["actual_returns"] * predictions_df["Predicted_1_or_-1"]*predictions_df['confidence_level_stock_going_up_or_down']
    )

    predictions_df['Contrarian_Strategy_Returns'] = (
        predictions_df["actual_returns"] * predictions_df["Contrarian_Predicted"]*predictions_df['confidence_level_stock_going_up_or_down']
    )

    # Next step is to calculate real time X (including 4 TA indicators) which is used to generate trading signal via the trained model
    # then hook the signal up to Alpaca Paper trading account
    trading_execute_signal = predictions_df.iloc[-1,1]
    print('trading_execute_signal is ',trading_execute_signal)
    shares_to_execute = int(predictions_df.iloc[-1,4])
    print('shares_to_execute is ',shares_to_execute)

    if trading_execute_signal >0:
        side = 'buy'
        side_reverse = 'sell'
    else: 
        side = 'sell'
        side_reverse = 'buy'

    api.submit_order(symbol=stock,qty=shares_to_execute,side=side,type='market',time_in_force='gtc')
    time.sleep(60)
    api.submit_order(symbol=stock,qty=shares_to_execute,side=side_reverse,type='market',time_in_force='gtc')