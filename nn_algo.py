"""This is the algorithm leveraging the Neural Network supervised learning method"""

# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from alpaca_trade_api import TimeFrame
import talib as ta
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from utils.helper import get_alpacas_info

# Set API information from alpacas
api = get_alpacas_info()[0]
alpaca_api_key = get_alpacas_info()[1]
alpaca_secret_key = get_alpacas_info()[2]

# define bunch of variables that can be easiy changed to fine turn the algo
stock = 'SPY' ### THIS IS WHERE WE NEED TO INPUT THE STOCK ###
shift_time = 1
start_date_historical = '2022-06-07'
end_date_historical = '2022-06-10'
periods_for_training_data = 800
sma_short_window =20
sma_long_window=50
RSI_time_period = 3
EMA_time_period = 50

# Define alpaca api object 
bars = api.get_bars("SPY", TimeFrame.Minute, start_date_historical, end_date_historical).df

bars["actual_returns"] = bars["close"].pct_change()

# Drop all NaN values from the DataFrame
bars = bars.dropna()

# use talib to calcualte SMA (Simple moving avera
bars['sma_short_window'] = ta.SMA(bars['close'],timeperiod=sma_short_window)
bars['sma_long_window'] = ta.SMA(bars['close'],timeperiod=sma_long_window)

bars['RSI'] = ta.RSI(bars['close'],timeperiod=RSI_time_period)
bars['EMA'] = ta.RSI(bars['close'],timeperiod=EMA_time_period)

bars = bars.dropna()

# Assign a copy of the technical variable columns to a new DataFrame called `X` and lag it.
X = bars[['sma_short_window', 'sma_long_window', 'RSI', 'EMA', ]].shift(shift_time).dropna().copy()

# Initialize the new `Signal` column
bars['signal'] = 0.0

# Generate signal to buy stock long
bars.loc[(bars['actual_returns'] >= 0), 'signal'] = 1

# Generate signal to sell stock short
bars.loc[(bars['actual_returns'] < 0), 'signal'] = -1

bars['contrarian_signal']= bars['signal']*(-1)

# Calculate the strategy returns and add them to the signals_df DataFrame
# add shift after signal is because the buying/selling will be one day delay after the reutrn
bars['Strategy_Returns'] = bars['actual_returns'] * bars['signal'].shift()
bars['contrarian_Strategy_Returns'] = bars['actual_returns'] * bars['contrarian_signal'].shift()

# Copy the new "signal" column to a new Series called `y`.
y = bars['signal']

# Import the neccessary Date function
from pandas.tseries.offsets import DateOffset

# Use the following code to select the start of the training period: `training_begin = X.index.min()`
training_begin = X.index.min()
print(training_begin)

# Use the following code to select the ending period for the training data: `training_end = X.index.min() + DateOffset(months=3)`
training_end = X.index.min() + DateOffset(minutes=periods_for_training_data)
print(training_end)

# Generate the X_train and y_train DataFrames using loc to select the rows from `training_begin` up to `training_end`
X_train = X.loc[training_begin:training_end]
y_train = y.loc[training_begin:training_end]

# Generate the X_test and y_test DataFrames using loc to select from `training_end` to the last row in the DataFrame.
X_test = X.loc[training_end:]
y_test = y.loc[training_end:]

# Use StandardScaler to scale the X_train and X_test data.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaler = scaler.fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
X_test_scaled.shape

from imblearn.over_sampling import RandomOverSampler

# Use RandomOverSampler to resample the datase using random_state=1
ros = RandomOverSampler(random_state=1)
X_resampled, y_resampled = ros.fit_resample(X_train_scaled, y_train)

# Define the the number of inputs
number_input_features = 4

# Review the number of features
number_input_features

# Define the number of neurons in the output layer
number_output_neurons = 1

# Define the number of hidden nodes for the first hidden layer
hidden_nodes_layer1 =  8

# Review the number hidden nodes in the first layer
hidden_nodes_layer1

nn = Sequential()

# First hidden layer
nn.add(Dense(units=hidden_nodes_layer1,input_dim = number_input_features,activation='relu'))

# Output layer
nn.add(Dense(units=number_output_neurons, activation='sigmoid'))

# Check the structure of the model
nn.summary()

# Compile the model
nn.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# Fit the model
# X_resampled, y_resampled
fit_model = nn.fit(X_train_scaled,y_train,epochs=100)

# Evaluate the model loss and accuracy metrics using the evaluate method and the test data
model_loss, model_accuracy = nn.evaluate(X_test_scaled, y_test, verbose=2)

# Display the model loss and accuracy results
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

# Use the trained model to predict the trading signals for the training data.
training_signal_predictions = nn.predict(X_test_scaled)

contrarian_training_signal_predictions = training_signal_predictions * (-1)

# Create a predictions DataFrame
predictions_df = pd.DataFrame(index=X_test.index)

# Add the SVM model predictions to the DataFrame
predictions_df['Predicted'] = training_signal_predictions

predictions_df['Contrarian_Predicted'] = contrarian_training_signal_predictions


# Add the actual returns to the DataFrame
predictions_df['actual_returns'] = bars['actual_returns'] 

# Add the strategy returns to the DataFrame
predictions_df['Strategy_Returns'] = (
    predictions_df["actual_returns"] * predictions_df["Predicted"]
)

predictions_df['Strategy_Returns'] = (
    predictions_df["actual_returns"] * predictions_df["Predicted"]
)

predictions_df['Contrarian_Strategy_Returns'] = (
    predictions_df["actual_returns"] * predictions_df["Contrarian_Predicted"]
)