""" Helper functions  """

# Import required libraries
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
import os

# Function to get Alpacas trading account info
def get_alpacas_info():
    load_dotenv('api.env')
    alpaca_api_key = os.getenv('APCA_API_KEY_ID')
    alpaca_secret_key = os.getenv('APCA_API_SECRET_KEY')
    base_url = 'https://paper-api.alpaca.markets'

    # Create alpacas API object
    api = tradeapi.REST(
        alpaca_api_key,
        alpaca_secret_key,
        base_url,
        api_version='v2'
    )

    return api, alpaca_api_key, alpaca_secret_key
