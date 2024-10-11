import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
import altair as alt
import mplfinance as mpf
import plotly.graph_objects as go
import yfinance as yf
from scipy.stats import norm
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Import Regression Models
from sklearn.linear_model import Lasso, Ridge, ElasticNet
import datetime


def predict_action(data, model):
    features = ['Volume', 'MA_10', 'MA_20', 'MA_50', 'MA_200', 'std_10',
                'std_20', 'std_50', 'std_200', 'upper_band_10', 'lower_band_10',
                'upper_band_20', 'lower_band_20', 'upper_band_50', 'lower_band_50',
                'upper_band_200', 'lower_band_200', 'Golden_Cross_Short', 'Golden_Cross_Medium',
                'Golden_Cross_Long', 'Death_Cross_Short', 'Death_Cross_Medium', 'Death_Cross_Long',
                'ROC', 'AVG_Volume_10', 'AVG_Volume_20', 'AVG_Volume_50', 'AVG_Volume_200', 'Doji',
                'Bullish_Engulfing', 'Bearish_Engulfing', 'MACD', 'Signal', 'MACD_Hist', 'TR', 'ATR',
                'RSI_10_Day', '10_Day_ROC', 'Resistance_10_Day', 'Support_10_Day', 'Resistance_20_Day',
                'Support_20_Day', 'Resistance_50_Day', 'Support_50_Day', 'Volume_MA_10', 'Volume_MA_20',
                'Volume_MA_50', 'OBV', 'Z-score']

    if type(model) == xgb.core.Booster:

        data_df = pd.DataFrame([data])

        # Select only the features used in training
        data_df = data_df[features]

        # Convert the DataFrame to DMatrix (required by XGBoost)
        dmatrix = xgb.DMatrix(data_df)

        # Predict using the loaded model
        prediction = model.predict(dmatrix)[0]
    else:
        data_df = pd.DataFrame([data])
        data_df = data_df[features]
        prediction = model.predict(data_df)[0]

    if prediction == 0:
        return 'Buy'
    elif prediction == 1:
        return 'Sell'
    else:
        return 'Hold'


def stock_market_simulation(model, initial_cash, days, stock, existing_shares=0, day=None):
    # Add Taxes and Fees
    cash = initial_cash
    invested = cash
    shares_held = existing_shares
    portfolio_value = []
    scaled = scale_data(stock)
    modelDecisionDf = pd.DataFrame(
        columns=['Stock Name', 'Day', 'Action', 'Cash', 'Shares Held', 'Portfolio Value','Stock Price'])
    
    days = min(days, len(stock))

    for i in range(days):
        stock_price = stock['Close'].iloc[i]
        strategy = predict_action(scaled.iloc[i].to_dict(), model)
        if day:
            day = day
        else:
            day = i
       
        
        if strategy == 'Buy' and cash >= stock_price:
            # Buy one share if cash is sufficient
            cash -= stock_price
            shares_held += 1
            print(f"Day {day}: Bought 1 share at {stock_price}, Cash left: {cash}")

        elif strategy == 'Buy' and cash < stock_price:
            shares_held += cash / stock_price
            cash = 0
            print(
                f"Day {day}: Bought {cash / stock_price} shares at {stock_price}, Cash left: {cash}")

        elif strategy == 'Sell' and shares_held > 0:
            # Sell one share if we hold any
            cash += stock_price
            shares_held -= 1
            print(f"Day {day}: Sold 1 share at {stock_price}, Cash: {cash}")

        # elif cash == 0:
        #     injection = input(
        #         f'Your current portfolio value is {portfolio_value[i-1]}. The stock price is {stock_price}. Enter the amount of cash you want to inject: ')
        #     if injection:
        #         cash += float(injection)
        #         invested += float(injection)
        #         print(f"Day {i}: Cash injected: {injection}, Cash: {cash}")
        #         if cash < stock_price:
        #             break
        
        # Calculate the total portfolio value (cash + stock holdings)
        portfolio_value_at_time = cash + (shares_held * stock_price)
        portfolio_value.append(portfolio_value_at_time)
        stock_name = stock['Symbol'].iloc[0]
        new_row = pd.DataFrame({
            'Stock Name': [stock_name],
            'Day': [day],
            'Date': [stock['Date'].iloc[i]],
            'Action': [strategy],
            'Stock Price': [stock_price],
            'Cash': [cash],
            'Shares Held': [shares_held],
            'Portfolio Value': [portfolio_value_at_time],
            'Stock Price': [stock_price]
        })
        modelDecisionDf = pd.concat(
            [modelDecisionDf, new_row], ignore_index=True)
    if existing_shares > 0:
        d = 1
    # Final results
    final_portfolio_value = cash + (shares_held * stock['Close'].iloc[-1])
    print(f'Total cash invested: {invested}')
    print(f'Stock {stock["Symbol"].iloc[0]}')
    print(f"Final Portfolio Value: {final_portfolio_value}")
    print(f"Cash: {cash}, Shares held: {shares_held}")
    return modelDecisionDf, cash
    # return portfolio_value, final_portfolio_value


def determine_action(row):
    if row['Close'] != 0 and (((row['Close'] - row['close_lag1'])/(row['Close'])*100) > 1):
        return 2
    elif (row['Golden_Cross_Short'] == 1 or
          row['MACD'] > row['Signal'] or
          50 < row['RSI_10_Day'] < 70):
        return 0  # Buy
    elif (row['Death_Cross_Short'] == 1 or
          row['MACD'] < row['Signal'] or
          row['RSI_10_Day'] > 80 and
          row['Daily_Return'] < -0.01):
        return 1  # Sell
    else:
        return 2  # Hold


def calculate_rsi(stock_df, window=10):
    # Calculate daily price changes
    delta = stock_df['Close'].diff()

    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate the average gain and average loss
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    # Calculate the Relative Strength (RS)
    rs = avg_gain / avg_loss

    # Calculate the RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi


def is_doji(row):
    return abs(row['Close'] - row['Open']) <= (row['High'] - row['Low']) * 0.1


def is_bullish_engulfing(current_row, previous_row):
    # Example logic for identifying a bullish engulfing pattern
    if previous_row['Close'] < previous_row['Open'] and current_row['Close'] > current_row['Open'] and current_row['Close'] > previous_row['Open'] and current_row['Open'] < previous_row['Close']:
        return True
    return False


def is_bearish_engulfing(current_row, previous_row):
    # Example logic for identifying a bearish engulfing pattern
    if previous_row['Close'] > previous_row['Open'] and current_row['Close'] < current_row['Open'] and current_row['Close'] < previous_row['Open'] and current_row['Open'] > previous_row['Close']:
        return True
    return False


def add_columns(stock_df):
    # Create new columns with Returns
    stock_df['1_Day_Return'] = (
        stock_df['Close'] - stock_df['Close'].shift(1)) / stock_df['Close'].shift(1) * 100
    stock_df['5_Day_Return'] = (
        stock_df['Close'] - stock_df['Close'].shift(5)) / stock_df['Close'].shift(5) * 100
    stock_df['10_Day_Return'] = (
        stock_df['Close'] - stock_df['Close'].shift(10)) / stock_df['Close'].shift(10) * 100
    stock_df['20_Day_Return'] = (
        stock_df['Close'] - stock_df['Close'].shift(20)) / stock_df['Close'].shift(20) * 100
    stock_df['50_Day_Return'] = (
        stock_df['Close'] - stock_df['Close'].shift(50)) / stock_df['Close'].shift(50) * 100
    stock_df['200_Day_Return'] = (
        stock_df['Close'] - stock_df['Close'].shift(200)) / stock_df['Close'].shift(200) * 100

    stock_df['Best_Return_Window'] = stock_df[['1_Day_Return', '5_Day_Return',
                                               '10_Day_Return', '20_Day_Return', '50_Day_Return', '200_Day_Return']].idxmax(axis=1)
    stock_df['Best_Return'] = stock_df[['1_Day_Return', '5_Day_Return',
                                        '10_Day_Return', '20_Day_Return', '50_Day_Return', '200_Day_Return']].max(axis=1)
    stock_df['Best_Return_Window'] = stock_df['Best_Return_Window'].replace(
        '_Day_Return', '', regex=True)

    # Create lag columns
    stock_df['close_lag1'] = stock_df['Close'].shift(1)
    stock_df['close_lag2'] = stock_df['Close'].shift(2)
    stock_df['close_lag3'] = stock_df['Close'].shift(3)
    stock_df['close_lag4'] = stock_df['Close'].shift(5)
    stock_df['close_lag5'] = stock_df['Close'].shift(10)

    stock_df['volume_lag1'] = stock_df['Volume'].shift(1)
    stock_df['volume_lag2'] = stock_df['Volume'].shift(2)
    stock_df['volume_lag3'] = stock_df['Volume'].shift(3)
    stock_df['volume_lag4'] = stock_df['Volume'].shift(5)
    stock_df['volume_lag5'] = stock_df['Volume'].shift(10)

    # Create new columns with Moving Averages and Standard Deviations
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])

    stock_df['MA_10'] = stock_df.groupby('Symbol')['Close'].rolling(
        window=10).mean().reset_index(level=0, drop=True)
    stock_df['MA_20'] = stock_df.groupby('Symbol')['Close'].rolling(
        window=20).mean().reset_index(level=0, drop=True)
    stock_df['MA_50'] = stock_df.groupby('Symbol')['Close'].rolling(
        window=50).mean().reset_index(level=0, drop=True)
    stock_df['MA_200'] = stock_df.groupby('Symbol')['Close'].rolling(
        window=200).mean().reset_index(level=0, drop=True)

    stock_df['std_10'] = stock_df.groupby('Symbol')['Close'].rolling(
        window=10).std().reset_index(level=0, drop=True)
    stock_df['std_20'] = stock_df.groupby('Symbol')['Close'].rolling(
        window=20).std().reset_index(level=0, drop=True)
    stock_df['std_50'] = stock_df.groupby('Symbol')['Close'].rolling(
        window=50).std().reset_index(level=0, drop=True)
    stock_df['std_200'] = stock_df.groupby('Symbol')['Close'].rolling(
        window=200).std().reset_index(level=0, drop=True)

    # Create new columns with Bollinger Bands for each Moving Average
    stock_df['upper_band_10'] = stock_df['MA_10'] + (stock_df['std_10'] * 2)
    stock_df['lower_band_10'] = stock_df['MA_10'] - (stock_df['std_10'] * 2)

    stock_df['upper_band_20'] = stock_df['MA_20'] + (stock_df['std_20'] * 2)
    stock_df['lower_band_20'] = stock_df['MA_20'] - (stock_df['std_20'] * 2)

    stock_df['upper_band_50'] = stock_df['MA_50'] + (stock_df['std_50'] * 2)
    stock_df['lower_band_50'] = stock_df['MA_50'] - (stock_df['std_50'] * 2)

    stock_df['upper_band_200'] = stock_df['MA_200'] + (stock_df['std_200'] * 2)
    stock_df['lower_band_200'] = stock_df['MA_200'] - (stock_df['std_200'] * 2)

    # Create new columns Indicating Golden Cross and Death Cross
    stock_df['Golden_Cross_Short'] = np.where((stock_df['MA_10'] > stock_df['MA_20']) & (
        stock_df['MA_10'].shift(1) <= stock_df['MA_20'].shift(1)), 1, 0)
    stock_df['Golden_Cross_Medium'] = np.where((stock_df['MA_20'] > stock_df['MA_50']) & (
        stock_df['MA_20'].shift(1) <= stock_df['MA_50'].shift(1)), 1, 0)
    stock_df['Golden_Cross_Long'] = np.where((stock_df['MA_50'] > stock_df['MA_200']) & (
        stock_df['MA_50'].shift(1) <= stock_df['MA_200'].shift(1)), 1, 0)

    stock_df['Death_Cross_Short'] = np.where((stock_df['MA_10'] < stock_df['MA_20']) & (
        stock_df['MA_10'].shift(1) >= stock_df['MA_20'].shift(1)), 1, 0)
    stock_df['Death_Cross_Medium'] = np.where((stock_df['MA_20'] < stock_df['MA_50']) & (
        stock_df['MA_20'].shift(1) >= stock_df['MA_50'].shift(1)), 1, 0)
    stock_df['Death_Cross_Long'] = np.where((stock_df['MA_50'] < stock_df['MA_200']) & (
        stock_df['MA_50'].shift(1) >= stock_df['MA_200'].shift(1)), 1, 0)

    # Create new columns with Rate of Change and Average Volume

    stock_df['ROC'] = (
        (stock_df['Close'] - stock_df['Close'].shift(1)) / stock_df['Close'].shift(1)) * 100

    stock_df['AVG_Volume_10'] = stock_df.groupby('Symbol')['Volume'].rolling(
        window=10).mean().reset_index(level=0, drop=True)
    stock_df['AVG_Volume_20'] = stock_df.groupby('Symbol')['Volume'].rolling(
        window=20).mean().reset_index(level=0, drop=True)
    stock_df['AVG_Volume_50'] = stock_df.groupby('Symbol')['Volume'].rolling(
        window=50).mean().reset_index(level=0, drop=True)
    stock_df['AVG_Volume_200'] = stock_df.groupby('Symbol')['Volume'].rolling(
        window=200).mean().reset_index(level=0, drop=True)

    # Doji Candlestick Pattern, identified by a small body and long wicks
    stock_df['Doji'] = stock_df.apply(is_doji, axis=1)

    # Bullish and Bearish Engulfing Candlestick Patterns, identified by a large body that engulfs the previous candle
    stock_df['Bullish_Engulfing'] = stock_df.apply(
        lambda row: is_bullish_engulfing(row, stock_df.shift(1).loc[row.name]), axis=1)
    stock_df['Bearish_Engulfing'] = stock_df.apply(
        lambda row: is_bearish_engulfing(row, stock_df.shift(1).loc[row.name]), axis=1)

    stock_df['EMA_short'] = stock_df['Close'].ewm(span=12, adjust=False).mean()

    # Calculate the long-term EMA
    stock_df['EMA_long'] = stock_df['Close'].ewm(span=26, adjust=False).mean()

    # Calculate the MACD line
    stock_df['MACD'] = stock_df['EMA_short'] - stock_df['EMA_long']

    # Calculate the Signal line
    stock_df['Signal'] = stock_df['MACD'].ewm(span=9, adjust=False).mean()

    # Calculate the MACD histogram
    stock_df['MACD_Hist'] = stock_df['MACD'] - stock_df['Signal']

    # Create new columns for Average True Range (ATR) and True Range (TR)

    stock_df['Previous_Close'] = stock_df['Close'].shift(1)

    # True Range, Shows the volatility of the stock
    stock_df['TR'] = stock_df.apply(
        lambda row: max(
            row['High'] - row['Low'],  # High - Low
            # |High - Previous Close|
            abs(row['High'] - row['Previous_Close']),
            abs(row['Low'] - row['Previous_Close'])  # |Low - Previous Close|
        ), axis=1
    )

    # Average True Range, Shows the average volatility of the stock
    stock_df['ATR'] = stock_df['TR'].rolling(window=10).mean()

    # Create new columns for Relative Strength Index (RSI) and Rate of Change (ROC)
    stock_df['RSI_10_Day'] = calculate_rsi(stock_df)
    stock_df['10_Day_ROC'] = (
        (stock_df['Close'] - stock_df['Close'].shift(10)) / stock_df['Close'].shift(10)) * 100
    stock_df['20_Day_ROC'] = (
        (stock_df['Close'] - stock_df['Close'].shift(20)) / stock_df['Close'].shift(20)) * 100
    stock_df['50_Day_ROC'] = (
        (stock_df['Close'] - stock_df['Close'].shift(50)) / stock_df['Close'].shift(50)) * 100

    # Create new columns for 10,20,50 day resistance and support levels
    stock_df['Resistance_10_Day'] = stock_df['Close'].rolling(window=10).max()
    stock_df['Support_10_Day'] = stock_df['Close'].rolling(window=10).min()
    stock_df['Resistance_20_Day'] = stock_df['Close'].rolling(window=20).max()
    stock_df['Support_20_Day'] = stock_df['Close'].rolling(window=20).min()
    stock_df['Resistance_50_Day'] = stock_df['Close'].rolling(window=50).max()
    stock_df['Support_50_Day'] = stock_df['Close'].rolling(window=50).min()

    # Create new columns for 10,20,50 day Volume Indicators
    stock_df['Volume_MA_10'] = stock_df['Volume'].rolling(window=10).mean()
    stock_df['Volume_MA_20'] = stock_df['Volume'].rolling(window=20).mean()
    stock_df['Volume_MA_50'] = stock_df['Volume'].rolling(window=50).mean()
    # Use a smoothed version of 'Close' to detect peaks and troughs
    stock_df['Smoothed_Close'] = stock_df['Close'].rolling(window=20).mean()

    # Find local minima (buy points) and local maxima (sell points)
    # Local minima (buy points)
    stock_df['Buy_Signal'] = (stock_df['Smoothed_Close'].shift(1) > stock_df['Smoothed_Close']) & (stock_df['Smoothed_Close'].shift(-1) > stock_df['Smoothed_Close'])

    # Local maxima (sell points)
    stock_df['Sell_Signal'] = (stock_df['Smoothed_Close'].shift(1) < stock_df['Smoothed_Close']) & (stock_df['Smoothed_Close'].shift(-1) < stock_df['Smoothed_Close'])

    # Initialize 'Optimal_Action' column with 'Hold'
    stock_df['Optimal_Action'] = 'Hold'

    # Assign 'Buy' where Buy_Signal is True
    stock_df.loc[stock_df['Buy_Signal'], 'Optimal_Action'] = 'Buy'

    # Assign 'Sell' where Sell_Signal is True
    stock_df.loc[stock_df['Sell_Signal'], 'Optimal_Action'] = 'Sell'

    # Clean up: drop the temporary signals if needed
    stock_df.drop(['Buy_Signal', 'Sell_Signal', 'Smoothed_Close'], axis=1, inplace=True)

    stock_df['OBV'] = 0
    for i in range(1, len(stock_df)):
        if stock_df['Close'].iloc[i] > stock_df['Close'].iloc[i - 1]:
            stock_df.loc[stock_df.index[i],
                         'OBV'] = stock_df['OBV'].iloc[i - 1] + stock_df['Volume'].iloc[i]
        elif stock_df['Close'].iloc[i] < stock_df['Close'].iloc[i - 1]:
            stock_df.loc[stock_df.index[i],
                         'OBV'] = stock_df['OBV'].iloc[i - 1] - stock_df['Volume'].iloc[i]
        else:
            stock_df.loc[stock_df.index[i],
                         'OBV'] = stock_df['OBV'].iloc[i - 1]

    stock_df['Action'] = stock_df.apply(determine_action, axis=1)
    stock_df['Z-score'] = (stock_df['Close'] -
                           stock_df['Close'].mean()) / stock_df['Close'].std()

    stock_df = stock_df.fillna(0)

    return stock_df


def scale_data(stock_df):
    features = ['Volume', 'MA_10', 'MA_20', 'MA_50', 'MA_200', 'std_10',
                'std_20', 'std_50', 'std_200', 'upper_band_10', 'lower_band_10',
                'upper_band_20', 'lower_band_20', 'upper_band_50', 'lower_band_50',
                'upper_band_200', 'lower_band_200', 'Golden_Cross_Short', 'Golden_Cross_Medium',
                'Golden_Cross_Long', 'Death_Cross_Short', 'Death_Cross_Medium', 'Death_Cross_Long',
                'ROC', 'AVG_Volume_10', 'AVG_Volume_20', 'AVG_Volume_50', 'AVG_Volume_200', 'Doji',
                'Bullish_Engulfing', 'Bearish_Engulfing', 'MACD', 'Signal', 'MACD_Hist', 'TR', 'ATR',
                'RSI_10_Day', '10_Day_ROC', 'Resistance_10_Day', 'Support_10_Day', 'Resistance_20_Day',
                'Support_20_Day', 'Resistance_50_Day', 'Support_50_Day', 'Volume_MA_10', 'Volume_MA_20',
                'Volume_MA_50', 'OBV', 'Z-score']
    min_max_scaler = MinMaxScaler()
    stock_df = add_columns(stock_df)
    stock_df[features] = min_max_scaler.fit_transform(stock_df[features])
    return stock_df[features]


def _select_stock():
    stock_df = pd.read_csv('data/sp500_stocks.csv')
    company_name = input('Enter the name of the company: ')
    return stock_df[stock_df['Symbol'] == company_name]


def get_stock_data(symbol):
    stock_df = pd.read_csv('data/sp500_stocks.csv')
    stock_df = stock_df[stock_df['Symbol'] == symbol]
    stock = yf.Ticker(symbol)
    data = stock.history(period='5d', interval='1d')

    if not data.empty:
        latest_data = data.iloc[-1]
        time = latest_data.name
        open_price = latest_data['Open']
        high = latest_data['High']
        low = latest_data['Low']
        close = latest_data['Close']
        volume = latest_data['Volume']
        new_row = pd.DataFrame({
            'Symbol': [symbol],
            'Date': [datetime.datetime.strftime(time, '%Y-%m-%d')],
            'Open': [open_price],
            'High': [high],
            'Low': [low],
            'Close': [close],
            'Volume': [volume]
        })
        stock_df = pd.concat([stock_df, new_row], ignore_index=True).fillna(0)
        return stock_df


def train_models():
    company_df = pd.read_csv('data/sp500_companies.csv')
    features = ['Volume', 'MA_10', 'MA_20', 'MA_50', 'MA_200', 'std_10',
                'std_20', 'std_50', 'std_200', 'upper_band_10', 'lower_band_10',
                'upper_band_20', 'lower_band_20', 'upper_band_50', 'lower_band_50',
                'upper_band_200', 'lower_band_200', 'Golden_Cross_Short', 'Golden_Cross_Medium',
                'Golden_Cross_Long', 'Death_Cross_Short', 'Death_Cross_Medium', 'Death_Cross_Long',
                'ROC', 'AVG_Volume_10', 'AVG_Volume_20', 'AVG_Volume_50', 'AVG_Volume_200', 'Doji',
                'Bullish_Engulfing', 'Bearish_Engulfing', 'MACD', 'Signal', 'MACD_Hist', 'TR', 'ATR',
                'RSI_10_Day', '10_Day_ROC', 'Resistance_10_Day', 'Support_10_Day', 'Resistance_20_Day',
                'Support_20_Day', 'Resistance_50_Day', 'Support_50_Day', 'Volume_MA_10', 'Volume_MA_20',
                'Volume_MA_50', 'OBV', 'Z-score']
    for symbol in company_df['Symbol'].unique():
        print('Loading data for', symbol, '...')
        stock_df = pd.read_csv('data/sp500_stocks.csv')
        stock_df = stock_df[stock_df['Symbol'] == symbol]
        print(f'Adding columns for {symbol}...')
        stock_df = add_columns(stock_df)
        preprocessed = scale_data(stock_df)
        X = preprocessed[features]
        y = stock_df['Action']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        print(f'Training model for {symbol}...')
        model = xgb.XGBClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        print(f'Saving model for {symbol}...')
        joblib.dump(model, f'models/{symbol}_model.pkl')


def train_model_incrementally():
    features = ['Volume', 'MA_10', 'MA_20', 'MA_50', 'MA_200', 'std_10',
                'std_20', 'std_50', 'std_200', 'upper_band_10', 'lower_band_10',
                'upper_band_20', 'lower_band_20', 'upper_band_50', 'lower_band_50',
                'upper_band_200', 'lower_band_200', 'Golden_Cross_Short', 'Golden_Cross_Medium',
                'Golden_Cross_Long', 'Death_Cross_Short', 'Death_Cross_Medium', 'Death_Cross_Long',
                'ROC', 'AVG_Volume_10', 'AVG_Volume_20', 'AVG_Volume_50', 'AVG_Volume_200', 'Doji',
                'Bullish_Engulfing', 'Bearish_Engulfing', 'MACD', 'Signal', 'MACD_Hist', 'TR', 'ATR',
                'RSI_10_Day', '10_Day_ROC', 'Resistance_10_Day', 'Support_10_Day', 'Resistance_20_Day',
                'Support_20_Day', 'Resistance_50_Day', 'Support_50_Day', 'Volume_MA_10', 'Volume_MA_20',
                'Volume_MA_50', 'OBV', 'Z-score']

    stock_df = pd.read_csv('data/sp500_stocks.csv')
    print("Data Loaded")

    # Initialize an empty DMatrix for incremental training
    initial_model = None
    num_round = 100  # Number of boosting rounds

    for symbol in stock_df['Symbol'].unique():
        print(f"Processing {symbol}...")

        # Filter the data for the current stock
        stock_data = stock_df[stock_df['Symbol'] == symbol].copy()

        print(f"Adding columns for {symbol}...")
        stock_data = add_columns(stock_data)
        stock_data.to_csv('data/sp500_stocks_columns.csv', index=False)

        print(f"Preprocessing data for {symbol}...")
        preprocessed = scale_data(stock_data)

        print(f"Splitting data for {symbol}...")
        X = preprocessed[features]
        y = stock_data['Action']

        # Split into training and testing data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # If it's the first iteration, initialize the model
        if initial_model is None:
            print(f"Initializing model for {symbol}...")
            # Number of unique actions (e.g., Buy, Sell, Hold)
            unique_classes = stock_data['Action'].nunique()

            # Initial model training
            initial_model = xgb.train(params={
                'objective': 'multi:softmax',  # Use 'multi:softprob' for probability outputs
                'eval_metric': 'mlogloss',
                'num_class': unique_classes},  # Specify the number of classes
                dtrain=dtrain,
                num_boost_round=num_round,
                evals=[(dtest, 'eval')],
                early_stopping_rounds=10)
        else:
            print(f"Continuing training on {symbol} data...")

            # Continue training with the same parameters
            initial_model = xgb.train(params={
                'objective': 'multi:softmax',  # Same objective as in the initial training
                'eval_metric': 'mlogloss',
                'num_class': unique_classes},  # Specify the number of classes
                dtrain=dtrain,
                num_boost_round=num_round,
                evals=[(dtest, 'eval')],
                xgb_model=initial_model,  # Continue training from the previous model
                early_stopping_rounds=10)

    # Save the final model
    print("Saving the final model...")
    initial_model.save_model('models/all_stocks_incremental_model.model')
    print("Model trained on all stocks and saved.")


def simulate_day(stock_df):
    # # Load the general model
    general_model = xgb.Booster()
    general_model.load_model('models/all_stocks_incremental_model.pkl') 

    # Load company data
    company_df = pd.read_csv('data/sp500_companies.csv')

    # Load the previous decision dataframes
    specific_decision_df = pd.read_csv('data/specific_model_decisions.csv')
    general_decision_df = pd.read_csv('data/general_model_decisions.csv')

    # Initialize empty dataframes for storing new decisions
    all_decisions_s = pd.DataFrame(columns=[
        'Stock Name', 'Day', 'Action', 'Stock Price', 'Cash', 'Shares Held', 'Portfolio Value'])

    all_decisions_g = pd.DataFrame(columns=[
        'Stock Name', 'Day', 'Action', 'Stock Price', 'Cash', 'Shares Held', 'Portfolio Value'])

    # Loop through each stock symbol
    for symbol in company_df['Symbol'].unique():
        # Get the most recent cash and shares held for the specific model
        if symbol in specific_decision_df['Stock Name'].unique():
            last_row_s = specific_decision_df[specific_decision_df['Stock Name'] == symbol].iloc[-1]
        else:
            last_row_s = {'Cash': 10000, 'Shares Held': 0}  # Initialize if no previous data

        # # Get the most recent cash and shares held for the general model
        if symbol in general_decision_df['Stock Name'].unique():
            last_row_g = general_decision_df[general_decision_df['Stock Name'] == symbol].iloc[-1]
        else:
            last_row_g = {'Cash': 10000, 'Shares Held': 0}  # Initialize if no previous data

        # Set the starting cash and shares for the current simulation
        cash_s = last_row_s['Cash']
        cash_g = last_row_g['Cash']

        # Get the stock data for the symbol
        updated_stock_df = get_stock_data(symbol)
        updated_stock_df = updated_stock_df.tail(1)

        # Load the specific model for the stock, or fallback to the general model if it doesn't exist
        try:
            specific_model = joblib.load(f'models/{symbol}_model.pkl')
            print(f"Using model for {symbol}")
        except:
            specific_model = general_model
            print(f"Using general model for {symbol}")


        # Simulate a day of trading for the stock with the specific model
        new_decisions_s, _ = stock_market_simulation(
            model=specific_model,
            initial_cash=cash_s,
            days=1,  # Simulate only one day
            stock=updated_stock_df,
            stock_name=symbol
        )
        print('Specific Model has been simulated. Now simulating General Model...')
        # Simulate a day of trading for the stock with the general model
        new_decisions_g, _ = stock_market_simulation(
            model=general_model,
            initial_cash=cash_g,
            days=1,  # Simulate only one day
            stock=updated_stock_df,
            stock_name=symbol
        )

        # Append the new decisions to the all_decisions dataframes
        all_decisions_s = pd.concat([all_decisions_s, new_decisions_s], ignore_index=True)
        all_decisions_g = pd.concat([all_decisions_g, new_decisions_g], ignore_index=True)

    # Save the new decisions
    all_decisions_s.to_csv('data/specific_model_decisions.csv', mode='a', header=False, index=False)
    all_decisions_g.to_csv('data/general_model_decisions.csv', mode='a', header=False, index=False)


def simulate_day_general():
    # Load the general model
    general_model = xgb.Booster()
    general_model.load_model('models/all_stocks_incremental_model.pkl') 

    # Load company data
    company_df = pd.read_csv('data/sp500_companies.csv')

    general_decision_df = pd.read_csv('data/general_model_decisions.csv')

    all_decisions_g = pd.DataFrame(columns=[
        'Stock Name', 'Day', 'Action', 'Stock Price', 'Cash', 'Shares Held', 'Portfolio Value'])
    
    for symbol in company_df['Symbol'].unique():

         # Get the most recent cash and shares held for the general model
        if symbol in general_decision_df['Stock Name'].unique():
            last_row_g = general_decision_df[general_decision_df['Stock Name'] == symbol].iloc[-1]
        else:
            last_row_g = {'Cash': 10000, 'Shares Held': 0, 'Day': 0}  # Initialize if no previous data

        cash_g = last_row_g['Cash']
        day = last_row_g['Day'] + 1
        existing_shares = last_row_g['Shares Held']


        # Get the stock data for the symbol
        updated_stock_df = get_stock_data(symbol)
        updated_stock_df = updated_stock_df.tail(1)

        print(f"Using general model for {symbol}")

        new_decisions_g, _ = stock_market_simulation(
            model=general_model,
            initial_cash=cash_g,
            days=1,  # Simulate only one day
            stock=updated_stock_df,
            day=day,
            existing_shares=existing_shares,
        )

        all_decisions_g = pd.concat([all_decisions_g, new_decisions_g], ignore_index=True)

    all_decisions_g.to_csv('data/general_model_decisions.csv', mode='a', header=False, index=False)


def simulate_day_specific():
    
    # Load company data
    company_df = pd.read_csv('data/sp500_companies.csv')

    # Load the previous decision dataframes
    specific_decision_df = pd.read_csv('data/specific_model_decisions.csv')

    # Initialize empty dataframes for storing new decisions
    all_decisions_s = pd.DataFrame(columns=[
        'Stock Name', 'Day', 'Action', 'Stock Price', 'Cash', 'Shares Held', 'Portfolio Value'])

    # Loop through each stock symbol
    for symbol in company_df['Symbol'].unique():
        # Get the most recent cash and shares held for the specific model
        if symbol in specific_decision_df['Stock Name'].unique():
            last_row_s = specific_decision_df[specific_decision_df['Stock Name'] == symbol].iloc[-1]
        else:
            last_row_s = {'Cash': 10000, 'Shares Held': 0, 'Day': 0}  # Initialize if no previous data

        # Set the starting cash and shares for the current simulation
        cash_s = last_row_s['Cash']
        day = last_row_s['Day'] + 1
        existing_shares = last_row_s['Shares Held']
        
        # Get the stock data for the symbol
        updated_stock_df = get_stock_data(symbol)
        updated_stock_df = updated_stock_df.tail(2)

        # Load the specific model for the stock, or fallback to the general model if it doesn't exist
        try:
            specific_model = joblib.load(f'models/{symbol}_model.pkl')
            print(f"Using model for {symbol}")
        except:
            general_model = xgb.Booster()
            general_model.load_model('models/all_stocks_incremental_model.pkl') 
            specific_model = general_model
            print(f"Using general model for {symbol}")
        
        # Simulate a day of trading for the stock with the specific model
        new_decisions_s, _ = stock_market_simulation(
            model=specific_model,
            initial_cash=cash_s,
            days=2,  # Simulate only one day
            stock=updated_stock_df,
            day=day,
            existing_shares=existing_shares
        )
        # Append the new decisions to the all_decisions dataframes
        all_decisions_s = pd.concat([all_decisions_s, new_decisions_s], ignore_index=True)
        if existing_shares == 1:
            continue

    # Save the new decisions
    all_decisions_s.to_csv('data/specific_model_decisions.csv', mode='a', header=False, index=False)

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    # Load your stock data (SP500 example)

    """
    FIX STOCK_SIMULATION TO PICK UP FROM PREVIOUS DAY
    CURRENTLY NOT GETTING PORTFOLIO VALUE CORRECT WHEN RUNNING ONE DAY 
    AT A TIME. 
    """

    # Simulate one day for all stocks, continuing from previous cash balances
    # simulate_day_general(stock_df)
    simulate_day_specific()

    # # Load the general decisions DataFrame
    # general_decisions = pd.read_csv('data/general_model_decisions.csv')
    # specific_decisions = pd.read_csv('data/specific_model_decisions.csv')
    # print('Model rows loaded. Now updating rows...')
    # # Iterate over the rows and update the 'Stock Price' column
    # for index, row in general_decisions.iterrows():
    #     stock = row['Stock Name']
    #     stock_price = get_stock_data(stock)
    #     general_decisions.at[index, 'Stock Price'] = stock_price['Close'].iloc[row['Day']]
    # print('General Model rows updated. Now updating Specific Model rows...')
    # for index, row in specific_decisions.iterrows():
    #     stock = row['Stock Name']
    #     stock_price = get_stock_data(stock)
    #     specific_decisions.at[index, 'Stock Price'] = stock_price['Close'].iloc[row['Day']]

    # # Save the updated DataFrame back to the CSV file
    # general_decisions.to_csv('data/general_model_decisions.csv', index=False)
    # specific_decisions.to_csv('data/specific_model_decisions.csv', index=False)