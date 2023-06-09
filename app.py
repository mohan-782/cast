import yfinance as yf
from prophet import Prophet
import streamlit as st
import datetime
import time
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import numpy as np

#Animation
import json
st.set_page_config(layout="wide")
base="dark"
font="serif"
import requests
from streamlit_lottie import st_lottie

#progress bar
progress = st.progress(0)
for i in range (100):
    time.sleep(0.01)
    progress.progress(i+1)

# Add a title
st.markdown("<h1 style='text-align: center; color: white;'>Stock Forecasting</h1>", unsafe_allow_html=True)
st.markdown("***")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
lottie_ani2 = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_7c1e8erd.json")
st_lottie(
    lottie_ani2,
    speed=1,
    reverse=False,
    quality="low",
    height=400,
    width=1400,
    key=None,
)

# Create a function to analyze the csv file
def analyze_csv_file(file_upload):
    # If a file is uploaded, read it into a Pandas DataFrame
    if file_upload is not None:
        df = pd.read_csv(file_upload)

        # Display the first few rows of the DataFrame
        st.dataframe(df.head())

        # Plot a line chart of the stock price
        st.line_chart(df, x="Close", y="High")
        #st.area_chart(forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']])
        fig = px.area(df, x="Close", y="High")
        st.plotly_chart(fig)

        # Plot a candlestick chart of the stock price
    from plotly.graph_objects import Candlestick
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

    fig.add_trace(
        Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']),
        row=1,
        col=1,
    )

    fig.update_layout(
        title='Candlestick Chart',
        xaxis_title='Close',
        yaxis_title='High',
        width=1350,
        height=700
    )

    st.plotly_chart(fig)

        # Display the summary statistics of the DataFrame
        #st.write(df.describe())

    # If no file is uploaded, display a message
    #else:
        #st.write("Please upload a CSV file of stock data.")
# Create a file uploader widget
file_upload = st.file_uploader("Upload a CSV file of stock data")

# If a file is uploaded, call the analyze_csv_file function
if file_upload is not None:
    analyze_csv_file(file_upload)

st.markdown("***")
# Add a ticker input function
ticker_input = st.text_input("Enter a stock ticker symbol: ")
# If no ticker symbol is entered, use AAPL as the default
if ticker_input == '':
    ticker_symbol = 'AAPL'
else:
    ticker_symbol = ticker_input
# Retrieve stock price data using yfinance
stock_data = yf.download(ticker_symbol, period='2y')
stock_data.reset_index(inplace=True)

# Prepare data for Prophet
#ds: The date of the stock price
#yhat: The predicted stock price
#yhat_lower: The lower bound of the prediction confidence interval
#yhat_upper: The upper bound of the prediction confidence interval
data = stock_data[['Date', 'Close']]
data = data.rename(columns={'Date': 'ds', 'Close': 'y'})
# Train the Prophet model
model = Prophet()
model.fit(data)

# Make future predictions
days_forecast = st.slider('Prediction for Next 7 Days', min_value=1, max_value=365, value=7)
future = model.make_future_dataframe(periods=days_forecast)
forecast = model.predict(future)

# Display the program using Streamlit
# Show current stock price
latest_price = stock_data['Close'].iloc[-1]
st.subheader(f'Current Stock Price ({ticker_symbol}): ₹{latest_price:.2f}')

# Calculate the accuracy of the forecast
actual_prices = stock_data['Close'].tail(days_forecast)
# Calculate the mean absolute error

mae = np.mean(np.abs(actual_prices - forecast['yhat']))
# Calculate the percentage of accuracy
percentage_accuracy = 100 - (mae / latest_price) * 100

# Display the accuracy of the forecast
st.subheader('Forecast Accuracy')
st.write(f'Mean Absolute Error: ₹{mae:.2f}')
st.write(f'Percentage Accuracy: {percentage_accuracy:.2f}%')

# Show forecasted prices
st.subheader(f'Forecasted Stock Prices ({ticker_symbol})')
st.dataframe(forecast[['ds', 'yhat']].tail(days_forecast))
st.markdown("***")

# Plot the forecast
# Predict the future values
forecast = model.predict(future)

# Plot the forecast
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], name='Actual'))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Predicted', line=dict(color='red')))
fig.update_layout(title='Stock Price Forecasted Chart', xaxis_title='Date', yaxis_title='Price', width=1250, height=600)
st.plotly_chart(fig)
st.subheader('Forecasted Stock Price Chart')

#st.line_chart(forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']])
#st.area_chart(forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']])
# Click to Download Forecasted Data button
if st.button('Click to Download Forecasted Data'):
    # Create a file name
    file_name = f'{ticker_symbol}_forecast.csv'
    # Write the data to a file
    forecast.to_csv(file_name, index=False)
    # Download the file
    with open(file_name, 'rb') as f:
        st.download_button(label='Ready to Download', data=f.read(), file_name=file_name)
    # Show the cancel button
    if st.button('Cancel Download'):
        # Delete the file if cancel button is clicked
        st.remove(file_name)

st.markdown("***")
st.markdown("<h6 style='text-align:center; color: white;'>© STOCK-BOAT</h6>", unsafe_allow_html=True)