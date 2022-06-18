# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import streamlit as st
import yfinance as yf
from datetime import date
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objects as go

st.title("Facebook Prophet - Stock Price Prediction")
st.write("This is a demonstrator app created on Streamlit to show a use case of the Facebook Prophet "
         "timeseries model to forecast stock market data.")

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

@st.cache
def load_stock_data(ticker:str):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

tickers = ['AAPL', 'GOOG', 'TSLA', 'MSFT']

ticker = st.selectbox("Select tickers", tickers)

n_years = st.slider("Years of prediction:", 1, 5)
period = n_years * 365

data_load_state = st.text("Loading... {} data".format(ticker))
data = load_stock_data(ticker)
data_load_state.text("Loading... Complete!")


st.write(data.tail())

def plot_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['High'], name="High"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Low'], name="Low"))
    fig_title = "{} data from {} to {}".format(ticker, START, TODAY)
    fig.update_layout(title_text=fig_title, xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_data()

def make_prediction():
    df_train = data[["Date", "Close"]]
    df_train = df_train.rename(columns={"Date":"ds", "Close":"y"})

    model = Prophet()
    model.fit(df_train)
    future = model.make_future_dataframe(periods=period)
    forecast = model.predict(future)

    st.subheader("Forecast")

    st.write("Forecast timeseries")
    fig1 = plot_plotly(model, forecast)
    st.plotly_chart(fig1)

    st.write("Forecast components")
    fig2 = model.plot_components(forecast)
    st.write(fig2)

make_prediction()

if __name__ == '__main__':
    print("Run `streamlit run main.py` to run this app")
