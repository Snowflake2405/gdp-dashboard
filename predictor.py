import streamlit as st
#import pickle
import pandas as pd
import numpy as np
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
#import seaborn as sns
#%matplotlib inline

# Load a pre-trained model (example: RandomForest)
# Ensure you've trained and saved your model using pickle or joblib
#model = pickle.load(open("stocks.pkl", "rb"))

#pandas_datareader.DataReader("GAZP", 'moex') не работает так что используем API, новый вариант
# 500 rows - one request limit
#interval=24 - 1 day, interval=60 - 1 hour, interval=10 - 10 min candles
#import requests
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split
import pandas as pd
import io

@st.cache_data
def get_moex(ticker, start_date, end_date):
    """
    Загружаем данные с iss.moex.com для данной акции и интервала времени

    Аргументы:
        secid (str): ID акции ('GAZP').
        start_date (str): начальная дата в формате datetime.
        end_date (str): Конечная дата в формате datetime.

    Возвращает:
        pandas.DataFrame с данными в формате yfinance
    """
    # 500 дней за раз, разбиваем на интервалы
    d=[]
    for i in range(start_date.year, end_date.year+1):
        url = 'https://iss.moex.com/iss/engines/stock/markets/shares/securities/' + ticker + '/candles.csv' \
            '?interval=24&from='+str(i)+'-01-01&till='+str(i)+'-12-31'
        print(url)
        d.append(pd.read_csv(url, delimiter=';', skiprows = 2))
    YY=pd.concat(d, ignore_index = True)
    YY['begin']=YY['begin'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    YY.set_index('begin', inplace=True)
    idxnum=["Close", 	"High", 	"Low", 	"Open", 	"Volume"]
    YY=YY.rename(columns={i.lower():i for i in idxnum},level=0)
    return YY

@st.cache_data
def prep_data(ticker,now):
    df=get_moex(ticker, datetime(now.year-2,1,1), now)
# Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    df['SClose'] = scaler.fit_transform(df['Close'].values.reshape(-1,1))

    X = []
    y = []
    for  i in range(len(df)):
        now=df.index[i]
        in11m=df.index[i+231]
        in12m=in11m+relativedelta(months=+1)
        if in12m>=datetime.now():
            break
        X.append(df['SClose'].iloc[i:i+231])
        y.append(np.mean(df['SClose'][(in11m<df.index)&(df.index<in12m)]))
    X=np.array(X)
    y=np.array(y)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    # Build the LSTM model
    model = Sequential([
  Bidirectional(LSTM(128, return_sequences=True)),
  Bidirectional(LSTM(64, return_sequences=False)),
  Dense(25),
  Dense(1)])

# Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

# Train the model
    model.fit(np.expand_dims(x_train, axis=-1), y_train, batch_size=16, epochs=5, validation_data=(x_test, y_test))
    return [df,model]
# Title of the app
st.title("Machine Learning Model Deployment")

# Description
st.write("Это веб-приложение предсказывает котировки акций.")

# Input fields for user
st.header("Входные данные: код акции и дата прогнозирования")
ticker = st.text_input("Введите акцию (4 letters):", max_chars=6)
now=datetime.now()
now = now.replace(hour=0, minute=0, second=0, microsecond=0)

import matplotlib.pyplot as plt
#print(st.session_state)
if 'time2' not in st.session_state:
    st.session_state.time2 = now
st.session_state.time2 = st.slider("дата прогнозирования", 
    min_value=datetime(now.year-1,1,1), max_value=now, step=timedelta(days=1), 
    value=st.session_state.time2)

if ticker!="":
    df, model=prep_data(ticker, now)
# Prediction button
#if st.button("Мне повезет!"):
    time1 = st.session_state.time2-timedelta(days=231)


    fig, ax=plt.subplots()
#plt.figure(figsize=(4, 2))
    df['SClose'].iloc[-250:].plot()

    mask=(df.index > time1) & (df.index < st.session_state.time2)
    prediction = model.predict(df['SClose'][mask])

    plt.plot([st.session_state.time2,st.session_state.time2+timedelta(days=30)],[prediction[0],prediction[0]], color='red', linewidth=1, label='Predicted')
    st.pyplot(fig)
#st.write(f"The predicted output is: {prediction}")
