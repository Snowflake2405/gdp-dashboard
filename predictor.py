import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, normalize, StandardScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
#import seaborn as sns
#%matplotlib inline


#pandas_datareader.DataReader("GAZP", 'moex') не работает так что используем API, новый вариант
# 500 rows - one request limit
#interval=24 - 1 day, interval=60 - 1 hour, interval=10 - 10 min candles
#import requests
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split
import pandas as pd
import io
import mplcursors

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
def prep_model():
# Load a pre-trained model (example: RandomForest)
# Ensure you've trained and saved your model using pickle or joblib
    model = load_model("model.keras", safe_mode=True)
    principal_components = pd.read_csv("./principal_components.csv", parse_dates=['begin'], index_col=0)

    return [model, principal_components]

@st.cache_data
def prep_data(ticker,now):
    df = get_moex(ticker, datetime(now.year-2,1,1), now)
    return df

# Title of the app
st.title("Machine Learning Model Deployment")

# Description
st.write("Это веб-приложение предсказывает котировки акций.")

# Input fields for user
st.header("Входные данные: код акции и дата прогнозирования")
ticker = st.text_input("Введите акцию (обычно 4 буквы):", max_chars=6)
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

    model, principal_components = prep_model()
    new_data = prep_data(ticker, now)
    normalized_components = pd.DataFrame(normalize(principal_components.T).T, index=principal_components.index)
    pc_norms = np.linalg.norm(principal_components, axis=0)

# Prediction button
#if st.button("Мне повезет!"):
    time1 = st.session_state.time2-timedelta(days=231)

    fig, ax=plt.subplots()
#plt.figure(figsize=(4, 2))
    new_data[0:].Close.plot(linewidth=3, ax=ax, title="Close price")

# Scale the data
    scaler = StandardScaler()

# разложение новой акции по PC
    if not normalized_components.index.isin(new_data.index).all():
        st.write("Error: price data missing for some of the dates in the PC time window")
        st.stop()
    aligned_data=new_data.Close.loc[normalized_components.index]

    new_scaler=StandardScaler()
    new_scaled_data = new_scaler.fit_transform(
        aligned_data.values.reshape(-1, 1)).flatten()

# коэффициенты разложения исторических цен нашей акции по 20 главным компонентам акций индекса Мосбиржи
    coefficients = normalized_components.T @ new_scaled_data[:len(normalized_components)]

    pca_restored=pd.DataFrame(new_scaler.inverse_transform((normalized_components @ coefficients).array.reshape(-1, 1)), 
                                index=principal_components[:len(normalized_components)].index, 
                                columns=["price expanded in PC"])
    pca_restored.plot(c='black', linewidth=1, ax=ax)
    sequence_length = 231  # Use past 11 months for prediction
    i = principal_components.index.get_indexer([st.session_state.time2], method='pad')[0]
    X1 = principal_components.iloc[i-sequence_length:i]
#    y1 = principal_components.iloc[i:i+22].mean(axis=0)
    y_new_stock_unscaled = pca_restored.iloc[i:i+22].mean()    
    
    pred=model.predict(np.expand_dims(X1.to_numpy(), axis=0))[0]

    pred_new_stock=(pred/pc_norms*coefficients).sum()
#    y_new_stock=(y1/pc_norms*coefficients).sum()

    plt.axvline(x=principal_components.index[sequence_length], color='gray', linestyle='--', label='', linewidth=1)
    plt.axvline(x=principal_components.index[i], color='red', linestyle='--', label='', linewidth=1)
    plt.axvline(x=principal_components.index[-1], color='gray', linestyle='--', label='', linewidth=1)

#    y_new_stock_unscaled=new_scaler.inverse_transform([[y_new_stock]])[0][0]
    pred_new_stock_unscaled=new_scaler.inverse_transform([[pred_new_stock]])[0][0]
    
    plt.plot([principal_components.index[i],principal_components.index[i]+timedelta(days=30)],[y_new_stock_unscaled,y_new_stock_unscaled], c='lightgreen', linewidth=2, label='Ground Truth')
    plt.plot([principal_components.index[i],principal_components.index[i]+timedelta(days=30)],[pred_new_stock_unscaled,pred_new_stock_unscaled], c='red', linewidth=2, linestyle='--', label='Predicted')
#    st.write([normalized_components.iloc[i].index.shape, aligned_data[i].shape])
    plt.scatter([principal_components.index[i]], [aligned_data[i]], marker='o', c='red', s=50, facecolors='none')

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, ncol=1, loc='best', bbox_to_anchor=(1.05, 1))
    plt.xlabel('')
    plt.ylabel(ticker+' price')
    plt.show()


    # Enable interactive cursor
    cursor = mplcursors.cursor(ax, hover=True)
    @cursor.connect("add")
    def on_click(sel):
        st.write(f"Clicked at: x={sel.target[0]:.2f}, y={sel.target[1]:.2f}")

#    plt.plot([st.session_state.time2,st.session_state.time2+timedelta(days=30)],[pred,pred], color='red', linewidth=1, label='Predicted')
    st.pyplot(fig)
#st.write(f"The predicted output is: {prediction}")
