# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 23:13:42 2025

@author: monas
"""

"""pr4"""

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from sklearn.preprocessing import StandardScaler
import numpy as np, pandas as pd, tensorflow as tf
import plotly.graph_objects as go
import matplotlib.pyplot as plt

df = pd.read_csv('GOOG.csv')
df = df[['Date', 'Close']]
df.info()
df['Date'].min(), df['Date'].max()

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close price'))
fig.update_layout(title='Google Stock Price 2004–2020', showlegend=True)
fig.show()

train = df.loc[df['Date'] <= '2017-12-24']
test = df.loc[df['Date'] > '2017-12-24']

scaler = StandardScaler()
scaler.fit(np.array(train['Close']).reshape(-1,1))
train['Close'] = scaler.transform(np.array(train['Close']).reshape(-1,1))
test['Close'] = scaler.transform(np.array(test['Close']).reshape(-1,1))

TIME_STEPS = 30

def create_sequences(X, y, time_steps=TIME_STEPS):
    X_out, y_out = [], []
    for i in range(len(X)-time_steps):
        X_out.append(X.iloc[i:(i+time_steps)].values)
        y_out.append(y.iloc[i+time_steps])
    return np.array(X_out), np.array(y_out)

X_train, y_train = create_sequences(train[['Close']], train['Close'])
X_test, y_test = create_sequences(test[['Close']], test['Close'])

model = Sequential([
    LSTM(128, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    RepeatVector(X_train.shape[1]),
    LSTM(128, activation='tanh', return_sequences=True),
    Dropout(0.2),
    TimeDistributed(Dense(X_train.shape[2]))
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=100, batch_size=32,
    validation_split=0.1,
    shuffle=False,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

X_train_pred = model.predict(X_train)
train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

plt.hist(train_mae_loss, bins=50)
plt.xlabel('Train MAE loss'); plt.ylabel('Samples')

threshold = np.max(train_mae_loss)
print('Reconstruction error threshold:', threshold)

X_test_pred = model.predict(X_test)
test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

plt.hist(test_mae_loss, bins=50)
plt.xlabel('Test MAE loss'); plt.ylabel('Samples')

anomaly_df = pd.DataFrame(test[TIME_STEPS:])
anomaly_df['loss'] = test_mae_loss
anomaly_df['threshold'] = threshold
anomaly_df['anomaly'] = anomaly_df['loss'] > anomaly_df['threshold']
anomaly_df.head()

fig = go.Figure()
fig.add_trace(go.Scatter(x=anomaly_df['Date'], y=anomaly_df['loss'], name='Test loss'))
fig.add_trace(go.Scatter(x=anomaly_df['Date'], y=anomaly_df['threshold'], name='Threshold'))
fig.update_layout(title='Test Loss vs Threshold')
fig.show()

# Mark anomalies
anomalies = anomaly_df[anomaly_df['anomaly'] == True]

fig = go.Figure()
fig.add_trace(go.Scatter(x=anomaly_df['Date'],
                         y=scaler.inverse_transform(anomaly_df[['Close']]),
                         name='Close Price'))
fig.add_trace(go.Scatter(x=anomalies['Date'],
                         y=scaler.inverse_transform(anomalies[['Close']]),
                         mode='markers', name='Anomalies'))
fig.update_layout(title='Detected Anomalies')
fig.show()

