# import numpy as np 
# import pandas as pd
# import matplotlib.pyplot as plt
# import plotly.express as px
# import plotly.graph_objects as go
# import math
# import keras
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import Dropout
# from keras.layers import *
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error
# from keras.callbacks import EarlyStopping
# from statsmodels.tsa.stattools import adfuller

# df = pd.read_csv('1.csv')
# print(df.dtypes)
# df.head()

# df=pd.read_csv('1.csv')

# df['Timestamp [ms]'] = pd.to_datetime(df['Timestamp [ms]'].index,unit='s')

# df.head(4)
# df.dtypes
# type(df.index)
# print(df.shape)
# df.columns
# df=df.iloc[:,[0,3]]

# df.columns=['time','cpu_usage']
# print(df.head(10))
# dataset = df.cpu_usage.values #numpy.ndarray
# dataset = dataset.astype('float32')
# dataset = np.reshape(dataset, (-1, 1))



# scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)
# train_size = int(len(dataset) * 0.80)
# test_size = len(dataset) - train_size
# train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# def create_dataset(dataset, look_back=1):
#     X, Y = [], []
#     for i in range(len(dataset)-look_back-1):
#         a = dataset[i:(i+look_back), 0]
#         X.append(a)
#         Y.append(dataset[i + look_back, 0])
#     return np.array(X), np.array(Y)
    
# look_back = 30
# X_train, Y_train = create_dataset(train, look_back)
# X_test, Y_test = create_dataset(test, look_back)

# # reshape input to be [samples, time steps, features]
# X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
# X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
# model = Sequential()
# model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(Dropout(0.2))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')

# history = model.fit(X_train, Y_train, epochs=20, batch_size=70, validation_data=(X_test, Y_test), 
#                     callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)

# model.summary()
# train_predict = model.predict(X_train)
# test_predict = model.predict(X_test)
# # invert predictions
# train_predict = scaler.inverse_transform(train_predict)
# Y_train = scaler.inverse_transform([Y_train])
# test_predict = scaler.inverse_transform(test_predict)
# Y_test = scaler.inverse_transform([Y_test])


# print('Train Mean Absolute Error:', mean_absolute_error(Y_train[0], train_predict[:,0]))
# print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_train[0], train_predict[:,0])))
# print('Test Mean Absolute Error:', mean_absolute_error(Y_test[0], test_predict[:,0]))
# print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_test[0], test_predict[:,0])))

# # Function to predict next day's CPU usage at different intervals
# def predict_next_day(recent_values, model, scaler, intervals=24):
#     """
#     Predict CPU usage for next day at specified intervals
#     Args:
#         recent_values: Recent CPU usage values
#         model: Trained LSTM model
#         scaler: Fitted MinMaxScaler
#         intervals: Number of predictions to make (default 24 for hourly predictions)
#     Returns:
#         List of predicted CPU usage values
#     """
#     predictions = []
#     timestamps = []
#     current_time = datetime.now()
    
#     # Use last 30 values for initial prediction
#     current_sequence = np.array(recent_values[-30:])
    
#     for i in range(intervals):
#         # Reshape sequence for prediction
#         sequence = np.reshape(current_sequence, (1, 1, len(current_sequence)))
        
#         # Get prediction
#         pred = model.predict(sequence, verbose=0)
        
#         # Inverse transform to get actual value
#         pred = scaler.inverse_transform(pred)[0][0]
        
#         # Calculate timestamp for this prediction
#         pred_time = current_time + timedelta(hours=i+1)
        
#         predictions.append(pred)
#         timestamps.append(pred_time.strftime("%Y-%m-%d %H:%M:%S"))
        
#         # Update sequence for next prediction
#         current_sequence = np.append(current_sequence[1:], pred)
    
#     return predictions, timestamps

# plt.figure(figsize=(8,4))
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Test Loss')
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epochs')
# plt.legend(loc='upper right')
# plt.show()
# print(train_size,test_size)
# df.time[train_size:]
# aa=df.time[train_size:][:100]
# plt.figure(figsize=(10,7))
# plt.plot(aa, Y_test[0][:100], marker='.', label="actual")
# plt.plot(aa, test_predict[:,0][:100], 'r', label="prediction")
# # plt.tick_params(left=False, labelleft=True) #remove ticks
# plt.tight_layout()
# # sns.despine(top=True)
# plt.subplots_adjust(left=0.07)
# plt.ylabel('CPU Usage in MHz', size=15)
# plt.xlabel('Time step', size=15)
# plt.legend(fontsize=15)
# plt.show();
