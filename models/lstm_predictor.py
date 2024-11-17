import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

class LSTMPredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.sequence_length = 10
        
    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def build_model(self):
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(self.sequence_length, 1), return_sequences=True),
            LSTM(50, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train(self, csv_path):
        # Load and preprocess data
        df = pd.read_csv(csv_path, nrows=1000)  # Load top 1000 rows
        data = df['cpu_utilization'].values.reshape(-1, 1)
        
        # Scale data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = self.create_sequences(scaled_data)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Build and train model
        self.model = self.build_model()
        self.model.fit(X, y, epochs=50, batch_size=32, verbose=1)
        
        # Save scaler
        joblib.dump(self.scaler, 'models/scaler.save')
        
        # Save model
        self.model.save('models/lstm_model.h5')
    
    def predict_next(self, recent_values):
        if len(recent_values) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} values for prediction")
            
        # Scale the input
        scaled_input = self.scaler.transform(np.array(recent_values).reshape(-1, 1))
        
        # Create sequence
        sequence = scaled_input[-self.sequence_length:]
        sequence = sequence.reshape(1, self.sequence_length, 1)
        
        # Predict
        scaled_prediction = self.model.predict(sequence)
        
        # Inverse transform
        prediction = self.scaler.inverse_transform(scaled_prediction)
        
        return float(prediction[0][0])
