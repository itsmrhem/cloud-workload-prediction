import numpy as np 
import pandas as pd
from datetime import datetime, timedelta
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

class CPUPercentagePredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.look_back = 30
        
    def prepare_data(self, dataset, look_back=30):
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            X.append(dataset[i:(i + look_back), 0])
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)
    
    def train(self):
        # Load and prepare the percentage-based dataset
        df = pd.read_csv('2.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Prepare the data
        dataset = df.cpu_usage.values.astype('float32')
        dataset = np.reshape(dataset, (-1, 1))
        dataset = self.scaler.fit_transform(dataset)
        
        # Split into train and test sets
        train_size = int(len(dataset) * 0.80)
        train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
        
        # Create datasets with lookback
        X_train, Y_train = self.prepare_data(train, self.look_back)
        X_test, Y_test = self.prepare_data(test, self.look_back)
        
        # Reshape input for LSTM [samples, time steps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        
        # Create and compile model
        self.model = Sequential([
            LSTM(100, input_shape=(1, self.look_back)),
            Dropout(0.2),
            Dense(1)
        ])
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        
        # Train model
        history = self.model.fit(
            X_train, Y_train,
            epochs=20,
            batch_size=70,
            validation_data=(X_test, Y_test),
            callbacks=[EarlyStopping(monitor='val_loss', patience=10)],
            verbose=1,
            shuffle=False
        )
        
        # Make predictions
        train_predict = self.model.predict(X_train)
        test_predict = self.model.predict(X_test)
        
        # Inverse transform predictions
        train_predict = self.scaler.inverse_transform(train_predict)
        Y_train_inv = self.scaler.inverse_transform([Y_train])
        test_predict = self.scaler.inverse_transform(test_predict)
        Y_test_inv = self.scaler.inverse_transform([Y_test])
        
        # Calculate and store metrics
        self.metrics = {
            'train_mae': mean_absolute_error(Y_train_inv[0], train_predict[:,0]),
            'train_rmse': np.sqrt(mean_squared_error(Y_train_inv[0], train_predict[:,0])),
            'test_mae': mean_absolute_error(Y_test_inv[0], test_predict[:,0]),
            'test_rmse': np.sqrt(mean_squared_error(Y_test_inv[0], test_predict[:,0]))
        }
        
        # Store predictions for plotting
        self.train_predictions = train_predict
        self.test_predictions = test_predict
        self.Y_train = Y_train_inv[0]
        self.Y_test = Y_test_inv[0]
        
        # Print metrics
        print('Train Mean Absolute Error:', self.metrics['train_mae'])
        print('Train Root Mean Squared Error:', self.metrics['train_rmse'])
        print('Test Mean Absolute Error:', self.metrics['test_mae'])
        print('Test Root Mean Squared Error:', self.metrics['test_rmse'])
        
        return history, self.metrics
        
    def predict_next_day(self, intervals=24):
        """
        Predict CPU usage percentage for next day at specified intervals
        Args:
            intervals: Number of predictions to make (default 24 for hourly predictions)
        Returns:
            predictions: List of predicted CPU usage percentages
            timestamps: List of corresponding timestamps
        """
        # Load recent data
        df = pd.read_csv('2.csv')
        recent_values = df.cpu_usage.values[-self.look_back:].astype('float32')
        
        predictions = []
        timestamps = []
        current_time = datetime.now()
        
        # Scale the recent values
        current_sequence = self.scaler.transform(recent_values.reshape(-1, 1))
        current_sequence = current_sequence.flatten()[-self.look_back:]
        
        for i in range(intervals):
            # Reshape sequence for prediction
            sequence = np.reshape(current_sequence, (1, 1, len(current_sequence)))
            
            # Get prediction
            pred = self.model.predict(sequence, verbose=0)
            
            # Inverse transform to get actual percentage
            pred = self.scaler.inverse_transform(pred)[0][0]
            
            # Calculate timestamp for this prediction
            pred_time = current_time + timedelta(hours=i+1)
            
            predictions.append(pred)
            timestamps.append(pred_time.strftime("%Y-%m-%d %H:%M:%S"))
            
            # Update sequence for next prediction
            current_sequence = np.append(current_sequence[1:], self.scaler.transform([[pred]])[0])
        
        return predictions, timestamps

if __name__ == "__main__":
    predictor = CPUPercentagePredictor()
    history, metrics = predictor.train()
    
    # Plot training history
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('Model Loss During Training')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    
    # Plot actual vs predicted values for test set
    plt.subplot(1,2,2)
    plt.plot(predictor.Y_test[:100], label='Actual', color='blue')
    plt.plot(predictor.test_predictions[:100], label='Predicted', color='red')
    plt.title('Actual vs Predicted Values (First 100 Test Samples)')
    plt.ylabel('CPU Usage (%)')
    plt.xlabel('Time Steps')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Print detailed metrics
    print("\nDetailed Model Performance Metrics:")
    print(f"Training Set Performance:")
    print(f"- Mean Absolute Error: {metrics['train_mae']:.2f}%")
    print(f"- Root Mean Squared Error: {metrics['train_rmse']:.2f}%")
    print(f"\nTest Set Performance:")
    print(f"- Mean Absolute Error: {metrics['test_mae']:.2f}%")
    print(f"- Root Mean Squared Error: {metrics['test_rmse']:.2f}%")
    
    # Make and visualize predictions for next 24 hours
    predictions, timestamps = predictor.predict_next_day()
    
    plt.figure(figsize=(12,6))
    plt.plot(timestamps, predictions, marker='o', linestyle='-', linewidth=2)
    plt.title('CPU Usage Predictions for Next 24 Hours')
    plt.xlabel('Time')
    plt.ylabel('CPU Usage (%)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
