import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import csv

# Load and preprocess data
df = pd.read_csv("HistoricalData.csv")
df['Date'] = pd.to_datetime(df['Date'])

for col in ['Close/Last', 'Open', 'High', 'Low']:
    df[col] = df[col].astype(str).str.replace('$', '', regex=False).astype(float)

df.sort_values(by='Date', inplace=True)
features = ['Open', 'High', 'Low', 'Volume']
target = 'Close/Last'

scaler = MinMaxScaler()
df[features + [target]] = scaler.fit_transform(df[features + [target]])
df = df.dropna()

# Sequence preparation
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length, :-1])
        y.append(data[i + seq_length, -1])
    return np.array(X), np.array(y)

seq_length = 10
data = df[features + [target]].values
X, y = create_sequences(data, seq_length)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model builder
def build_lstm(optimizer='adam', dropout_rate=0.2, units_per_layer=[50], activation='relu'):
    model = Sequential()
    for i, units in enumerate(units_per_layer):
        return_seq = i < (len(units_per_layer) - 1)
        if i == 0:
            model.add(LSTM(units=units, return_sequences=return_seq, activation=activation,
                           input_shape=(X_train.shape[1], X_train.shape[2])))
        else:
            model.add(LSTM(units=units, return_sequences=return_seq, activation=activation))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Hyperparameter grid
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'dropout_rate': [0.2, 0.3],
    'units_per_layer': [[50], [50, 100]],
    'batch_size': [16, 32],
    'epochs': [20],
    'activation': ['relu', 'sigmoid']
}

best_loss = float('inf')
best_params = None

# CSV file for output
output_csv = 'B190305004.csv'

# Open CSV and write
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Parameter', 'Value'])  # CSV Header

    trial = 1
    for optimizer in param_grid['optimizer']:
        for dropout_rate in param_grid['dropout_rate']:
            for units_per_layer in param_grid['units_per_layer']:
                for batch_size in param_grid['batch_size']:
                    for epochs in param_grid['epochs']:
                        for activation in param_grid['activation']:
                            print(f"\nSearch: Running Trial #{trial}")
                            print("-" * 40)

                            model = build_lstm(optimizer=optimizer, dropout_rate=dropout_rate,
                                               units_per_layer=units_per_layer, activation=activation)
                            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
                            loss = model.evaluate(X_test, y_test, verbose=0)

                            # Display parameters vertically in terminal
                            hyperparams = {
                                'Trial': trial,
                                'optimizer': optimizer,
                                'dropout_rate': dropout_rate,
                                'units_per_layer': units_per_layer,
                                'batch_size': batch_size,
                                'epochs': epochs,
                                'activation': activation,
                                'loss': loss
                            }
                            for key, value in hyperparams.items():
                                print(f"{key:<20} | {value}")

                            # Write to CSV
                            for key, value in hyperparams.items():
                                writer.writerow([key, value])
                            writer.writerow([])  # Add a blank row for readability in CSV

                            # Update best parameters
                            if loss < best_loss:
                                best_loss = loss
                                best_params = hyperparams

                            trial += 1

# Display best hyperparameters
print("\nBest Hyperparameters:")
print("-" * 40)
for key, value in best_params.items():
    print(f"{key:<20} | {value}")