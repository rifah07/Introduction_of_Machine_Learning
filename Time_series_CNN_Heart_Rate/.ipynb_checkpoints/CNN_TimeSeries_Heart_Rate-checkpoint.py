import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load the dataset from a CSV file
df = pd.read_csv('heart_rate_time_series.csv')

# Assume the dataset has 'timestamp' and 'heart_rate' columns
heart_rates = df['heart_rate'].values

# Normalize the heart rate values
heart_rates = (heart_rates - heart_rates.mean()) / heart_rates.std()

# Create sequences for time series prediction
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 10  # Length of the sequence
X, y = create_sequences(heart_rates, seq_length)

# Split the dataset into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape data for CNN input (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], seq_length, 1))
X_test = X_test.reshape((X_test.shape[0], seq_length, 1))

# Print the shapes to ensure they are correct
print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')

# Build the CNN model with adjusted kernel sizes and pooling
model = models.Sequential([
    layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(seq_length, 1)),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(filters=128, kernel_size=2, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(filters=256, kernel_size=2, activation='relu'),
    layers.Flatten(),
    layers.Dense(100, activation='relu'),
    layers.Dropout(0.5),  # Add dropout to prevent overfitting
    layers.Dense(1)  # Regression output
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model with more epochs and capture the history
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Function to plot training and validation loss
def plot_loss(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')
    plt.show()

# Plot the training and validation loss
plot_loss(history)

# Function to take user input and make predictions
def predict_with_user_input():
    print(f"Enter {seq_length} heart rate values for prediction (comma separated):")
    user_input = input()
    user_input = [float(x) for x in user_input.split(',')]
    
    if len(user_input) != seq_length:
        print(f"Please enter exactly {seq_length} heart rate values.")
        return
    
    # Normalize the user input using the same mean and std as the training data
    user_input = (np.array(user_input) - heart_rates.mean()) / heart_rates.std()
    
    # Convert user input to numpy array and reshape for model prediction
    user_input = user_input.reshape((1, seq_length, 1))
    
    # Make prediction
    prediction = model.predict(user_input)
    print(f"Predicted next heart rate value: {prediction[0][0]:.2f}")

# Test the model with user input
predict_with_user_input()