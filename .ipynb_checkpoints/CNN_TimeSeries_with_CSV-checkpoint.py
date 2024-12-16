#pip install numpy pandas tensorflow

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# Load the dataset from a CSV file
df = pd.read_csv('time_series_data.csv')

# Assume the target column is named 'target' and the rest are feature columns
X = df.drop(columns=['target']).values
y = df['target'].values

# Split the dataset into train and test sets
train_size = int(len(df) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape data for CNN input (samples, timesteps, features, channels)
num_features = X_train.shape[1]
X_train = X_train.reshape((X_train.shape[0], num_features, 1))
X_test = X_test.reshape((X_test.shape[0], num_features, 1))

# Build the CNN model
model = models.Sequential([
    layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(num_features, 1)),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(100, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')