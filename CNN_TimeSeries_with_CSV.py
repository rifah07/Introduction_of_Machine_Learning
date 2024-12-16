import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

df = pd.read_csv('time_series_data.csv')

X = df.drop(columns=['target']).values
y = df['target'].values

train_size = int(len(df) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape data for CNN input (samples, timesteps, features, channels)
num_features = X_train.shape[1]
X_train = X_train.reshape((X_train.shape[0], num_features, 1))
X_test = X_test.reshape((X_test.shape[0], num_features, 1))

model = models.Sequential([
    layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(num_features, 1)),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(100, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Function to take user input and make predictions
def predict_with_user_input():
    print("Enter feature values for prediction:")
    user_input = []
    for i in range(num_features):
        value = float(input(f"Enter value for feature_{i+1}: "))
        user_input.append(value)
    
    user_input = np.array(user_input).reshape((1, num_features, 1))
    
    prediction = model.predict(user_input)
    print(f"Predicted class: {'1' if prediction > 0.5 else '0'} (Probability: {prediction[0][0]:.4f})")

predict_with_user_input()