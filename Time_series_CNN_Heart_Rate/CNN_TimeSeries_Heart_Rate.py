import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import os

# Load the dataset
data = pd.read_csv('heart_rate_time_series.csv')

# Ensure the dataset has the correct columns
if 'timestamp' not in data.columns or 'heart_rate' not in data.columns:
    raise ValueError("The dataset must have 'timestamp' and 'heart_rate' columns.")

# Extract the heart rate data
timestamps = data['timestamp']
heart_rates = data['heart_rate'].values.reshape(-1, 1)

# Normalize the heart rate data
scaler = MinMaxScaler()
heart_rates_scaled = scaler.fit_transform(heart_rates)

# Create sequences for time series
sequence_length = 20  # Number of timesteps in each sequence
X, y = [], []
for i in range(len(heart_rates_scaled) - sequence_length):
    X.append(heart_rates_scaled[i:i + sequence_length])
    y.append(heart_rates_scaled[i + sequence_length])

X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check if a saved model exists
model_path = 'cnn_heart_rate_model.h5'
if os.path.exists(model_path):
    print("Loading saved model...")
    model = load_model(model_path)
else:
    print("Building and training a new model...")
    # Build the CNN model
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(sequence_length, 1)),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1)  # Regression output
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=32
    )

    # Save the model
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Plot training vs validation performance
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('Training vs Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig('train_vs_test_loss.png')
    print("The training vs testing loss plot has been saved as 'train_vs_test_loss.png'.")

# Evaluate the model on the test data
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")


# Generate predictions for the test data
y_pred = model.predict(X_test)

# Inverse scale the predictions and actual values
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test)

# Create a DataFrame for easy comparison
comparison_df = pd.DataFrame({
    'Actual': y_test_rescaled.flatten(),
    'Predicted': y_pred_rescaled.flatten()
})

# Plot the actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, label='Actual Heart Rate')
plt.plot(y_pred_rescaled, label='Predicted Heart Rate', linestyle='--')
plt.title('Actual vs Predicted Heart Rate')
plt.xlabel('Sample')
plt.ylabel('Heart Rate')
plt.legend()
plt.grid()
plt.savefig('actual_vs_predicted_heart_rate.png')
print("The training vs testing loss plot has been saved as 'actual_vs_predicted_heart_rate.png'.")

#plt.show()

# Optionally, print some of the comparison
print("First 5 testing vs predicted values:")
print(comparison_df.head())



# Predict heart rate based on user input
def predict_heart_rate():
    print("Enter the last 20 heart rate values (separated by spaces):")
    user_input = input()
    try:
        user_sequence = np.array([float(x) for x in user_input.split()])
        if len(user_sequence) != sequence_length:
            raise ValueError(f"Please enter exactly {sequence_length} values.")
        user_sequence = user_sequence.reshape(-1, 1)
        user_sequence_scaled = scaler.transform(user_sequence)
        user_sequence_scaled = user_sequence_scaled.reshape(1, sequence_length, 1)
        prediction_scaled = model.predict(user_sequence_scaled)
        prediction = scaler.inverse_transform(prediction_scaled)
        print(f"Predicted next heart rate value: {prediction[0][0]:.2f}")
    except Exception as e:
        print(f"Error: {e}")

# Call the prediction function
predict_heart_rate()