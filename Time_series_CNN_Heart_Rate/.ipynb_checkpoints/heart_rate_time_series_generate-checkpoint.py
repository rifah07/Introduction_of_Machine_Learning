import pandas as pd
import numpy as np
import datetime

# Generate timestamps
start_time = datetime.datetime(2023, 1, 1, 0, 0)
num_minutes = 1000  # Number of minutes (data points)
timestamps = [start_time + datetime.timedelta(minutes=i) for i in range(num_minutes)]

# Generate synthetic heart rate values (random values between 60 and 100)
heart_rates = np.random.randint(60, 101, num_minutes)

# Create a DataFrame
df = pd.DataFrame({'timestamp': timestamps, 'heart_rate': heart_rates})

# Save to CSV file
df.to_csv('heart_rate_time_series.csv', index=False)
print("Synthetic heart rate dataset created and saved as 'heart_rate_time_series.csv'")