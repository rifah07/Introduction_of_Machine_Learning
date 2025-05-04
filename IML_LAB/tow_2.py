import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

df = pd.read_csv('student_habits_performance.csv')

print("Loaded Data Columns:", df.columns)

label_encoder = LabelEncoder()
df['gender_encoded'] = label_encoder.fit_transform(df['gender'])

gender_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Gender mapping:", gender_mapping)

unique_genders = df['gender_encoded'].unique()
print("Unique gender values:", unique_genders)
print("Gender class distribution:", np.bincount(df['gender_encoded']))

X = df.drop(['gender', 'gender_encoded', 'student_id'], axis=1)
y = df['gender_encoded']

for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_scaled, y)
print("Class distribution after SMOTE:", np.bincount(y_resampled))

y_onehot = to_categorical(y_resampled)

X_reshaped = np.reshape(X_resampled, (X_resampled.shape[0], X_resampled.shape[1], 1))

X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y_onehot, test_size=0.2, random_state=42
)

model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(32))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dense(y_onehot.shape[1], activation='softmax'))

model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    X_train, y_train, 
    epochs=50, 
    batch_size=16, 
    validation_data=(X_test, y_test), 
    verbose=1
)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"LSTM Test Accuracy: {accuracy:.4f}")

y_pred_probabilities = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probabilities, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_test_classes, y_pred_classes))