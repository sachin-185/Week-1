# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the CSV file
df = pd.read_csv("data.csv")  # Replace with your file path
df.drop(columns=["Unnamed: 0"], inplace=True)

# Step 2: Separate input and output
X = df[[f"sensor_{i}" for i in range(20)]]  # Sensor inputs
y = df["parcel_0"]  # Label: 1 = irrigate, 0 = no need

# Step 3: Scale the sensor values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 5: Build ANN model
model = Sequential([
    Dense(32, activation='relu', input_shape=(20,)),  # Hidden Layer 1
    Dense(16, activation='relu'),                     # Hidden Layer 2
    Dense(1, activation='sigmoid')                    # Output Layer
])

# Step 6: Compile model
model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Step 7: Train model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Step 8: Predict and evaluate
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype("int32")

print("\n📊 Model Performance:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 9: Plot prediction probability distribution
plt.figure(figsize=(8,4))
plt.hist(y_pred_prob, bins=30, color='skyblue')
plt.title("Prediction Probabilities")
plt.xlabel("Probability of Irrigation")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Step 10: Automation logic (decision function)
def should_irrigate(sensor_values):
    sensor_values_scaled = scaler.transform([sensor_values])
    prediction_prob = model.predict(sensor_values_scaled)[0][0]
    print(f"\nNew Sensor Input → Predicted Probability: {prediction_prob:.2f}")
    if prediction_prob > 0.5:
        print("🚿 Irrigation Needed → Turn ON pump!")
    else:
        print("🌿 Soil Moist Enough → No need to irrigate.")

# Step 11: Try with sample input (e.g., row 10 from your data)
sample_input = X.iloc[10].tolist()
should_irrigate(sample_input)
