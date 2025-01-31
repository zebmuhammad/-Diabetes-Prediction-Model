import joblib
import numpy as np

# Load the trained model
model = joblib.load("diabetes_logistic_model.pkl")

# Define a new sample input
sample_input = np.array([[2, 120, 70, 30, 0, 32.0, 0.5, 28]])

# Predict
sample_prediction = model.predict(sample_input)

# Print the result
print("\n🔍 Prediction:", "Diabetic" if sample_prediction[0] == 1 else "Non-Diabetic")
