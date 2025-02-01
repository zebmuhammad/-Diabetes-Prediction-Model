# Step 1: Import Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Step 2: Load the Dataset
df = pd.read_csv('diabetes.csv') 
print(df.head())  # Check the first 5 rows

# Step 3: Explore the Data
print(df.info())  # Check for missing values and data types
print(df.describe())  # Summary statistics

# Step 4: Handle Missing Values
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
df.fillna(df.mean(numeric_only=True), inplace=True)  # Avoid Pandas warning

# Step 5: Split the Data into Features (X) and Target (y)
X = df.drop('Outcome', axis=1)  # Features
y = df['Outcome']  # Target

# Step 6: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Fix: Now check class distribution after defining y_train
unique, counts = np.unique(y_train, return_counts=True)
print(f"Class Distribution in Training Set: {dict(zip(unique, counts))}")

# Step 8: Train the Logistic Regression Model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Model training check
if hasattr(model, "coef_"):
    print("✅ Model trained successfully!")
else:
    print("❌ Model not trained. Check your script.")

# Step 9: Make Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC-AUC

# Take 5 test samples
sample_input = X_test[:5]  
predictions = model.predict(sample_input)

# Print classification report
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Print confusion matrix
print("\n🟦 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("Predictions:", predictions)
print("Actual labels:", y_test[:5].values)

# Step 10: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'ROC-AUC Score: {roc_auc:.2f}')

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Step 11: Cross-Validation (More Robust Evaluation)
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
print(f'Cross-Validation Accuracy: {np.mean(cv_scores) * 100:.2f}%')

# Step 12: Save the Model
with open('diabetes_logistic_model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("Model saved as 'diabetes_logistic_model.pkl'")

# Step 13: Feature Importance (Coefficients)
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_[0]})
coefficients = coefficients.sort_values(by='Coefficient', ascending=False)
print("Feature Coefficients:")
print(coefficients)

# Plot Feature Coefficients
sns.barplot(x='Coefficient', y='Feature', data=coefficients)
plt.title('Feature Coefficients (Logistic Regression)')
plt.show()
