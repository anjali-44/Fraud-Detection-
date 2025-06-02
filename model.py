import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import joblib

# Load data
df = pd.read_csv('synthetic_data.csv')

# Preprocess data
# One-hot encode categorical features
encoder = OneHotEncoder(sparse_output=False, drop='first')  # Drop first to avoid redundancy
encoded_time_of_day = encoder.fit_transform(df[['time_of_day']])
encoded_device = pd.get_dummies(df['device'], drop_first=True)

# Combine encoded features with numerical ones
X = pd.concat([
    df[['amount', 'transaction_frequency']],
    pd.DataFrame(encoded_time_of_day, columns=encoder.get_feature_names_out(['time_of_day'])),
    encoded_device
], axis=1)

# Target variable
y = df['label']

# Balance the dataset using SMOTE
print("Before SMOTE: Non-Fraudulent =", sum(y == 0), "Fraudulent =", sum(y == 1))
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("After SMOTE: Non-Fraudulent =", sum(y_resampled == 0), "Fraudulent =", sum(y_resampled == 1))

# Split the balanced data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Feature Importance Analysis
feature_importance = model.feature_importances_
features = X.columns

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(features, feature_importance, color="skyblue")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance in Fraud Detection")
plt.show()

# Save the model
joblib.dump(model, 'fraud_detection_model.pkl')
print("Balanced Random Forest model trained and saved as fraud_detection_model.pkl")

import json

# Save feature importance
feature_importance = model.feature_importances_
features = list(X.columns)
importance_data = {feature: importance for feature, importance in zip(features, feature_importance)}

# Save to a JSON file
with open("feature_importance.json", "w") as f:
    json.dump(importance_data, f)
print("Feature importance saved as feature_importance.json")

