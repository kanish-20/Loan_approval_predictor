import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Load your dataset
df = pd.read_csv("loan_data.csv")  # replace with actual CSV name

# Convert categorical to numeric if needed
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})

# ✅ Include 5 features here
X = df[['Gender', 'Married', 'ApplicantIncome', 'LoanAmount', 'Credit_History']]
y = df['Loan_Status']  # or whatever your label column is

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(model, 'loan_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model is successfully trained!")
