# Import needed libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load datasets
application_df = pd.read_csv('application_record.csv')
credit_df = pd.read_csv('credit_record.csv')

# Display the first few rows of the dataset application record
print(application_df.head())

# Display the first few rows of the dataset credit record
print(credit_df.head())

# Merge datasets on the 'ID' column
merged_df = pd.merge(application_df, credit_df, on='ID', how='inner')

# Checking for missing values
merged_df.isnull().sum()

# Replace missing values in OCCUPATION_TYPE with 'Unknown'
merged_df['OCCUPATION_TYPE'].fillna('Unknown', inplace=True)

# Checking if the missing values are gone
merged_df.isnull().sum()

# Convert categorical features to numerical using Label Encoding
features = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY','CNT_CHILDREN', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
            'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL','OCCUPATION_TYPE','CNT_FAM_MEMBERS','MONTHS_BALANCE']

le = LabelEncoder()

for col in features:
    merged_df[col] = le.fit_transform(merged_df[col])

# Create a target variable 'TARGET' based on the credit record
merged_df['TARGET'] = merged_df['STATUS'].apply(lambda x: 1 if x in ['C', 'X'] else 0)

# Feature engineering
X = merged_df[features]
y = merged_df['TARGET']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Plot feature importances
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), feature_importance[sorted_idx], align="center")
plt.xticks(range(X.shape[1]),np.array(features)[sorted_idx], rotation=45)
plt.title("Feature Importances")
plt.show()
