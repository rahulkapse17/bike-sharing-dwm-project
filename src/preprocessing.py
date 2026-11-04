import pandas as pd
import numpy as np
import os

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Load dataset (supports both repo structure and flat folder)
raw_path = os.path.join("data", "raw", "hour.csv")
input_file = raw_path if os.path.exists(raw_path) else "hour.csv"
df = pd.read_csv(input_file)

print("BIKE SHARING DATASET")
print("--------------------")

# Display first five records
print("\nOriginal Dataset:")
print(df.head())

# Display dataset information
print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# Remove duplicate records
df = df.drop_duplicates()

# Handle missing numerical values
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())

# Handle missing categorical values
for col in df.select_dtypes(include=["object", "str"]).columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

# Save cleaned dataset
df.to_csv("outputs/cleaned_bike_sharing.csv", index=False)

# Display final results
print("\nPreprocessing Completed Successfully!")

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nRemaining Missing Values:")
print(df.isnull().sum())

print("\nCleaned Dataset Saved Successfully!")
