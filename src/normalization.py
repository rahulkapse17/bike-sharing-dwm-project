
import pandas as pd
import numpy as np
import os

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Load cleaned dataset
df = pd.read_csv("outputs/cleaned_bike_sharing.csv")

print("BIKE SHARING DATA NORMALIZATION")
print("--------------------------------")

# Select numerical attributes for normalization
features = ["temp", "atemp", "hum", "windspeed"]

print("\nOriginal Data:")
print(df[features].head())

# --------------------------------
# 1. MIN-MAX NORMALIZATION
# --------------------------------

min_max_df = df.copy()

for col in features:
    min_val = df[col].min()
    max_val = df[col].max()

    min_max_df[col] = (df[col] - min_val) / (max_val - min_val)

print("\nMin-Max Normalized Data:")
print(min_max_df[features].head())

# Save Min-Max dataset
min_max_df.to_csv("outputs/minmax_normalized.csv", index=False)

# --------------------------------
# 2. Z-SCORE STANDARDIZATION
# --------------------------------

zscore_df = df.copy()

for col in features:
    mean_val = df[col].mean()
    std_val = df[col].std()

    zscore_df[col] = (df[col] - mean_val) / std_val

print("\nZ-Score Standardized Data:")
print(zscore_df[features].head())

# Save Z-Score dataset
zscore_df.to_csv("outputs/zscore_normalized.csv", index=False)

# --------------------------------
# DISPLAY RESULTS
# --------------------------------

print("\nMin-Max Range:")
print(min_max_df[features].agg(["min", "max"]))

print("\nZ-Score Mean:")
print(zscore_df[features].mean())

print("\nNormalization Completed Successfully!")

print("\nFiles Saved:")
print("1. outputs/minmax_normalized.csv")
print("2. outputs/zscore_normalized.csv")
