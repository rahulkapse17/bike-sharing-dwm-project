
import pandas as pd
import numpy as np
import os

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Load cleaned dataset
df = pd.read_csv("outputs/cleaned_bike_sharing.csv")

print("BIKE SHARING DATA REDUCTION")
print("----------------------------")

# ---------------------------------
# 1. ATTRIBUTE SUBSET SELECTION
# ---------------------------------

# Select relevant input attributes
features = [
    "season", "mnth", "hr", "weekday",
    "weathersit", "temp", "atemp",
    "hum", "windspeed", "workingday"
]

X = df[features]
y = df["cnt"]

# Select top 5 important attributes
selector = SelectKBest(score_func=f_regression, k=5)

X_selected = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]

print("\nSelected Attributes:")
print(list(selected_features))

# Save selected attributes
selected_df = df[list(selected_features) + ["cnt"]]
selected_df.to_csv(
    "outputs/selected_attributes.csv", index=False
)

# ---------------------------------
# 2. PRINCIPAL COMPONENT ANALYSIS
# ---------------------------------

# Standardize selected numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=0.95)

X_pca = pca.fit_transform(X_scaled)

print("\nPCA Results:")
print("Original Number of Attributes:", X.shape[1])
print("Reduced Number of Components:", X_pca.shape[1])

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("\nTotal Explained Variance:")
print(sum(pca.explained_variance_ratio_))

# Save PCA results
pca_columns = [
    f"PC{i+1}" for i in range(X_pca.shape[1])
]

pca_df = pd.DataFrame(X_pca, columns=pca_columns)
pca_df["cnt"] = y.values

pca_df.to_csv("outputs/pca_reduced.csv", index=False)

print("\nData Reduction Completed Successfully!")

print("\nFiles Saved:")
print("1. outputs/selected_attributes.csv")
print("2. outputs/pca_reduced.csv")
