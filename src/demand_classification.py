
import pandas as pd
import os

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Load cleaned dataset from Module 2
df = pd.read_csv("outputs/cleaned_bike_sharing.csv")

# Sort records chronologically
df["dteday"] = pd.to_datetime(df["dteday"])
df = df.sort_values("dteday").reset_index(drop=True)

# Define Low, Medium and High demand using 33rd and 66th percentiles
q1 = df["cnt"].quantile(0.33)
q2 = df["cnt"].quantile(0.66)

df["demand_class"] = pd.cut(
    df["cnt"],
    bins=[-float("inf"), q1, q2, float("inf")],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)

print("Demand classification thresholds:")
print("Low demand: count <=", round(q1, 2))
print("Medium demand: count >", round(q1, 2), "and <=", round(q2, 2))
print("High demand: count >", round(q2, 2))

print("\nDemand class distribution:")
print(df["demand_class"].value_counts())

# Select input features
# Exclude cnt, casual and registered to prevent target leakage
categorical_features = [
    "season", "yr", "mnth", "hr",
    "holiday", "weekday", "workingday", "weathersit"
]

numerical_features = [
    "temp", "atemp", "hum", "windspeed"
]

features = categorical_features + numerical_features

X = df[features]
y = df["demand_class"]

# Split chronologically: first 80% for training, last 20% for testing
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Three classification models
models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        random_state=42
    ),

    # A shallow tree represents understandable IF-THEN rules
    "Rule Based": DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=50,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    )
}

results = []

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", model)
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Predict test data
    y_pred = pipeline.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, average="macro", zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print("\n==============================")
    print(name)
    print("==============================")

    print(classification_report(
        y_test,
        y_pred,
        zero_division=0
    ))

    # Save confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=["Low", "Medium", "High"]
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Low", "Actual Medium", "Actual High"],
        columns=["Predicted Low", "Predicted Medium", "Predicted High"]
    )

    safe_name = name.lower().replace(" ", "_")
    cm_df.to_csv(f"outputs/confusion_matrix_{safe_name}.csv")

# Compare all models
results_df = pd.DataFrame(results)

print("\nCLASSIFICATION MODEL COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv(
    "outputs/classification_comparison.csv",
    index=False
)

print("\nClassification completed successfully!")
print("Results saved in the outputs folder.")
