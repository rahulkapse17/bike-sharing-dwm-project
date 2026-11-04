
import pandas as pd
import os
import time

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Load cleaned dataset from Module 2
df = pd.read_csv("outputs/cleaned_bike_sharing.csv")

# Define demand classes using the same thresholds as classification
q1 = df["cnt"].quantile(0.33)
q2 = df["cnt"].quantile(0.66)

df["demand_class"] = pd.cut(
    df["cnt"],
    bins=[-float("inf"), q1, q2, float("inf")],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)

# Convert hour into meaningful time periods
def get_time_period(hour):
    if hour < 6:
        return "Night"
    elif hour < 12:
        return "Morning"
    elif hour < 17:
        return "Afternoon"
    elif hour < 21:
        return "Evening"
    else:
        return "Night"

df["time_period"] = df["hr"].apply(get_time_period)

# Convert numerical and categorical attributes into items
transactions = []

for _, row in df.iterrows():

    transaction = [
        "Season_" + str(int(row["season"])),
        "Weather_" + str(int(row["weathersit"])),
        "Time_" + str(row["time_period"]),
        "Demand_" + str(row["demand_class"])
    ]

    transactions.append(transaction)

# Convert transactions into a one-hot encoded DataFrame
encoder = TransactionEncoder()
encoded_array = encoder.fit(transactions).transform(transactions)

encoded_df = pd.DataFrame(
    encoded_array,
    columns=encoder.columns_
)

print("Total transactions:", len(transactions))
print("Total unique items:", len(encoder.columns_))

# -------------------------------
# APRIORI ALGORITHM
# -------------------------------

print("\n========== APRIORI ALGORITHM ==========")

start_time = time.time()

apriori_itemsets = apriori(
    encoded_df,
    min_support=0.03,
    use_colnames=True
)

apriori_time = time.time() - start_time

print("\nFrequent Itemsets using Apriori:")
print(apriori_itemsets.head(10))

apriori_itemsets.to_csv(
    "outputs/apriori_frequent_itemsets.csv",
    index=False
)

# Generate association rules
if not apriori_itemsets.empty:
    apriori_rules = association_rules(
        apriori_itemsets,
        metric="confidence",
        min_threshold=0.5
    )

    if not apriori_rules.empty:
        apriori_rules = apriori_rules.sort_values(
            by="confidence",
            ascending=False
        )

        print("\nTop Apriori Association Rules:")
        print(
            apriori_rules[
                ["antecedents", "consequents", "support",
                 "confidence", "lift"]
            ].head(10).to_string(index=False)
        )

        apriori_rules.to_csv(
            "outputs/apriori_association_rules.csv",
            index=False
        )
    else:
        print("No Apriori association rules found.")
else:
    print("No Apriori frequent itemsets found.")

print("Apriori execution time:", round(apriori_time, 4), "seconds")

# -------------------------------
# FP-GROWTH ALGORITHM
# -------------------------------

print("\n========== FP-GROWTH ALGORITHM ==========")

start_time = time.time()

fp_itemsets = fpgrowth(
    encoded_df,
    min_support=0.03,
    use_colnames=True
)

fp_time = time.time() - start_time

print("\nFrequent Itemsets using FP-Growth:")
print(fp_itemsets.head(10))

fp_itemsets.to_csv(
    "outputs/fpgrowth_frequent_itemsets.csv",
    index=False
)

# Generate association rules
if not fp_itemsets.empty:
    fp_rules = association_rules(
        fp_itemsets,
        metric="confidence",
        min_threshold=0.5
    )

    if not fp_rules.empty:
        fp_rules = fp_rules.sort_values(
            by="confidence",
            ascending=False
        )

        print("\nTop FP-Growth Association Rules:")
        print(
            fp_rules[
                ["antecedents", "consequents", "support",
                 "confidence", "lift"]
            ].head(10).to_string(index=False)
        )

        fp_rules.to_csv(
            "outputs/fpgrowth_association_rules.csv",
            index=False
        )
    else:
        print("No FP-Growth association rules found.")
else:
    print("No FP-Growth frequent itemsets found.")

print("FP-Growth execution time:", round(fp_time, 4), "seconds")

# -------------------------------
# ALGORITHM COMPARISON
# -------------------------------

comparison = pd.DataFrame({
    "Algorithm": ["Apriori", "FP-Growth"],
    "Frequent Itemsets": [
        len(apriori_itemsets),
        len(fp_itemsets)
    ],
    "Execution Time (seconds)": [
        apriori_time,
        fp_time
    ]
})

print("\n========== ALGORITHM COMPARISON ==========")
print(comparison.to_string(index=False))

comparison.to_csv(
    "outputs/association_algorithm_comparison.csv",
    index=False
)

print("\nAssociation Rule Mining completed successfully!")
print("All results are saved in the outputs folder.")
