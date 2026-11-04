
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Load cleaned dataset
df = pd.read_csv("outputs/cleaned_bike_sharing.csv")

# Convert date column
df["dteday"] = pd.to_datetime(df["dteday"])

# Create daily usage profiles
daily_profiles = df.pivot_table(
    index="dteday",
    columns="hr",
    values="cnt",
    aggfunc="mean"
)

daily_profiles = daily_profiles.fillna(0)

daily_profiles.columns = [
    "Hour_" + str(hour)
    for hour in daily_profiles.columns
]

# Store daily information
daily_info = df.groupby("dteday").agg(
    season=("season", "first"),
    weekday=("weekday", "first"),
    daily_demand=("cnt", "sum")
)

daily_profiles = daily_profiles.sort_index()
daily_info = daily_info.reindex(daily_profiles.index)

print("Total daily profiles:", len(daily_profiles))
print("Features used:", len(daily_profiles.columns))

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(daily_profiles)

# ---------------------------------
# 1. K-MEANS CLUSTERING
# ---------------------------------

print("\n========== K-MEANS CLUSTERING ==========")

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(X_scaled)

daily_info["KMeans_Cluster"] = kmeans_labels

kmeans_score = silhouette_score(
    X_scaled,
    kmeans_labels
)

print("Number of Clusters: 4")
print("Silhouette Score:", round(kmeans_score, 4))

# ---------------------------------
# 2. HIERARCHICAL CLUSTERING
# ---------------------------------

print("\n========== HIERARCHICAL CLUSTERING ==========")

hierarchical = AgglomerativeClustering(
    n_clusters=4,
    linkage="ward"
)

hierarchical_labels = hierarchical.fit_predict(X_scaled)

daily_info["Hierarchical_Cluster"] = hierarchical_labels

hierarchical_score = silhouette_score(
    X_scaled,
    hierarchical_labels
)

print("Number of Clusters: 4")
print("Silhouette Score:", round(hierarchical_score, 4))

# ---------------------------------
# 3. DBSCAN PARAMETER TUNING
# ---------------------------------

print("\n========== DBSCAN CLUSTERING ==========")

best_score = -1
best_labels = None
best_eps = None

eps_values = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

dbscan_results = []

for eps_value in eps_values:

    model = DBSCAN(
        eps=eps_value,
        min_samples=5
    )

    labels = model.fit_predict(X_scaled)

    clusters = len(set(labels)) - (
        1 if -1 in labels else 0
    )

    noise = list(labels).count(-1)

    score = np.nan

    # Calculate silhouette only when at least two clusters exist
    mask = labels != -1
    valid_labels = labels[mask]

    if len(set(valid_labels)) >= 2:
        score = silhouette_score(
            X_scaled[mask],
            valid_labels
        )

        if score > best_score:
            best_score = score
            best_labels = labels
            best_eps = eps_value

    dbscan_results.append({
        "Epsilon": eps_value,
        "Clusters": clusters,
        "Noise_Points": noise,
        "Silhouette_Score": score
    })

    print(
        "Epsilon:", eps_value,
        "| Clusters:", clusters,
        "| Noise:", noise,
        "| Silhouette:", score
    )

# Select best valid DBSCAN result
if best_labels is not None:

    dbscan_labels = best_labels
    dbscan_score = best_score

    print("\nBest DBSCAN Epsilon:", best_eps)
    print("Best Silhouette Score:", round(best_score, 4))

else:

    # No tested setting produced two valid clusters
    dbscan = DBSCAN(
        eps=3.0,
        min_samples=5
    )

    dbscan_labels = dbscan.fit_predict(X_scaled)
    dbscan_score = np.nan

    print("\nNo tested epsilon produced two valid clusters.")
    print("DBSCAN result will be reported without a silhouette score.")

daily_info["DBSCAN_Cluster"] = dbscan_labels

number_of_clusters = len(set(dbscan_labels)) - (
    1 if -1 in dbscan_labels else 0
)

number_of_noise = list(dbscan_labels).count(-1)

print("Final DBSCAN Clusters:", number_of_clusters)
print("Final DBSCAN Noise Points:", number_of_noise)

# Save DBSCAN tuning results
pd.DataFrame(dbscan_results).to_csv(
    "outputs/dbscan_parameter_comparison.csv",
    index=False
)

# ---------------------------------
# SAVE CLUSTER RESULTS
# ---------------------------------

daily_info.to_csv(
    "outputs/daily_clustering_results.csv"
)

# K-Means cluster summary
summary = daily_info.groupby("KMeans_Cluster").agg(
    Number_of_Days=("daily_demand", "count"),
    Average_Daily_Demand=("daily_demand", "mean")
)

summary.to_csv(
    "outputs/kmeans_cluster_summary.csv"
)

# ---------------------------------
# ALGORITHM COMPARISON
# ---------------------------------

comparison = pd.DataFrame({
    "Algorithm": [
        "K-Means",
        "Hierarchical",
        "DBSCAN"
    ],
    "Number_of_Clusters": [
        len(set(kmeans_labels)),
        len(set(hierarchical_labels)),
        number_of_clusters
    ],
    "Silhouette_Score": [
        kmeans_score,
        hierarchical_score,
        dbscan_score
    ],
    "Noise_Points": [
        0,
        0,
        number_of_noise
    ]
})

print("\n========== CLUSTERING COMPARISON ==========")
print(comparison.to_string(index=False))

comparison.to_csv(
    "outputs/clustering_comparison.csv",
    index=False
)

# ---------------------------------
# PCA VISUALIZATION
# ---------------------------------

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=kmeans_labels,
    cmap="viridis",
    s=25
)

plt.title("K-Means Clustering of Daily Bike Usage Profiles")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Cluster")

plt.tight_layout()

plt.savefig(
    "outputs/kmeans_cluster_plot.png"
)

plt.show()

print("\nClustering completed successfully!")
print("All results saved in the outputs folder.")
