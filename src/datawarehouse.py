
import pandas as pd
import sqlite3
import os

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Load cleaned dataset
df = pd.read_csv("outputs/cleaned_bike_sharing.csv")

# Connect to SQLite database
conn = sqlite3.connect("outputs/bike_sharing_warehouse.db")

print("BIKE SHARING DATA WAREHOUSE")
print("---------------------------")

# Create Time Dimension
time_dim = df[
    ["dteday", "hr", "weekday", "mnth", "yr"]
].drop_duplicates().reset_index(drop=True)

time_dim["time_id"] = range(1, len(time_dim) + 1)

# Create Season Dimension
season_dim = pd.DataFrame({
    "season_id": [1, 2, 3, 4],
    "season_name": [
        "Spring", "Summer", "Fall", "Winter"
    ]
})

# Create Weather Dimension
weather_dim = df[
    ["weathersit", "temp", "atemp", "hum", "windspeed"]
].drop_duplicates().reset_index(drop=True)

weather_dim["weather_id"] = range(1, len(weather_dim) + 1)

# Create Fact Table
fact = df.merge(
    time_dim,
    on=["dteday", "hr", "weekday", "mnth", "yr"]
)

fact = fact.merge(
    weather_dim,
    on=["weathersit", "temp", "atemp", "hum", "windspeed"]
)

fact_table = fact[
    ["time_id", "season", "weather_id",
     "casual", "registered", "cnt"]
].rename(columns={
    "season": "season_id",
    "cnt": "total_demand"
})

# Store tables in SQLite
time_dim.to_sql("Dim_Time", conn, if_exists="replace", index=False)
season_dim.to_sql("Dim_Season", conn, if_exists="replace", index=False)
weather_dim.to_sql("Dim_Weather", conn, if_exists="replace", index=False)
fact_table.to_sql("Fact_Bike_Demand", conn, if_exists="replace", index=False)

print("\nWarehouse Tables Created Successfully!")

print("\nTime Dimension Records:", len(time_dim))
print("Season Dimension Records:", len(season_dim))
print("Weather Dimension Records:", len(weather_dim))
print("Fact Table Records:", len(fact_table))

print("\nFact Table Sample:")
print(fact_table.head())

conn.close()

print("\nDatabase Saved Successfully!")
print("outputs/bike_sharing_warehouse.db")
