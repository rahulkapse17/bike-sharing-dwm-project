
import sqlite3
import pandas as pd

# Connect to the Bike Sharing Data Warehouse
conn = sqlite3.connect("outputs/bike_sharing_warehouse.db")

print("BIKE SHARING OLAP OPERATIONS")
print("----------------------------")

# 1. ROLL-UP: Total demand by season
print("\n1. ROLL-UP: Demand by Season")

rollup = pd.read_sql_query("""
SELECT season_id,
       SUM(total_demand) AS total_rentals
FROM Fact_Bike_Demand
GROUP BY season_id
ORDER BY season_id
""", conn)

print(rollup)
rollup.to_csv("outputs/olap_rollup.csv", index=False)


# 2. DRILL-DOWN: Demand by month and hour
print("\n2. DRILL-DOWN: Demand by Month and Hour")

drilldown = pd.read_sql_query("""
SELECT t.mnth,
       t.hr,
       SUM(f.total_demand) AS total_rentals
FROM Fact_Bike_Demand f
JOIN Dim_Time t ON f.time_id = t.time_id
GROUP BY t.mnth, t.hr
ORDER BY t.mnth, t.hr
""", conn)

print(drilldown.head(20))
drilldown.to_csv("outputs/olap_drilldown.csv", index=False)


# 3. SLICE: Demand for one season
print("\n3. SLICE: Demand in Season 1")

slice_result = pd.read_sql_query("""
SELECT season_id,
       SUM(casual) AS casual_rentals,
       SUM(registered) AS registered_rentals,
       SUM(total_demand) AS total_rentals
FROM Fact_Bike_Demand
WHERE season_id = 1
GROUP BY season_id
""", conn)

print(slice_result)
slice_result.to_csv("outputs/olap_slice.csv", index=False)


# 4. DICE: Filter by season, hour and weather
print("\n4. DICE: Selected Seasons, Hours and Weather")

dice = pd.read_sql_query("""
SELECT f.season_id,
       t.hr,
       w.weathersit,
       SUM(f.total_demand) AS total_rentals
FROM Fact_Bike_Demand f
JOIN Dim_Time t ON f.time_id = t.time_id
JOIN Dim_Weather w ON f.weather_id = w.weather_id
WHERE f.season_id IN (1, 2)
AND t.hr BETWEEN 8 AND 17
AND w.weathersit IN (1, 2)
GROUP BY f.season_id, t.hr, w.weathersit
ORDER BY f.season_id, t.hr
""", conn)

print(dice.head(20))
dice.to_csv("outputs/olap_dice.csv", index=False)

conn.close()

print("\nAll OLAP Operations Completed Successfully!")
print("Results saved in the outputs folder.")
