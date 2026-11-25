"""
04_explore.py
Create 4 exploratory charts from data/processed/features_hourly.csv and save
them in the figures/ folder.
"""
import matplotlib
matplotlib.use("Agg")           # no window: save images to disk
import matplotlib.pyplot as plt
import pandas as pd

from config import DAYTIME_FIRST_HOUR, DAYTIME_LAST_HOUR, FEATURES_CSV, FIGURES

FIGURES.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(FEATURES_CSV, parse_dates=["hour_local"])

# 1. Average trips per hour of the day: weekdays vs weekends
by_hour = df.groupby(["hour", "is_weekend"])["trips"].mean().unstack()
by_hour.columns = ["Weekday", "Weekend"]
ax = by_hour.plot(figsize=(9, 5), marker="o")
ax.set_title("Average BIXI trips by hour of day")
ax.set_xlabel("Hour of day")
ax.set_ylabel("Trips per hour")
ax.set_xticks(range(0, 24, 2))
plt.tight_layout()
plt.savefig(FIGURES / "1_trips_by_hour.png", dpi=150)
plt.close()

# 2. Trips by temperature (daytime hours, 7 am to 9 pm)
day = df[df["hour"].between(DAYTIME_FIRST_HOUR, DAYTIME_LAST_HOUR)].copy()
day["temp_bin"] = pd.cut(day["TEMP"], bins=range(-5, 40, 5))
by_temp = day.groupby("temp_bin", observed=True)["trips"].mean()
ax = by_temp.plot(kind="bar", figsize=(9, 5), color="steelblue")
ax.set_title("Average trips by temperature (7 am - 9 pm)")
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Trips per hour")
plt.tight_layout()
plt.savefig(FIGURES / "2_trips_vs_temperature.png", dpi=150)
plt.close()

# 3. Rain vs no rain (daytime hours)
day["rain"] = day["PRECIP_AMOUNT"] > 0
by_rain = day.groupby("rain")["trips"].mean()
by_rain.index = ["No rain", "Rain"]
ax = by_rain.plot(kind="bar", figsize=(6, 5), color=["#f4a261", "#2a9d8f"], rot=0)
ax.set_title("Effect of rain on trips (7 am - 9 pm)")
ax.set_ylabel("Trips per hour")
plt.tight_layout()
plt.savefig(FIGURES / "3_rain_effect.png", dpi=150)
plt.close()

# 4. Total trips per day
daily = df.set_index("hour_local")["trips"].resample("D").sum()
ax = daily.plot(figsize=(11, 5), color="steelblue")
ax.set_title("BIXI trips per day (April - August 2026)")
ax.set_xlabel("")
ax.set_ylabel("Trips per day")
plt.tight_layout()
plt.savefig(FIGURES / "4_trips_per_day.png", dpi=150)
plt.close()

# Key numbers for the README
print("Correlation with number of trips (daytime hours):")
print(day[["TEMP", "PRECIP_AMOUNT", "WIND_SPEED", "RELATIVE_HUMIDITY", "trips"]]
      .corr()["trips"].drop("trips").round(2).to_string())
print(f"\nAverage with rain   : {by_rain['Rain']:.0f} trips/hour")
print(f"Average without rain: {by_rain['No rain']:.0f} trips/hour")
print(f"\nCharts saved in: {FIGURES}")
