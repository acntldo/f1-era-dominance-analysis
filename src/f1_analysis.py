import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# PROJECT PATH SETUP
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "drivers_updated.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# LOAD DATA
df = pd.read_csv(DATA_PATH)

# CLEAN DATA
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df = df.dropna(subset=["year"])


# ERA CREATION
def get_era(year):
    if year <= 2000:
        return "1950–2000"
    elif year <= 2010:
        return "2001–2010"
    elif year <= 2020:
        return "2011–2020"
    else:
        return "2021–2024"

df["era"] = df["year"].apply(get_era)


# CORE ANALYSIS
top_drivers = df.groupby("Driver")["PTS"].sum().sort_values(ascending=False).head(10)
top_teams = df.groupby("Car")["PTS"].sum().sort_values(ascending=False).head(10)
nationality_perf = df.groupby("Nationality")["PTS"].sum().sort_values(ascending=False)
year_trend = df.groupby("year")["PTS"].sum()
era_perf = df.groupby("era")["PTS"].sum()


# SUMMARY METRICS
total_points = df["PTS"].sum()
num_drivers = df["Driver"].nunique()
num_teams = df["Car"].nunique()
num_years = df["year"].nunique()

most_dominant_driver = top_drivers.idxmax()
most_dominant_driver_points = top_drivers.max()

most_dominant_team = top_teams.idxmax()
most_dominant_team_points = top_teams.max()

most_dominant_nationality = nationality_perf.idxmax()
most_dominant_nationality_points = nationality_perf.max()

peak_year = year_trend.idxmax()
peak_year_points = year_trend.max()

most_competitive_era = era_perf.idxmax()
most_competitive_era_points = era_perf.max()


# SAVE RESULTS
results = pd.concat({
    "top_drivers": top_drivers,
    "top_teams": top_teams,
    "top_nationalities": nationality_perf.head(10),
    "year_trend": year_trend,
    "era_performance": era_perf,

    "summary": pd.Series({
        "total_points": total_points,
        "num_drivers": num_drivers,
        "num_teams": num_teams,
        "num_years": num_years,
        "most_dominant_driver": most_dominant_driver,
        "most_dominant_driver_points": most_dominant_driver_points,
        "most_dominant_team": most_dominant_team,
        "most_dominant_team_points": most_dominant_team_points,
        "most_dominant_nationality": most_dominant_nationality,
        "most_dominant_nationality_points": most_dominant_nationality_points,
        "peak_year": peak_year,
        "peak_year_points": peak_year_points,
        "most_competitive_era": most_competitive_era,
        "most_competitive_era_points": most_competitive_era_points
    })
}, axis=0)

results = results.reset_index()
results.columns = ["category", "name", "value"]

results.to_csv(os.path.join(RESULTS_DIR, "analysis_results.csv"),
               index=False, encoding="utf-8")

# STYLE
plt.style.use("seaborn-v0_8-whitegrid")


# 1. YEAR TREND
plt.figure(figsize=(10,5))

year_sorted = year_trend.sort_index()

plt.plot(year_sorted.index, year_sorted.values,
         marker="o", linewidth=2)

plt.title("F1 Performance Over Time", fontweight="bold")
plt.xlabel("Year")
plt.ylabel("Total Points")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "year_trend.png"), dpi=300)
plt.close()


# 2. TOP DRIVERS
plt.figure(figsize=(10,6))

drivers_sorted = top_drivers.sort_values()

plt.barh(drivers_sorted.index, drivers_sorted.values)

plt.title("Top F1 Drivers (Total Points)", fontweight="bold")
plt.xlabel("Points")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "top_drivers.png"), dpi=300)
plt.close()


# 3. TOP TEAMS
plt.figure(figsize=(10,6))

teams_sorted = top_teams.sort_values()

plt.barh(teams_sorted.index, teams_sorted.values)

plt.title("Top F1 Teams (Total Points)", fontweight="bold")
plt.xlabel("Points")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "top_teams.png"), dpi=300)
plt.close()


# 4. ERA PERFORMANCE
plt.figure(figsize=(8,5))

era_sorted = era_perf.reindex(["1950–2000", "2001–2010", "2011–2020", "2021–2024"])

plt.bar(era_sorted.index, era_sorted.values)

plt.title("Performance by F1 Era", fontweight="bold")
plt.ylabel("Total Points")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "era_performance.png"), dpi=300)
plt.close()


# 5. HEATMAP
pivot = df.pivot_table(values="PTS",
                       index="Driver",
                       columns="year",
                       aggfunc="sum",
                       fill_value=0)

top_drivers_heat = pivot.loc[top_drivers.index]

plt.figure(figsize=(12,6))

sns.heatmap(top_drivers_heat,
            cmap="YlOrRd",
            linewidths=0.3)

plt.title("Driver Performance Heatmap Over Years", fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "driver_heatmap.png"), dpi=300)
plt.close()