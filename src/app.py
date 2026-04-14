import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# PATH SETUP
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "drivers_updated.csv")

# LOAD DATA
df = pd.read_csv(DATA_PATH)

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df = df.dropna(subset=["year"])

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

# SIDEBAR FILTERS
st.sidebar.title("🎛️ Filters")

year_min, year_max = int(df["year"].min()), int(df["year"].max())

year_range = st.sidebar.slider("Year Range", year_min, year_max, (year_min, year_max))

drivers = st.sidebar.multiselect("Drivers", df["Driver"].unique())
teams = st.sidebar.multiselect("Teams", df["Car"].unique())

# FILTER DATA
filtered_df = df[
    (df["year"] >= year_range[0]) &
    (df["year"] <= year_range[1])
]

if drivers:
    filtered_df = filtered_df[filtered_df["Driver"].isin(drivers)]

if teams:
    filtered_df = filtered_df[filtered_df["Car"].isin(teams)]

# ANALYSIS (FILTERED)
top_drivers = filtered_df.groupby("Driver")["PTS"].sum().sort_values(ascending=False).head(10)
top_teams = filtered_df.groupby("Car")["PTS"].sum().sort_values(ascending=False).head(10)
year_trend = filtered_df.groupby("year")["PTS"].sum()

# TABS
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Overview", "🏆 Drivers", "🏎️ Teams", "📈 Trends"])

# OVERVIEW
with tab1:
    st.title("🏁 F1 Era Dominance Dashboard")

    st.metric("Total Points", int(filtered_df["PTS"].sum()))
    st.metric("Drivers", filtered_df["Driver"].nunique())
    st.metric("Teams", filtered_df["Car"].nunique())
    st.metric("Years Covered", filtered_df["year"].nunique())

    st.subheader("🧠 Quick Insight")

    if not top_drivers.empty:
        st.write(f"Most dominant driver in selection: **{top_drivers.idxmax()}**")

    if not top_teams.empty:
        st.write(f"Most dominant team in selection: **{top_teams.idxmax()}**")

# DRIVERS
with tab2:
    st.subheader("🏆 Top Drivers")

    if not top_drivers.empty:
        st.bar_chart(top_drivers)

# TEAMS
with tab3:
    st.subheader("🏎️ Top Teams")

    if not top_teams.empty:
        st.bar_chart(top_teams)

# TRENDS
with tab4:
    st.subheader("📈 Performance Over Time")

    if not year_trend.empty:
        st.line_chart(year_trend)

    st.subheader("🔥 Era Breakdown")

    era_perf = filtered_df.groupby("era")["PTS"].sum()
    st.bar_chart(era_perf)

# HEATMAP (BELOW TABS)
st.subheader("🔥 Driver Consistency Heatmap")

if len(filtered_df) > 0:
    pivot = filtered_df.pivot_table(values="PTS",
                                    index="Driver",
                                    columns="year",
                                    aggfunc="sum",
                                    fill_value=0)

    top_heat = pivot.loc[top_drivers.index] if not top_drivers.empty else pivot.head(5)

    fig, ax = plt.subplots(figsize=(12,6))
    sns.heatmap(top_heat, cmap="YlOrRd", ax=ax)

    st.pyplot(fig)