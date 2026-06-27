"""
Spotify Charts analytics dashboard (Streamlit).
Deploy free on Streamlit Community Cloud.

Run locally:
    streamlit run app/dashboard.py
"""
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Spotify Charts Explorer", layout="wide")

CHARTS_CSV = "data/sample_charts.csv"  # swap to data/full/charts.csv for the real run


@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


st.title("🎵 Spotify Charts Explorer")
st.caption("Companion analytics dashboard for the lyrics genre/mood classifier project.")

try:
    df = load_data(CHARTS_CSV)
except FileNotFoundError:
    st.error(f"Could not find {CHARTS_CSV}. See data/README.md to download the full dataset.")
    st.stop()

# ---- Filters ----
regions = sorted(df["region"].dropna().unique())
region = st.sidebar.selectbox("Region", regions, index=0)
sub = df[df["region"] == region]

c1, c2, c3 = st.columns(3)
c1.metric("Tracks tracked", f"{sub['track'].nunique():,}")
c2.metric("Artists", f"{sub['artist'].nunique():,}")
c3.metric("Total streams", f"{int(sub['streams'].sum()):,}")

# ---- Top artists ----
st.subheader("Top artists by total streams")
top = (sub.groupby("artist")["streams"].sum()
       .sort_values(ascending=False).head(10).reset_index())
st.plotly_chart(px.bar(top, x="streams", y="artist", orientation="h"),
                use_container_width=True)

# ---- Streams over time ----
st.subheader("Daily streams over time")
daily = sub.groupby("date")["streams"].sum().reset_index()
st.plotly_chart(px.line(daily, x="date", y="streams"), use_container_width=True)

# ---- Track explorer ----
st.subheader("Track explorer")
track = st.selectbox("Pick a track", sorted(sub["track"].unique()))
tdf = sub[sub["track"] == track].sort_values("date")
st.plotly_chart(px.line(tdf, x="date", y="rank", title=f"Chart rank — {track}")
                .update_yaxes(autorange="reversed"), use_container_width=True)
