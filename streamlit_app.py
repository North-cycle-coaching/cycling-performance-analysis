import streamlit as st
import pandas as pd
from fitparse import FitFile
import numpy as np

st.set_page_config(page_title="Cycling Race File Analysis", layout="wide")
st.title("Cycling Race Analysis Tool")

# --- Sidebar Inputs ---
st.sidebar.header("Upload & Rider Info")
fit_file = st.sidebar.file_uploader("Upload .fit file")  # Removed strict type check
body_weight = st.sidebar.number_input("Body weight (kg)", min_value=30.0, max_value=120.0, step=0.5)
critical_power = st.sidebar.number_input("Critical Power (W)", min_value=100, max_value=500, step=1)

# --- Helper functions ---
def semicircles_to_degrees(semicircles):
    return semicircles * (180 / 2**31)

def parse_fit(f):
    fitfile = FitFile(f)
    records = []
    for record in fitfile.get_messages("record"):
        r = {}
        for field in record:
            r[field.name] = field.value
        records.append(r)
    df = pd.DataFrame(records)

    # Convert GPS
    if 'position_lat' in df.columns:
        df['position_lat'] = df['position_lat'].apply(semicircles_to_degrees)
    if 'position_long' in df.columns:
        df['position_long'] = df['position_long'].apply(semicircles_to_degrees)

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def calculate_race_impact(df, cp):
    df['zone'] = pd.cut(
        df['power'],
        bins=[0, 0.95*cp, cp, 1.1*cp, 1.2*cp, 1.3*cp, np.inf],
        labels=["<95%", "95-100%", "100-110%", "110-120%", "120-130%", ">130%"]
    )
    weights = {
        "95-100%": 1/60,
        "100-110%": 1.5/60,
        "110-120%": 2.5/60,
        "120-130%": 4/60,
        ">130%": 6/60
    }
    ris = 0
    for zone, group in df.groupby("zone"):
        if zone in weights:
            ris += len(group) * weights[zone]
    return round(ris, 1)

def get_peak_power(df, durations):
    peaks = {}
    power = df['power'].fillna(0).to_numpy()
    for d in durations:
        window = int(d)
        if len(power) >= window:
            rolling = pd.Series(power).rolling(window).mean()
            peaks[f"{d}s"] = int(rolling.max())
        else:
            peaks[f"{d}s"] = None
    return peaks

# --- Main Logic ---
if fit_file:
    if not fit_file.name.lower().endswith('.fit'):
        st.error("Please upload a .fit file.")
    elif critical_power and body_weight:
        df = parse_fit(fit_file)

        st.subheader("Hero Metrics")
        peak_5min = get_peak_power(df, [300])["300s"]
        ris = calculate_race_impact(df, critical_power)

        col1, col2, col3 = st.columns(3)
        col1.metric("Critical Power", f"{critical_power} W")
        col2.metric("MAP (5-min)", f"{peak_5min} W")
        col3.metric("Race Impact Score", f"{ris}")

        st.divider()

        st.subheader("Peak Powers")
        durations = [1, 10, 30, 60, 180, 300, 720]
        peaks = get_peak_power(df, durations)
        st.table(pd.DataFrame.from_dict(peaks, orient='index', columns=['Watts']))

        st.divider()

        st.subheader("Raw Data Preview")
        st.dataframe(df[['timestamp', 'power', 'heart_rate', 'cadence', 'speed', 'altitude']].head(200))

    else:
        st.info("Please enter your body weight and critical power.")
else:
    st.info("Upload a .fit file to begin analysis.")
