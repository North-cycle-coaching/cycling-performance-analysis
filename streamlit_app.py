import streamlit as st
import pandas as pd
from fitparse import FitFile
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cycling Race File Analysis", layout="wide")
st.title("Cycling Race Analysis Tool")

# --- Sidebar Inputs ---
st.sidebar.header("Upload & Rider Info")
fit_file = st.sidebar.file_uploader("Upload .fit file")
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
    if 'position_lat' in df.columns:
        df['position_lat'] = df['position_lat'].apply(semicircles_to_degrees)
    if 'position_long' in df.columns:
        df['position_long'] = df['position_long'].apply(semicircles_to_degrees)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['seconds'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds().astype(int)
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
    return round(ris, 1), df

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

def get_quarter_data(df, body_weight, cp):
    total_secs = df['seconds'].iloc[-1]
    split_points = [0, total_secs//4, total_secs//2, 3*total_secs//4, total_secs+1]
    quarters = []
    for i in range(4):
        qdf = df[(df['seconds'] >= split_points[i]) & (df['seconds'] < split_points[i+1])]
        duration = len(qdf)
        kjs = qdf['power'].fillna(0).sum() / 1000
        kj_per_kg = round(kjs / body_weight, 2)
        ris, _ = calculate_race_impact(qdf, cp)
        avg_power = int(qdf['power'].mean()) if not qdf['power'].isna().all() else 0
        quarters.append({
            'Quarter': f'Q{i+1}',
            'Duration (s)': duration,
            'Avg Power (W)': avg_power,
            'kJ/kg': kj_per_kg,
            'RIS': ris
        })
    return pd.DataFrame(quarters)

def detect_matches(df, cp):
    threshold = 1.2 * cp
    above = df['power'].fillna(0) >= threshold
    df['match_group'] = (above != above.shift()).cumsum()
    matches = df[above].groupby('match_group').filter(lambda x: len(x) >= 10)
    return matches['match_group'].nunique()

def estimate_cp_wprime(peaks):
    try:
        p3 = peaks['180s']
        p12 = peaks['720s']
        if p3 is None or p12 is None:
            return None, None
        cp = int((p12 * 180 - p3 * 720) / (180 - 720))
        wprime = int((p3 - cp) * 180)
        return cp, wprime
    except:
        return None, None

def detect_climbs(df):
    if 'position_lat' not in df or 'altitude' not in df:
        return pd.DataFrame()
    df = df[['timestamp', 'position_lat', 'position_long', 'altitude']].dropna().copy()
    df['delta_alt'] = df['altitude'].diff().fillna(0)
    df['delta_dist'] = 0.0001  # assume ~10m step if GPS missing
    df['gradient'] = (df['delta_alt'] / (df['delta_dist'] * 1000)) * 100
    df['climb'] = df['gradient'] > 3
    df['climb_group'] = (df['climb'] != df['climb'].shift()).cumsum()
    climbs = df[df['climb']].groupby('climb_group').filter(lambda g: len(g) >= 30)
    results = []
    for name, g in climbs.groupby('climb_group'):
        results.append({
            'Start': g['timestamp'].iloc[0],
            'End': g['timestamp'].iloc[-1],
            'Duration (s)': (g['timestamp'].iloc[-1] - g['timestamp'].iloc[0]).total_seconds(),
            'Elevation Gain (m)': round(g['altitude'].iloc[-1] - g['altitude'].iloc[0], 1),
            'Avg Gradient (%)': round(g['gradient'].mean(), 2),
        })
    return pd.DataFrame(results)

# --- Main Logic ---
if fit_file:
    if not fit_file.name.lower().endswith('.fit'):
        st.error("Please upload a .fit file.")
    elif critical_power and body_weight:
        df = parse_fit(fit_file)
        ris, df = calculate_race_impact(df, critical_power)
        peak_durations = [180, 300, 720]
        peaks = get_peak_power(df, peak_durations)
        map_5min = peaks['300s']
        est_cp, est_wprime = estimate_cp_wprime(peaks)

        st.subheader("Hero Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Input CP", f"{critical_power} W")
        col2.metric("MAP (5-min)", f"{map_5min} W")
        col3.metric("Race Impact Score", f"{ris}")
        col4.metric("Matches (120% CP)", detect_matches(df, critical_power))
        if est_cp and est_wprime:
            col5.metric("Modelled W′", f"{est_wprime} J")

        st.divider()

        st.subheader("Quarter Breakdown")
        st.dataframe(get_quarter_data(df, body_weight, critical_power))

        st.divider()

        st.subheader("Peak Powers (Whole Race)")
        all_peaks = get_peak_power(df, [1, 10, 30, 60, 180, 300, 720])
        st.table(pd.DataFrame.from_dict(all_peaks, orient='index', columns=['Watts']))

        st.divider()

        st.subheader("Time in Zone")
        zone_counts = df['zone'].value_counts().sort_index()
        fig, ax = plt.subplots()
        zone_counts.plot(kind='bar', ax=ax)
        ax.set_ylabel("Seconds")
        ax.set_title("Time in Power Zones")
        st.pyplot(fig)

        st.divider()

        st.subheader("Detected Climbs")
        climbs_df = detect_climbs(df)
        if not climbs_df.empty:
            st.dataframe(climbs_df)
        else:
            st.write("No climbs detected (requires altitude and GPS data).")

        st.divider()

        st.subheader("Raw Data Preview")
        st.dataframe(df[['timestamp', 'power', 'heart_rate', 'cadence', 'speed', 'altitude']].head(200))

    else:
        st.info("Please enter your body weight and critical power.")
else:
    st.info("Upload a .fit file to begin analysis.")
