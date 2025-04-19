import streamlit as st
import pandas as pd
import numpy as np
import fitparse
from io import BytesIO
import matplotlib.pyplot as plt

# Helper functions
def extract_power_from_fit(fitfile):
    power = []
    for record in fitfile.get_messages('record'):
        for data in record:
            if data.name == 'power' and data.value is not None:
                power.append(data.value)
    return power

def peak_power(power_data, duration_seconds):
    rolling_avg = pd.Series(power_data).rolling(window=duration_seconds).mean()
    return rolling_avg.max()

def calculate_cp_w_map(powers, durations=[180,360,720]):
    durations = np.array(durations)
    inverse_durations = 1/durations
    powers = np.array(powers)

    coeffs = np.polyfit(inverse_durations, powers, 1)
    cp = coeffs[1]
    w_prime = coeffs[0]
    map_power = powers[0]

    fractional_utilisation = cp / map_power

    return cp, w_prime, map_power, fractional_utilisation

# Streamlit App
st.set_page_config(layout="wide")
st.title("Elite Cycling Physiology Analysis")

# Athlete Profile
st.sidebar.header("Athlete Profile")
weight = st.sidebar.number_input("Athlete Weight (kg)", value=70.0)

# FIT File Analysis
st.header("1. Upload .fit Files for CP Calculation")
uploaded_files = st.file_uploader("Upload multiple .fit files", accept_multiple_files=True)

if uploaded_files:
    all_power_data = []
    for uploaded_file in uploaded_files:
        fitfile = fitparse.FitFile(BytesIO(uploaded_file.read()))
        power_data = extract_power_from_fit(fitfile)
        all_power_data.extend(power_data)

    if len(all_power_data) > 720:
        p3 = peak_power(all_power_data, 180)
        p6 = peak_power(all_power_data, 360)
        p12 = peak_power(all_power_data, 720)

        cp, w_prime, map_power, fractional_utilisation = calculate_cp_w_map([p3,p6,p12])

        cp_rel = cp / weight
        map_rel = map_power / weight

        st.subheader("Critical Power & Aerobic Profile")
        cols = st.columns(2)
        cols[0].metric("Critical Power (W)", f"{cp:.1f}")
        cols[1].metric("Critical Power (W/kg)", f"{cp_rel:.2f}")
        cols[0].metric("MAP (W)", f"{map_power:.1f}")
        cols[1].metric("MAP (W/kg)", f"{map_rel:.2f}")
        st.metric("W′ (Anaerobic Capacity)", f"{w_prime:.0f} J")

        st.header("2. Upload Lactate & Performance CSV")
        lactate_csv = st.file_uploader("Upload CSV", type=["csv"])

        if lactate_csv:
            lactate_df = pd.read_csv(lactate_csv)
            baseline_lactate = lactate_df['Baseline_Lactate'][0]
            peak_40s_power = lactate_df['Peak_Power_40s'][0]
            peak_lactate_40s = lactate_df['Peak_Lactate_40s'][0]

            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(lactate_df['Watts'], lactate_df['Lactate'], marker='o', linewidth=3, color='blue')
            ax.axhline(4, color='red', linestyle='--', label='LT2 (~4 mmol/L)')
            ax.axhline(2, color='green', linestyle='--', label='LT1 (~2 mmol/L)')
            ax.axhline(baseline_lactate, color='purple', linestyle='--', label='Baseline')
            ax.scatter(peak_40s_power, peak_lactate_40s, color='orange', s=120, label='Peak Lactate (40s)')
            ax.set_xlabel('Power (W)', fontsize=14)
            ax.set_ylabel('Lactate (mmol/L)', fontsize=14)
            ax.set_title('Detailed Lactate Curve', fontsize=16)
            ax.legend()
            st.pyplot(fig)

            lt2_watts = lactate_df.iloc[(lactate_df['Lactate'] - 4).abs().argsort()[:1]]['Watts'].values[0]
            lt1_watts = lactate_df.iloc[(lactate_df['Lactate'] - 2).abs().argsort()[:1]]['Watts'].values[0]

            st.subheader("Elite Integrated Physiological Insights")
            col1, col2, col3 = st.columns(3)
            col1.metric("Peak Lactate (40s)", f"{peak_lactate_40s:.2f} mmol/L")
            col2.metric("LT2 Power (~4 mmol/L)", f"{lt2_watts:.1f} W")
            col3.metric("LT1 Power (~2 mmol/L)", f"{lt1_watts:.1f} W")

            st.markdown("---")
            st.markdown("### World-Class Physiological Summary")
            st.markdown(f"- **Anaerobic Capacity (W′)** vs. **Peak Lactate**: `{w_prime:.0f} J` vs `{peak_lactate_40s:.2f} mmol/L`.")
            st.markdown(f"- **Critical Power ({cp:.1f} W)** aligns with **LT2 ({lt2_watts:.1f} W)**, clearly defining aerobic sustainability.")
            st.markdown(f"- **Aerobic efficiency** (Baseline to LT1) clearly observed at `{lt1_watts:.1f} W` (~{(lt1_watts/cp)*100:.1f}% CP).")

    else:
        st.error("Not enough data for analysis (minimum 12 minutes required).")
