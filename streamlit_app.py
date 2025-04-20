import streamlit as st
import pandas as pd
import numpy as np
import fitparse
from io import BytesIO
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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

# Interpolation function for lactate thresholds
def interpolate_threshold(df, target):
    for i in range(1, len(df)):
        if df['Lactate'][i] >= target:
            x0, y0 = df['Watts'][i-1], df['Lactate'][i-1]
            x1, y1 = df['Watts'][i], df['Lactate'][i]
            return x0 + (target - y0) * (x1 - x0) / (y1 - y0)

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
        fig_cp = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = cp,
            title = {'text': "Critical Power (W)"},
            gauge = {'axis': {'range': [None, map_power*1.2]}, 'bar': {'color': "blue"}}
        ))
        st.plotly_chart(fig_cp)

        fig_wprime = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = w_prime,
            title = {'text': "W′ (J)"},
            gauge = {'axis': {'range': [0, 30000]}, 'bar': {'color': "purple"}}
        ))
        st.plotly_chart(fig_wprime)

        st.header("2. Upload Lactate & Performance CSV")
        lactate_csv = st.file_uploader("Upload CSV", type=["csv"])

        if lactate_csv:
            lactate_df = pd.read_csv(lactate_csv)
            baseline_lactate = lactate_df['Baseline_Lactate'][0]
            peak_40s_power = lactate_df['Peak_Power_40s'][0]
            peak_lactate_40s = lactate_df['Peak_Lactate_40s'][0]

            lt1_watts = interpolate_threshold(lactate_df, 2.0)
            lt2_watts = interpolate_threshold(lactate_df, 4.0)

            hr_lt1 = np.interp(lt1_watts, lactate_df['Watts'], lactate_df['Heart_Rate'])
            hr_lt2 = np.interp(lt2_watts, lactate_df['Watts'], lactate_df['Heart_Rate'])

            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(lactate_df['Watts'], lactate_df['Lactate'], marker='o', linewidth=3, color='blue')
            ax.axhline(4, color='red', linestyle='--', label='LT2 (~4 mmol/L)')
            ax.axhline(2, color='green', linestyle='--', label='LT1 (~2 mmol/L)')
            ax.scatter([lt1_watts, lt2_watts], [2,4], color='black', s=100, label='Interpolated LT1 & LT2')
            ax.set_xlabel('Power (W)', fontsize=14)
            ax.set_ylabel('Lactate (mmol/L)', fontsize=14)
            ax.set_title('Lactate Curve with Stage Data', fontsize=16)
            ax.legend()
            st.pyplot(fig)

            st.subheader("Final Elite Integrated Analysis")
            cols = st.columns(3)
            cols[0].metric("CP vs LT2", f"{cp:.1f} W vs {lt2_watts:.1f} W")
            cols[1].metric("W′ vs Peak Lactate", f"{w_prime:.0f} J vs {peak_lactate_40s:.1f} mmol/L")
            cols[2].metric("LT1 as % CP", f"{(lt1_watts/cp)*100:.1f}%")

            fig_intensity = go.Figure()
            fig_intensity.add_trace(go.Bar(x=['Moderate','Heavy','Severe','Extreme'], 
                                           y=[lt1_watts, lt2_watts-lt1_watts, map_power-lt2_watts, map_power*0.2],
                                           marker_color=['green','yellow','orange','red']))
            fig_intensity.update_layout(title='Intensity Domains & Phase Transitions', xaxis_title='Domain', yaxis_title='Power (W)')
            st.plotly_chart(fig_intensity)
    else:
        st.error("Not enough data for analysis (minimum 12 minutes required).")
