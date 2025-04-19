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

        st.header("2. Manual Entry of Lactate & Performance Data")
        baseline_lactate = st.number_input("Baseline Lactate (mmol/L)", value=1.2)
        num_stages = st.number_input("Number of stages", min_value=1, max_value=12, value=6)

        with st.form("lactate_form"):
            lactate_data = []
            for i in range(num_stages):
                st.subheader(f"Stage {i+1}")
                watts = st.number_input(f"Watts Stage {i+1}", key=f"watts_{i}")
                hr = st.number_input(f"Heart Rate Stage {i+1}", key=f"hr_{i}")
                lactate = st.number_input(f"Lactate (mmol/L) Stage {i+1}", key=f"lactate_{i}")
                lactate_data.append([watts, hr, lactate])

            st.subheader("40s Max Effort")
            peak_40s_power = st.number_input("Peak 40s Power (W)")
            peak_lactate_40s = st.number_input("Peak Lactate after 40s (mmol/L)")

            submit = st.form_submit_button("Analyse Physiological Data")

        if submit:
            lactate_df = pd.DataFrame(lactate_data, columns=['Watts','HR','Lactate'])
            lactate_df.insert(0, 'Stage', np.arange(1, num_stages+1))

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(lactate_df['Watts'], lactate_df['Lactate'], marker='o', linewidth=2)
            ax.axhline(4, color='r', linestyle='--', label='LT2 (~4 mmol/L)')
            ax.axhline(2, color='g', linestyle='--', label='LT1 (~2 mmol/L)')
            ax.axhline(baseline_lactate, color='blue', linestyle='--', label='Baseline')
            ax.scatter(peak_40s_power, peak_lactate_40s, color='purple', s=100, label='Peak Lactate (40s Effort)')
            ax.set_xlabel('Power (W)')
            ax.set_ylabel('Lactate (mmol/L)')
            ax.set_title('Detailed Lactate Curve')
            ax.legend()
            st.pyplot(fig)

            lt2_watts = lactate_df.iloc[(lactate_df['Lactate'] - 4).abs().argsort()[:1]]['Watts'].values[0]
            lt1_watts = lactate_df.iloc[(lactate_df['Lactate'] - 2).abs().argsort()[:1]]['Watts'].values[0]

            st.subheader("Advanced Physiological Insights")
            col1, col2, col3 = st.columns(3)
            col1.metric("Peak Lactate (40s)", f"{peak_lactate_40s:.2f} mmol/L")
            col2.metric("LT2 Power (~4 mmol/L)", f"{lt2_watts:.1f} W")
            col3.metric("LT1 Power (~2 mmol/L)", f"{lt1_watts:.1f} W")

            st.markdown("### Elite Integrated Analysis")
            st.markdown(f"- **Anaerobic Capacity (W′)** clearly related to maximal lactate tolerance: {w_prime:.0f} J vs {peak_lactate_40s:.2f} mmol/L.")
            st.markdown(f"- **Critical Power ({cp:.1f} W)** vs **LT2 ({lt2_watts:.1f} W)** highlighting aerobic sustainability.")
            st.markdown(f"- **Baseline to LT1** clearly indicates aerobic efficiency at {lt1_watts:.1f} W.")

    else:
        st.error("Not enough data for analysis (minimum 12 minutes required).")
