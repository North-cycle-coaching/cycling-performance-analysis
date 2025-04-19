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
st.title("Cycling Performance & Lactate Analysis")

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

        st.subheader("Critical Power Analysis")
        st.write(f"**3-min Peak Power:** {p3:.1f} W")
        st.write(f"**6-min Peak Power:** {p6:.1f} W")
        st.write(f"**12-min Peak Power:** {p12:.1f} W")
        st.write(f"**Critical Power (CP):** {cp:.1f} W")
        st.write(f"**W′:** {w_prime:.1f} J")
        st.write(f"**MAP:** {map_power:.1f} W")
        st.write(f"**Fractional Utilisation:** {fractional_utilisation:.2%}")

        st.header("2. Suggested Test Stages")
        increments = [0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.05,1.1]
        stages = [cp * inc for inc in increments]
        stages_df = pd.DataFrame({"Stage": range(1,len(stages)+1), "Wattage (W)": np.round(stages,1)})
        st.table(stages_df)

        st.header("3. Manual Entry of Lactate Test Data")
        num_stages = st.number_input("Number of stages completed", min_value=1, max_value=12, value=6)

        with st.form("lactate_form"):
            lactate_data = []
            for i in range(num_stages):
                st.subheader(f"Stage {i+1}")
                watts = st.number_input(f"Watts Stage {i+1}", key=f"watts_{i}")
                hr = st.number_input(f"Heart Rate Stage {i+1}", key=f"hr_{i}")
                lactate = st.number_input(f"Lactate (mmol/L) Stage {i+1}", key=f"lactate_{i}")
                lactate_data.append([watts, hr, lactate])

            submit = st.form_submit_button("Analyse Lactate Data")

        if submit:
            lactate_df = pd.DataFrame(lactate_data, columns=['Watts','HR','Lactate'])
            st.table(lactate_df)

            st.header("4. Lactate Threshold Analysis")
            fig, ax = plt.subplots()
            ax.plot(lactate_df['Watts'], lactate_df['Lactate'], marker='o')
            ax.set_xlabel('Watts')
            ax.set_ylabel('Lactate (mmol/L)')
            ax.set_title('Lactate Curve')
            st.pyplot(fig)

            st.header("5. Combined Insights")
            peak_lactate = lactate_df['Lactate'].max()
            lt2_watts = lactate_df.iloc[(lactate_df['Lactate'] - 4).abs().argsort()[:1]]['Watts'].values[0]
            lt1_watts = lactate_df.iloc[(lactate_df['Lactate'] - 2).abs().argsort()[:1]]['Watts'].values[0]

            st.write(f"**Peak Lactate:** {peak_lactate:.2f} mmol/L")
            st.write(f"**LT2 Watts (~4 mmol/L):** {lt2_watts:.1f} W")
            st.write(f"**LT1 Watts (~2 mmol/L):** {lt1_watts:.1f} W")

            st.subheader("Integrated Analysis")
            st.write(f"- **W′ ({w_prime:.0f} J) vs Peak Lactate ({peak_lactate:.2f} mmol/L)**")
            st.write(f"- **CP ({cp:.1f} W) vs LT2 ({lt2_watts:.1f} W)** Difference: {cp - lt2_watts:.1f} W")
            st.write(f"- **LT1 ({lt1_watts:.1f} W) vs Fractional Utilisation ({fractional_utilisation:.2%})**")

    else:
        st.error("Not enough data for analysis (minimum 12 minutes required).")
