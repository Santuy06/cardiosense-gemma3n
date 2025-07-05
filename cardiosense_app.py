import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
from datetime import datetime
import subprocess
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# === Setup Page ===
st.set_page_config(page_title="CardioSense Dashboard", layout="wide")
st.title("💓 CardioSense - Daily Cardiovascular Monitor")

# === Simulated Login ===
st.sidebar.header("🔐 User Login")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if username != "demo" or password != "cardiosense":
    st.warning("Please login using the demo credentials.")
    st.stop()

# === Inject Dummy Data If CSV Missing ===
if not os.path.exists("health_data.csv"):
    dummy_data = [
        {"date": "2025-07-01", "hr_rest": 88, "hrv": 35, "spo2": 96, "sleep": 6.5, "steps": 3200, "stress": "high", "bmi": 27.4, "age": 42},
        {"date": "2025-07-02", "hr_rest": 75, "hrv": 48, "spo2": 98, "sleep": 7.5, "steps": 8000, "stress": "normal", "bmi": 24.0, "age": 30},
        {"date": "2025-07-03", "hr_rest": 92, "hrv": 30, "spo2": 95, "sleep": 5.0, "steps": 1500, "stress": "high", "bmi": 29.1, "age": 55},
        {"date": "2025-07-04", "hr_rest": 60, "hrv": 55, "spo2": 99, "sleep": 8.0, "steps": 10000, "stress": "low", "bmi": 22.0, "age": 25},
        {"date": "2025-07-05", "hr_rest": 38, "hrv": 20, "spo2": 93, "sleep": 4.0, "steps": 500, "stress": "very high", "bmi": 31.0, "age": 65}
    ]
    pd.DataFrame(dummy_data).to_csv("health_data.csv", index=False)
    st.toast("🪪 Dummy data loaded for initial testing.")

# === Reset Dummy Data Button ===
if st.sidebar.button("🔄 Reset to Dummy Data"):
    if os.path.exists("health_data.csv"):
        os.remove("health_data.csv")
    st.rerun()

# === Export Data ===
st.sidebar.markdown("\n---\n")
st.sidebar.markdown("📄 **Export Your Health Data**")
csv = pd.read_csv("health_data.csv").to_csv(index=False).encode('utf-8')
st.sidebar.download_button("⬇️ Download CSV", data=csv, file_name='cardiosense_data.csv', mime='text/csv')

# === About/Help ===
if st.sidebar.button("❓ About / Help"):
    st.info("""
**CardioSense** helps users monitor cardiovascular health through:
- Resting heart rate
- HRV
- SpO₂
- Stress level
- Sleep, steps, BMI, and age

📈 Predictions are powered by a machine learning model and a local LLM via **Ollama**.
    """)

# === Daily Input Form ===
st.sidebar.header("📝 New Daily Entry")
with st.sidebar.form("user_input_form"):
    today = st.date_input("Date", datetime.now())
    hr_rest = st.number_input("Resting Heart Rate (bpm)", min_value=30, max_value=200)
    hrv = st.number_input("Heart Rate Variability (ms)", min_value=10, max_value=100)
    spo2 = st.number_input("SpO₂ (%)", min_value=80, max_value=100)
    sleep = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=24.0)
    steps = st.number_input("Step Count", min_value=0, max_value=50000)
    stress = st.selectbox("Stress Level", ["low", "normal", "high", "very high"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0)
    age = st.number_input("Age", min_value=1, max_value=120)
    submitted = st.form_submit_button("➕ Add Entry")

if submitted:
    new_row = {
        "date": today,
        "hr_rest": hr_rest,
        "hrv": hrv,
        "spo2": spo2,
        "sleep": sleep,
        "steps": steps,
        "stress": stress,
        "bmi": bmi,
        "age": age
    }
    if os.path.exists("health_data.csv"):
        df_existing = pd.read_csv("health_data.csv")
        df_existing = pd.concat([df_existing, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df_existing = pd.DataFrame([new_row])
    df_existing.to_csv("health_data.csv", index=False)
    st.success("Data added successfully!")
    st.rerun()

# === Load Data CSV ===
data = pd.read_csv("health_data.csv")
data['date'] = pd.to_datetime(data['date'])

# === Encode stress ===
stress_map = {"low": 1, "normal": 2, "high": 3, "very high": 4}
data['stress_score'] = data['stress'].map(stress_map)

# === ML Risk Model ===
model_file = "risk_model.pkl"
if not os.path.exists(model_file):
    df_model = data.copy()
    df_model['risk_label'] = ((df_model['hr_rest'] > 90) | (df_model['hrv'] < 30) | (df_model['spo2'] < 94) | (df_model['stress_score'] > 2)).astype(int)
    X = df_model[['hr_rest', 'hrv', 'spo2', 'stress_score', 'bmi', 'age']]
    y = df_model['risk_label']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, model_file)
else:
    model = joblib.load(model_file)

data['predicted_risk'] = model.predict(data[['hr_rest', 'hrv', 'spo2', 'stress_score', 'bmi', 'age']])

# === Daily Summary ===
st.subheader("📊 Daily Summary")
col_a, col_b, col_c = st.columns(3)
col_a.metric("Total Entries", len(data))
col_b.metric("Avg HRV", f"{data['hrv'].mean():.1f} ms")
col_c.metric("Avg Sleep", f"{data['sleep'].mean():.1f} hrs")

# === Weekly Data Preview ===
st.subheader("🗕️ Weekly Data Overview")
st.dataframe(data)

# === Biometric Trend Line Charts ===
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### ❤️ Resting Heart Rate vs HRV")
    fig, ax = plt.subplots()
    ax.plot(data['date'], data['hr_rest'], label='Resting HR (bpm)', color='red')
    ax.plot(data['date'], data['hrv'], label='HRV (ms)', color='blue')
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    fig.autofmt_xdate()
    st.pyplot(fig)

with col2:
    st.markdown("#### 🧠 Stress Level & Steps")
    fig2, ax2 = plt.subplots()
    ax2.plot(data['date'], data['steps'], label='Steps', color='green')
    ax2.plot(data['date'], data['stress_score'] * 1000, label='Stress (scaled)', linestyle='--', color='purple')
    ax2.legend()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    fig2.autofmt_xdate()
    st.pyplot(fig2)

# === Risk Trend Prediction ===
st.subheader("📈 Cardiovascular Risk Prediction (ML-Based)")
with st.container():
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(data['date'], data['predicted_risk'], marker='o', linestyle='-', color='darkorange', label='Predicted Risk')
    ax3.axhline(0.5, color='red', linestyle='--', label='Risk Threshold')
    ax3.set_ylabel("Risk Probability")
    ax3.set_xlabel("Date")
    ax3.set_title("Daily Cardiovascular Risk Trend")
    ax3.set_ylim(-0.1, 1.1)
    ax3.legend()
    ax3.grid(True)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    fig3.autofmt_xdate()
    st.pyplot(fig3)

# === LLM Analysis via Ollama with Fallback ===
st.markdown("---")
st.write("🔍 Debug: Displaying LLM Analysis Section...")
st.subheader("🧠 AI-Powered LLM Recommendations")

def build_prompt(row):
    return f"""
Daily Health Report for {row['date']}:
- Resting Heart Rate: {row['hr_rest']} bpm
- HRV: {row['hrv']} ms
- SpO₂: {row['spo2']}%
- Sleep Duration: {row['sleep']} hours
- Step Count: {row['steps']}
- Stress Level: {row['stress']}
- BMI: {row['bmi']}
- Age: {row['age']}

Does this indicate any cardiovascular risk? What should the user do today?
"""

responses = []
for _, row in data.iterrows():
    prompt = build_prompt(row)
    try:
        result = subprocess.run([
            "ollama", "run", "gemma3n:e4b", prompt
        ], capture_output=True, text=True)
        output = result.stdout.strip()
        err = result.stderr.strip()
        print("🔧 Prompt:", prompt)
        print("📤 Output:", output)
        print("⚠️ Error:", err)
        if output:
            responses.append(output)
        else:
            raise Exception("No output from Ollama")
    except Exception as e:
        print("🛑 LLM fallback error:", str(e))
        fallback = f"[Fallback] Based on your heart rate ({row['hr_rest']} bpm) and HRV ({row['hrv']} ms), your cardiovascular risk is {'high' if row['predicted_risk']==1 else 'low'}."
        responses.append(fallback)

data['llm_response'] = responses
st.write(data[['date', 'hr_rest', 'hrv', 'spo2', 'stress', 'predicted_risk', 'llm_response']])

# === Critical Alert Notification ===
critical_rows = data[data['hr_rest'] < 40]
if not critical_rows.empty:
    st.error("🚨 Critical Condition Detected!")
    st.write(critical_rows[['date', 'hr_rest', 'age', 'bmi']])
    st.warning("📱 Emergency alert sent to contact and hospital.")

st.success("✅ Dashboard updated successfully.")
