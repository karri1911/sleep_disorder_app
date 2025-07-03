import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# Load saved model and preprocessing tools with error handling
try:
    model = tf.keras.models.load_model("sleep_model.h5")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

try:
    scaler = joblib.load("scaler.pkl")
    le = joblib.load("label_encoder.pkl")
    columns_template = pd.read_csv("X_train_columns.csv").columns.tolist()
except Exception as e:
    st.error(f"❌ Error loading preprocessing files: {e}")
    st.stop()

# --- Wellness Score Function ---
def calculate_wellness_score(sleep, quality, activity, stress):
    score = (
        (sleep / 8.0) * 25 +
        (quality / 10.0) * 25 +
        (activity / 10.0) * 25 +
        ((10 - stress) / 10.0) * 25
    )
    return round(score, 2)

# --- Prediction Function ---
def predict_sleep_disorder(user_input_df):
    for col in columns_template:
        if col not in user_input_df.columns:
            user_input_df[col] = 0
    user_input_df = user_input_df[columns_template]
    scaled = scaler.transform(user_input_df)
    pred = np.argmax(model.predict(scaled), axis=1)[0]
    return le.inverse_transform([pred])[0]

# --- Streamlit UI ---
st.set_page_config(page_title="Sleep Disorder Predictor", page_icon="😴")
st.title("🌙 Sleep Disorder Predictor")
st.write("Complete the questionnaire to receive insights about your sleep health.")

with st.form("sleep_form"):
    gender = st.radio("1. Gender", ["Male", "Female"])
    age = st.slider("2. Age", 18, 80, 25)
    occupation = st.selectbox("3. Occupation", [
        "Nurse", "Doctor", "Engineer", "Teacher", "Accountant",
        "Sales Representative", "Software Engineer", "Lawyer", "Scientist"
    ])
    sleep_duration = st.slider("4. Sleep Duration (hrs)", 0.0, 12.0, 7.0)
    sleep_quality = st.slider("5. Sleep Quality (1–10)", 1, 10, 7)
    activity = st.slider("6. Physical Activity Level (1–10)", 1, 10, 5)
    stress = st.slider("7. Stress Level (1–10)", 1, 10, 5)
    bmi = st.radio("8. BMI Category", ["Underweight", "Normal", "Overweight", "Obese"])
    heart_rate = st.number_input("9. Heart Rate (bpm)", 40, 120, 70)
    daily_steps = st.number_input("10. Average Daily Steps", 0, 30000, 8000)
    submitted = st.form_submit_button("🔮 Predict")

if submitted:
    input_dict = {
        "Age": age,
        "Sleep Duration": sleep_duration,
        "Quality of Sleep": sleep_quality,
        "Physical Activity Level": activity,
        "Stress Level": stress,
        "Heart Rate": heart_rate,
        "Daily Steps": daily_steps,
        f"Gender_{gender}": 1,
        f"Occupation_{occupation}": 1,
        f"BMI Category_{bmi}": 1
    }

    user_df = pd.DataFrame([input_dict])
    prediction = predict_sleep_disorder(user_df)
    wellness_score = calculate_wellness_score(sleep_duration, sleep_quality, activity, stress)

    st.markdown(f"### 💤 Prediction: **{prediction}**")
    st.markdown(f"### 💙 Sleep Wellness Score: **{wellness_score}/100**")

    # --- Condition-Based Suggestions ---
    if prediction == "Insomnia":
        st.warning("😐 You may be experiencing symptoms of **Insomnia**.")
        st.markdown("""
**What is it?**  
Difficulty falling or staying asleep.

**Possible Causes:**  
Stress, mental health, caffeine, irregular sleep routines, screen exposure before bed.

**Suggestions:**  
- Stick to a consistent sleep-wake cycle  
- Reduce screen time 1 hour before bed  
- Try journaling, breathing exercises, or meditation  
""")
        if wellness_score < 60:
            st.error("⚠️ Your wellness score is low. Consider seeking clinical guidance.")

    elif prediction == "Sleep Apnea":
        st.error("⚠️ Sleep Apnea may be present — consult a specialist.")
        st.markdown("""
**What is it?**  
A sleep disorder where breathing repeatedly stops and starts.

**Risk Factors:**  
Obesity, loud snoring, high blood pressure, smoking, or alcohol consumption.

**Suggestions:**  
- Avoid sleeping on your back  
- Monitor weight and physical health  
- Consider a sleep study  
""")
        if wellness_score < 60:
            st.warning("🚨 Sleep quality is severely impacted. Medical evaluation is advised promptly.")

    else:
        st.success("✅ No sleep disorder detected.")
        if wellness_score < 60:
            st.warning("😴 However, your Sleep Wellness Score suggests your sleep habits need attention.")
            st.markdown("""
**Tips for Improving Sleep Quality:**  
- Go to bed and wake up at consistent times  
- Get at least 30 minutes of daylight exposure daily  
- Avoid caffeine after 2 PM  
- Engage in light physical activity during the day  
""")
        else:
            st.info("🌟 You’re sleeping well! Maintain those healthy routines.")
