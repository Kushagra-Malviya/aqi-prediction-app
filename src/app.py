# 📚 Import libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# 🎯 Streamlit page setup
st.set_page_config(page_title="AQI Forecast (Anand Vihar)", layout="wide")

# 📦 Load trained model
from tensorflow.keras.metrics import MeanSquaredError

# Register custom metric if necessary
custom_objects = {'mse': MeanSquaredError()}

model = load_model('models/best_aqi_model (1).h5', custom_objects=custom_objects)

# 📢 Title
st.title("🌍 Anand Vihar, Delhi: AQI Forecast")
st.markdown("Predict AQI automatically from 23/04/2025 till today, and forecast the next 7 days.")

# 📜 Given AQI values (up to 23/04/2025)
initial_aqi_values = [
    365, 400, 382, 332, 341, 500, 368, 388, 339, 325, 282,
    239, 309, 309, 309, 331, 324, 330, 324, 359, 400, 397, 394
]

# ⚙️ Prepare data
df_known = pd.DataFrame({'AQI': initial_aqi_values})

# 📈 Scale AQI values
scaler = MinMaxScaler()
scaled_known = scaler.fit_transform(df_known[['AQI']])

# 🗓️ Start from 23/04/2025
start_date = pd.to_datetime('2025-04-23')
today = pd.to_datetime('today').normalize()

# 💾 Predict AQI from 24/04/2025 to today
time_steps = 15
current_sequence = scaled_known[-time_steps:]  # last 15 values

predicted_aqi_till_today = []
current_date = start_date + pd.Timedelta(days=1)

while current_date <= today:
    X_input = np.expand_dims(current_sequence, axis=0)  # (1, 15, 1)
    pred_aqi = model.predict(X_input)[0][0]
    predicted_aqi_till_today.append((current_date.strftime("%d-%m-%Y"), pred_aqi))

    # Prepare next input sequence
    current_sequence = np.vstack([current_sequence[1:], [[pred_aqi]]])  # slide window

    current_date += pd.Timedelta(days=1)

# 📊 Show AQI predicted till today
if predicted_aqi_till_today:
    st.subheader("📅 Predicted AQI up to Today")

    # Inverse transform predictions
    pred_scaled = np.array([aqi for date, aqi in predicted_aqi_till_today]).reshape(-1, 1)
    pred_inv = scaler.inverse_transform(pred_scaled).flatten()

    predicted_df = pd.DataFrame({
        'Date': [date for date, aqi in predicted_aqi_till_today],
        'Predicted AQI': pred_inv
    })

    st.dataframe(predicted_df.style.format({"Predicted AQI": "{:.2f}"}))

    # ✨ Now forecast next 7 days
    st.subheader("🔮 AQI Forecast for Next 7 Days")

    future_predictions = []
    future_dates = []

    current_sequence = np.vstack([scaled_known[-(time_steps - len(predicted_aqi_till_today)):], pred_scaled])

    for i in range(7):
        X_input = np.expand_dims(current_sequence[-time_steps:], axis=0)
        pred_future_aqi = model.predict(X_input)[0][0]
        future_predictions.append(pred_future_aqi)

        # Update sequence
        current_sequence = np.vstack([current_sequence, [[pred_future_aqi]]])

        future_date = today + pd.Timedelta(days=i+1)
        future_dates.append(future_date.strftime("%d-%m-%Y"))

    # Inverse transform future predictions
    future_pred_scaled = np.array(future_predictions).reshape(-1, 1)
    future_pred_inv = scaler.inverse_transform(future_pred_scaled).flatten()

    future_df = pd.DataFrame({'Date': future_dates, 'Forecasted AQI': future_pred_inv})
    st.dataframe(future_df.style.format({"Forecasted AQI": "{:.2f}"}))

    # 📈 Plot severity bars
    st.subheader("🚦 AQI Severity for Next 7 Days")

    for idx, row in future_df.iterrows():
        st.markdown(f"**{row['Date']}**")
        fig, ax = plt.subplots(figsize=(8, 0.6))
        norm_value = row['Forecasted AQI'] / 700  # Assuming 500 is maximum AQI scale
        ax.barh(0, norm_value, color=plt.cm.RdYlGn_r(norm_value))
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(f"AQI: {row['Forecasted AQI']:.2f}", fontsize=12)
        st.pyplot(fig)

else:
    st.warning("No predictions needed: Already up to date!")
