import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('revenue_model.pkl')

st.title("📱 Google Play Revenue Predictor")
st.write("Enter your app's details below to estimate expected installs and revenue.")

st.divider()

rating = st.slider("App Rating", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
price  = st.number_input("App Price (USD)", min_value=0.01, max_value=999.99, value=2.99, step=0.01)

if st.button("Predict Revenue"):
    log_installs = model.predict([[rating, price]])[0]
    installs     = max(0, round(np.expm1(log_installs)))
    revenue      = installs * price

    st.divider()
    st.subheader("📊 Prediction Results")

    col1, col2 = st.columns(2)
    col1.metric("Expected Installs", f"{installs:,}")
    col2.metric("Estimated Revenue", f"${revenue:,.2f}")

    st.divider()
    st.caption(f"Model input → Rating: {rating} | Price: ${price:.2f}")
