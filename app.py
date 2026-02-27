import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

st.title("Google Play Revenue Prediction")

rating = st.slider("App Rating", 1.0, 5.0, 4.0)
price = st.number_input("App Price ($)", 0.0, 500.0, 0.0)
installs = st.number_input("Number of Installs", 0, 100000000, 100000)

if st.button("Predict Revenue"):
    
    log_installs = np.log1p(installs)
    input_data = np.array([[rating, price, log_installs]])
    
    log_prediction = model.predict(input_data)
    
    revenue = np.expm1(log_prediction[0])  # reverse log
    
    st.success(f"Estimated Revenue: ${revenue:,.2f}")
