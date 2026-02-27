import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

st.title("Google Play Revenue Prediction")

rating = st.slider("App Rating", 1.0, 5.0, 4.0)
price = st.number_input("App Price ($)", 0.0, 500.0, 0.0)
installs = st.number_input("Number of Installs", 0, 100000000, 100000)

if st.button("Predict Revenue"):
    
    input_data = np.array([[rating, price, installs]])
    prediction = model.predict(input_data)

    st.success(f"Estimated Revenue: ${prediction[0]:,.2f}")
