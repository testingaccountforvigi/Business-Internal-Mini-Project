import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

st.title("Google Play Rating vs Revenue Prediction")

st.write("Predict Estimated Revenue Based on App Rating")

rating = st.slider("App Rating", 1.0, 5.0, 4.0)

if st.button("Predict Revenue"):
    
    input_data = np.array([[rating]])
    prediction = model.predict(input_data)

    st.success(f"Estimated Revenue: ${prediction[0]:,.2f}")
