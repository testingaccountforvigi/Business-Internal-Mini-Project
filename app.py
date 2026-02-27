import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

st.title("Google Play Revenue Prediction")

rating = st.slider("App Rating", 1.0, 5.0, 4.0, step=0.01)

price = st.number_input("App Price ($)", 
                        min_value=0.0, 
                        max_value=500.0, 
                        value=0.0, 
                        step=0.1)

installs = st.number_input("Number of Installs", 
                           min_value=0, 
                           max_value=100000000, 
                           value=100000, 
                           step=1000)

if st.button("Predict Revenue"):
    
    base_revenue = price * installs
    log_base_revenue = np.log1p(base_revenue)
    
    input_data = np.array([[rating, log_base_revenue]])
    
    log_prediction = model.predict(input_data)
    
    final_revenue = np.expm1(log_prediction[0])
    
    st.success(f"Estimated Revenue: ${final_revenue:,.2f}")
