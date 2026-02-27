import streamlit as st
import pickle
import pandas as pd
import numpy as np

@st.cache_resource
def load_models():
    lin_model = pickle.load(open('lin_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    le_cat = pickle.load(open('le_cat.pkl', 'rb'))
    return lin_model, scaler, le_cat

st.title("🚀 App Expected Installs & Revenue Predictor")

category = st.selectbox("Category", ['TOOLS', 'GAME', 'Other'])  # Use EXACT casing from your data
rating = st.slider("Rating (1-5)", 1.0, 5.0, 4.2)
price = st.number_input("Price ($)", 0.0, 50.0, 0.0)

if st.button("Predict"):
    try:
        lin_model, scaler, le_cat = load_models()
        
        # Safe encoding - check if category exists
        if category not in le_cat.classes_:
            st.error(f"Category '{category}' not in trained data. Available: {list(le_cat.classes_)}")
            st.stop()
        
        cat_encoded = le_cat.transform([category])[0]
        input_df = pd.DataFrame({
            'Rating': [rating], 
            'LogPrice': [np.log1p(price)], 
            'Category_Encoded': [cat_encoded]
        })
        input_scaled = scaler.transform(input_df)
        pred_log_installs = lin_model.predict(input_scaled)[0]
        expected_installs = np.expm1(pred_log_installs)
        expected_revenue = expected_installs * price
        
        st.success(f"**Expected Installs:** {expected_installs:,.0f}")
        st.success(f"**Expected Revenue:** ${expected_revenue:,.0f}")
        st.balloons()
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
