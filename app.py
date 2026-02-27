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
st.markdown("Predict installs & revenue based on category, rating, and price")

# EXACT categories from your trained LabelEncoder
categories = ['Arcade', 'Books & Reference', 'Business', 'Casual', 'Communication', 
              'Education', 'Entertainment', 'Finance', 'Food & Drink', 
              'Health & Fitness', 'Lifestyle', 'Music & Audio', 'Other', 
              'Personalization', 'Productivity', 'Puzzle', 'Shopping', 
              'Social', 'Sports', 'Tools', 'Travel & Local']

category = st.selectbox("**Category**", categories)
rating = st.slider("**Rating** (1-5)", 1.0, 5.0, 4.2)
price = st.number_input("**Price** ($)", 0.0, 50.0, 0.0)

if st.button("🎯 Predict Installs & Revenue", type="primary"):
    try:
        lin_model, scaler, le_cat = load_models()
        cat_encoded = le_cat.transform([category])[0]
        
        # Prepare input exactly like training
        input_df = pd.DataFrame({
            'Rating': [rating], 
            'LogPrice': [np.log1p(price)], 
            'Category_Encoded': [cat_encoded]
        })
        input_scaled = scaler.transform(input_df)
        
        pred_log_installs = lin_model.predict(input_scaled)[0]
        expected_installs = np.expm1(pred_log_installs)
        expected_revenue = expected_installs * price
        
        st.success(f"""
        ## ✅ **Prediction Results**
        **Expected Installs:** {expected_installs:,.0f}  
        **Expected Revenue:** **${expected_revenue:,.0f}**
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Installs", f"{expected_installs:,.0f}")
        with col2:
            st.metric("Revenue", f"${expected_revenue:,.0f}")
            
        st.balloons()
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("Models may need re-upload. Check GitHub repo files.")

st.markdown("---")
st.caption("Built for Business Analytics Mini-Project | K-Means + Linear Regression")
