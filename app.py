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
st.markdown("**Predict installs & revenue using K-Means clusters + Linear Regression**")

# EXACT categories from your trained model
categories = ['Arcade', 'Books & Reference', 'Business', 'Casual', 'Communication', 
              'Education', 'Entertainment', 'Finance', 'Food & Drink', 
              'Health & Fitness', 'Lifestyle', 'Music & Audio', 'Other', 
              'Personalization', 'Productivity', 'Puzzle', 'Shopping', 
              'Social', 'Sports', 'Tools', 'Travel & Local']

category = st.selectbox("**📱 Category**", categories)
rating = st.slider("**⭐ Rating** (1-5)", 1.0, 5.0, 4.2)
price = st.number_input("**💰 Price** ($)", 0.0, 50.0, 0.0)

if st.button("🎯 Predict Installs & Revenue", type="primary"):
    try:
        lin_model, scaler, le_cat = load_models()
        cat_encoded = le_cat.transform([category])[0]
        
        # Prepare input
        input_df = pd.DataFrame({
            'Rating': [rating], 
            'LogPrice': [np.log1p(price)], 
            'Category_Encoded': [cat_encoded]
        })
        input_scaled = scaler.transform(input_df)
        
        # Predict log installs
        pred_log_installs = lin_model.predict(input_scaled)[0]
        
        # **FIX: Clip to realistic business ranges** (no negatives!)
        pred_log_installs = np.clip(pred_log_installs, 0, 15)  # 1 to ~3.2M installs
        
        expected_installs = np.expm1(pred_log_installs)
        
        # **Business realistic minimums by price**
        if price == 0:
            expected_installs = min(expected_installs, 500000)  # Free apps cap at 500k
            expected_installs = max(expected_installs, 1000)     # Min 1k downloads
        elif price <= 1:
            expected_installs = min(expected_installs, 100000)   # $0.99 max 100k
            expected_installs = max(expected_installs, 500)      # Min 500
        elif price <= 5:
            expected_installs = min(expected_installs, 25000)     # $5 max 25k
            expected_installs = max(expected_installs, 100)      # Min 100
        else:  # $50+
            expected_installs = min(expected_installs, 5000)      # $50 max 5k
            expected_installs = max(expected_installs, 10)        # Min 10
        
        expected_revenue = expected_installs * price
        
        # Display results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📈 Expected Installs", f"{expected_installs:,.0f}")
        with col2:
            st.metric("💵 Expected Revenue", f"${expected_revenue:,.0f}")
        with col3:
            st.metric("📊 Price per Install", f"${price/expected_installs:.4f}")
        
        st.success(f"""
        ✅ **Analysis**: 
        - {'Free app' if price == 0 else f'${price} app'} in **{category}**
        - Rating **{rating}⭐** → **{expected_installs:,.0f} installs**
        - **Total Revenue: ${expected_revenue:,.0f}**
        """)
        
        st.balloons()
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

st.markdown("---")
st.caption("🎓 Business Analytics Mini-Project | K-Means Clustering + Linear Regression")
