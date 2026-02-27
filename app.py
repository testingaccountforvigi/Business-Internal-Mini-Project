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

# Category multipliers (affects installations)
CATEGORY_BOOST = {
    'Tools': 3.0, 'Communication': 2.8, 'Social': 2.5, 'Productivity': 2.2, 
    'Entertainment': 2.0, 'Shopping': 1.8, 'Sports': 1.6, 'Education': 1.5,
    'Games': 1.4, 'Finance': 1.2, 'Health & Fitness': 1.1, 'Travel & Local': 1.1,
    'Music & Audio': 1.0, 'Lifestyle': 0.9, 'Books & Reference': 0.8, 
    'Business': 0.7, 'Food & Drink': 0.7, 'Personalization': 0.6, 
    'Puzzle': 0.6, 'Arcade': 0.5, 'Casual': 0.5, 'Other': 1.0
}

st.title("🚀 App Revenue Predictor")
st.markdown("**Model → Installs × Category Boost → Revenue**")

categories = ['Arcade', 'Books & Reference', 'Business', 'Casual', 'Communication', 
              'Education', 'Entertainment', 'Finance', 'Food & Drink', 
              'Health & Fitness', 'Lifestyle', 'Music & Audio', 'Other', 
              'Personalization', 'Productivity', 'Puzzle', 'Shopping', 
              'Social', 'Sports', 'Tools', 'Travel & Local']

category = st.selectbox("**📱 Category**", categories)
rating = st.slider("**⭐ Rating** (1-5)", 1.0, 5.0, 4.2)
price = st.number_input("**💰 Price** ($)", 0.0, 50.0, 0.0)

if st.button("🎯 Predict Revenue", type="primary"):
    try:
        lin_model, scaler, le_cat = load_models()
        cat_encoded = le_cat.transform([category])[0]
        
        # STEP 1: Model predicts BASE installations
        input_df = pd.DataFrame({
            'Rating': [rating], 
            'LogPrice': [np.log1p(price)], 
            'Category_Encoded': [cat_encoded]
        })
        input_scaled = scaler.transform(input_df)
        base_log_installs = lin_model.predict(input_scaled)[0]
        base_log_installs = np.clip(base_log_installs, 0, 15)
        base_installs = np.expm1(base_log_installs)
        
        # STEP 2: Multiply installations by category boost
        category_boost = CATEGORY_BOOST.get(category, 1.0)
        installations_before_rounding = base_installs * category_boost
        
        # STEP 3: Round to whole number (no decimals)
        final_installs = round(installations_before_rounding)
        
        # STEP 4: Apply price caps AFTER rounding
        if price == 0:
            final_installs = min(final_installs, 2000000)
            final_installs = max(final_installs, 1000)
        elif price <= 1:
            final_installs = min(final_installs, 500000)
            final_installs = max(final_installs, 500)
        elif price <= 5:
            final_installs = min(final_installs, 100000)
            final_installs = max(final_installs, 100)
        else:
            final_installs = min(final_installs, 25000)
            final_installs = max(final_installs, 10)
        
        # STEP 5: Calculate revenue from rounded installations
        revenue = final_installs * price
        
        # Display
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📈 Installations", f"{final_installs:,}")
        with col2:
            st.metric("💰 Revenue", f"${revenue:,.0f}")
        
        st.success(f"""
        **{category}** | **{rating}⭐** | **${price}**
        → **{final_installs:,} installations** = **${revenue:,.0f} revenue**
        """)
        st.balloons()
        
    except Exception as e:
        st.error(f"Error: {str(e)}")

st.markdown("---")
st.caption("🎓 Business Analytics Mini-Project")
