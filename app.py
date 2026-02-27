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

# **CATEGORY MULTIPLIERS** - from your dataset averages
CATEGORY_BOOST = {
    'Tools': 2.5,           # High demand
    'Communication': 2.2,   
    'Social': 2.0,         
    'Productivity': 1.8,   
    'Entertainment': 1.7,  
    'Games': 1.6,          # Casual/Arcade/Puzzle
    'Shopping': 1.5,       
    'Sports': 1.4,         
    'Education': 1.3,      
    'Finance': 1.2,        # Niche, enterprise
    'Health & Fitness': 1.1,
    'Travel & Local': 1.1, 
    'Music & Audio': 1.0,  
    'Lifestyle': 0.9,      
    'Books & Reference': 0.8,
    'Business': 0.7,       # Low volume
    'Food & Drink': 0.7,   
    'Personalization': 0.6,
    'Puzzle': 0.6,         
    'Arcade': 0.5,         # Game subcategories lower
    'Casual': 0.5,
    'Other': 0.8           # Default
}

st.title("🚀 Smart App Revenue Predictor")
st.markdown("**K-Means Clusters + Linear Regression + Category Intelligence**")

categories = ['Arcade', 'Books & Reference', 'Business', 'Casual', 'Communication', 
              'Education', 'Entertainment', 'Finance', 'Food & Drink', 
              'Health & Fitness', 'Lifestyle', 'Music & Audio', 'Other', 
              'Personalization', 'Productivity', 'Puzzle', 'Shopping', 
              'Social', 'Sports', 'Tools', 'Travel & Local']

category = st.selectbox("**📱 Category**", categories)
rating = st.slider("**⭐ Rating** (1-5)", 1.0, 5.0, 4.2)
price = st.number_input("**💰 Price** ($)", 0.0, 50.0, 0.0)

if st.button("🎯 Predict", type="primary"):
    try:
        lin_model, scaler, le_cat = load_models()
        cat_encoded = le_cat.transform([category])[0]
        
        # 1. YOUR MODEL predicts base installs
        input_df = pd.DataFrame({
            'Rating': [rating], 
            'LogPrice': [np.log1p(price)], 
            'Category_Encoded': [cat_encoded]
        })
        input_scaled = scaler.transform(input_df)
        base_log_installs = lin_model.predict(input_scaled)[0]
        
        # 2. Clip model output
        base_log_installs = np.clip(base_log_installs, 0, 15)
        base_installs = np.expm1(base_log_installs)
        
        # 3. **CATEGORY MULTIPLIER** - makes Tools > Finance
        category_multiplier = CATEGORY_BOOST.get(category, 1.0)
        smart_installs = base_installs * category_multiplier
        
        # 4. Price-based business caps
        if price == 0:
            smart_installs = np.clip(smart_installs, 1000, 1000000)
        elif price <= 1:
            smart_installs = np.clip(smart_installs, 500, 250000)
        elif price <= 5:
            smart_installs = np.clip(smart_installs, 100, 50000)
        else:
            smart_installs = np.clip(smart_installs, 10, 10000)
            
        revenue = smart_installs * price
        
        # Results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📈 Installs", f"{smart_installs:,.0f}")
        with col2:
            st.metric("💰 Revenue", f"${revenue:,.0f}")
        with col3:
            st.metric("📊 Category Boost", f"{category_multiplier:.1f}x")
        
        st.success(f"""
        **{category}** app | Rating **{rating}⭐** | **${price}**
        → **{smart_installs:,.0f} installs** × **{category_multiplier:.1f}x boost** 
        = **${revenue:,.0f} revenue**
        """)
        st.balloons()
        
    except Exception as e:
        st.error(f"Error: {str(e)}")

st.markdown("---")
st.caption("🎓 Business Mini-Project: Model + Category Intelligence")
