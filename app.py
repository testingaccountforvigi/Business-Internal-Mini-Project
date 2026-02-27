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
        # Load all 3 models: lin_model (LinearRegression), scaler (StandardScaler), le_cat (LabelEncoder)
        lin_model, scaler, le_cat = load_models()
        
        # MODEL 1: le_cat (LabelEncoder) - Encodes category to numeric ✓ USED
        cat_encoded = le_cat.transform([category])[0]
        
        # STEP 1: Prepare input for prediction
        input_df = pd.DataFrame({
            'Rating': [rating], 
            'LogPrice': [np.log1p(price)], 
            'Category_Encoded': [cat_encoded]
        })
        
        # MODEL 2: scaler (StandardScaler) - Scales features ✓ USED
        input_scaled = scaler.transform(input_df)
        
        # MODEL 3: lin_model (LinearRegression) - Predicts installations ✓ USED
        base_log_installs = lin_model.predict(input_scaled)[0]
        base_log_installs = np.clip(base_log_installs, 0, 15)
        base_installs = np.expm1(base_log_installs)
        
        # STEP 2: Get category boost multiplier
        category_boost = CATEGORY_BOOST.get(category, 1.0)
        
        # STEP 3: Calculate boosted installations (multiply base * boost)
        boosted_installations = base_installs * category_boost
        
        # STEP 4: Round to whole number (no decimal points)
        rounded_installations = round(boosted_installations)
        
        # STEP 5: Apply price-based caps
        if price == 0:
            rounded_installations = min(rounded_installations, 2000000)
            rounded_installations = max(rounded_installations, 1000)
        elif price <= 1:
            rounded_installations = min(rounded_installations, 500000)
            rounded_installations = max(rounded_installations, 500)
        elif price <= 5:
            rounded_installations = min(rounded_installations, 100000)
            rounded_installations = max(rounded_installations, 100)
        else:
            rounded_installations = min(rounded_installations, 25000)
            rounded_installations = max(rounded_installations, 10)
        
        # STEP 6: Calculate revenue from rounded installations
        revenue = rounded_installations * price
        
        # Display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📈 Installations", f"{rounded_installations:,}")
        with col2:
            st.metric("💰 Revenue", f"${revenue:,.0f}")
        with col3:
            st.metric("🎯 Category Boost", f"{category_boost}x")
        
        st.success(f"""
        **{category}** | **{rating}⭐** | **${price}**
        → **{rounded_installations:,} installations** = **${revenue:,.0f} revenue**
        (Base: {int(base_installs):,} × Boost: {category_boost}x = {int(boosted_installations):,})
        """)
        st.balloons()
        
    except Exception as e:
        st.error(f"Error: {str(e)}")

st.markdown("---")
st.caption("🎓 Business Analytics Mini-Project")
