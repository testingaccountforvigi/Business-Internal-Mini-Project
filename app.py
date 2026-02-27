import streamlit as st
import pandas as pd
import numpy as np

st.title("Google Play Revenue Prediction")

rating = st.slider("App Rating",1.0,5.0,4.0)
price = st.number_input("App Price",0.0,100.0,0.0)

if st.button("Predict Revenue Category"):
    if rating > 4 and price < 5:
        st.success("High Revenue Probability")
    else:
        st.warning("Low Revenue Probability")
