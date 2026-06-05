import streamlit as st
import pandas as pd 
import numpy as np
import joblib

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

import streamlit as st
import pandas as pd
import joblib

# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide"
)

# -------------------------
# CUSTOM CSS
# -------------------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #1e293b 40%,
        #1d4ed8 100%
    );
}

.main-title{
    font-size:42px;
    font-weight:bold;
    color:#ffffff;
    text-align:center;
}

.subtitle{
    font-size:18px;
    color:#dbeafe;
    text-align:center;
    margin-bottom:30px;
}

.stButton>button{
    width:100%;
    background:#38bdf8;
    color:black;
    border-radius:12px;
    height:3em;
    font-weight:bold;
    border:none;
}

.stButton>button:hover{
    background:#0ea5e9;
    color:black;
}

div[data-baseweb="select"]{
    background-color:white;
    border-radius:10px;
}

input{
    border-radius:10px !important;
}

.metric-box{
    padding:20px;
    border-radius:15px;
    background:rgba(255,255,255,0.15);
}

</style>
""",unsafe_allow_html=True)
