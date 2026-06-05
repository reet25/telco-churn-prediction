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

.main {
background-color:#f6f8fc;
}

.stApp{
background: linear-gradient(
180deg,
#eef2ff 0%,
#f8fafc 100%
);
}

.big-title{
font-size:42px;
font-weight:700;
color:#1e293b;
}

.subtitle{
font-size:18px;
color:#475569;
}

.metric-box{
padding:20px;
border-radius:15px;
background:white;
box-shadow:0px 2px 12px rgba(0,0,0,0.08);
}

.pred-box{
padding:25px;
border-radius:15px;
font-size:28px;
font-weight:bold;
text-align:center;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# LOAD MODEL
# -------------------------

model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

# -------------------------
# HEADER
# -------------------------

st.markdown(
"""
<div class='big-title'>
📉 Customer Churn Prediction Dashboard
</div>

<div class='subtitle'>
Predict customer churn risk using machine learning built on the IBM Telco Dataset
</div>
""",
unsafe_allow_html=True
)

st.write("")

col1,col2,col3=st.columns(3)

with col1:
    st.markdown(
    """
    <div class='metric-box'>
    <h4>Goal</h4>
    Identify customers likely to churn
    </div>
    """,
    unsafe_allow_html=True
    )

with col2:
    st.markdown(
    """
    <div class='metric-box'>
    <h4>Model</h4>
    L1 Regularized Logistic Regression
    </div>
    """,
    unsafe_allow_html=True
    )

with col3:
    st.markdown(
    """
    <div class='metric-box'>
    <h4>Focus</h4>
    High Recall Customer Retention
    </div>
    """,
    unsafe_allow_html=True
    )

st.write("")
st.header("Customer Information")

# -------------------------
# INPUTS
# -------------------------

col1,col2=st.columns(2)

with col1:

    gender=st.selectbox("Gender",["Male","Female"])

    senior=st.selectbox(
        "Senior Citizen",
        [0,1]
    )

    partner=st.selectbox(
        "Partner",
        ["Yes","No"]
    )

    dependents=st.selectbox(
        "Dependents",
        ["Yes","No"]
    )

    tenure=st.slider(
        "Tenure",
        0,
        72,
        12
    )

    monthly=st.number_input(
        "Monthly Charges",
        value=70.0
    )

with col2:

    contract=st.selectbox(
        "Contract",
        ["Month-to-month","One year","Two year"]
    )

    internet=st.selectbox(
        "Internet Service",
        ["DSL","Fiber optic","No"]
    )

    tech=st.selectbox(
        "Tech Support",
        ["Yes","No","No internet service"]
    )

    online=st.selectbox(
        "Online Security",
        ["Yes","No","No internet service"]
    )

    payment=st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

# -------------------------
# BUILD INPUT
# -------------------------

input_dict={

'gender':gender,
'SeniorCitizen':senior,
'Partner':partner,
'Dependents':dependents,
'tenure':tenure,
'PhoneService':'Yes',
'MultipleLines':'No',
'InternetService':internet,
'OnlineSecurity':online,
'OnlineBackup':'No',
'DeviceProtection':'No',
'TechSupport':tech,
'StreamingTV':'No',
'StreamingMovies':'No',
'Contract':contract,
'PaperlessBilling':'Yes',
'PaymentMethod':payment,
'MonthlyCharges':monthly,
'TotalCharges':monthly*tenure

}

input_df=pd.DataFrame([input_dict])

input_df['tenure_bucket']=pd.cut(
input_df['tenure'],
bins=[0,11,41,72],
labels=['new','moderate','longterm'],
include_lowest=True
)

input_df['service_usage']=2

input_df['high_charge_new_customer']=(
(monthly>70) &
(input_df['tenure_bucket']=='new')
).astype(int)

input_df=pd.get_dummies(
input_df,
drop_first=True
)

input_df=input_df.reindex(
columns=columns,
fill_value=0
)

# -------------------------
# PREDICT
# -------------------------

if st.button("Predict Churn Risk"):

    prediction=model.predict(input_df)[0]

    probability=model.predict_proba(
        input_df
    )[0][1]

    st.write("")

    if prediction=="Yes":

        st.markdown(
        f"""
        <div class='pred-box'
        style='background:#ffe2e2;color:#b91c1c'>

        ⚠ High Churn Risk

        <br>

        Probability: {probability:.2%}

        </div>
        """,
        unsafe_allow_html=True
        )

    else:

        st.markdown(
        f"""
        <div class='pred-box'
        style='background:#dcfce7;color:#166534'>

        ✓ Low Churn Risk

        <br>

        Probability: {probability:.2%}

        </div>
        """,
        unsafe_allow_html=True
        )
