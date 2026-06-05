
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------
# LOAD FILES
# -------------------------

model=joblib.load("model.pkl")
scaler=joblib.load("scaler.pkl")
columns=joblib.load("columns.pkl")

# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide"
)

# -------------------------
# CSS
# -------------------------

st.markdown("""

<style>

.stApp{
background:
linear-gradient(
135deg,
#172554 0%,
#1e3a8a 40%,
#0f766e 100%
);
color:white;
}

.big-title{
font-size:42px;
font-weight:700;
text-align:center;
color:white;
margin-bottom:10px;
}

.subtitle{
font-size:18px;
text-align:center;
color:#dbeafe;
margin-bottom:30px;
}

.metric-box{

padding:20px;

background:
rgba(255,255,255,0.12);

border-radius:15px;

backdrop-filter:blur(10px);

text-align:center;

}

.pred-box{

padding:25px;

border-radius:15px;

font-size:24px;

font-weight:bold;

text-align:center;

}

.stButton button{

width:100%;

height:55px;

border-radius:12px;

background:#38bdf8;

color:black;

font-weight:bold;

border:none;

font-size:18px;

}

.stButton button:hover{

background:#0ea5e9;

}

section[data-testid="stSidebar"]{

background:rgba(255,255,255,0.05);

}

</style>

""",unsafe_allow_html=True)

# -------------------------
# HEADER
# -------------------------

st.markdown("""

<div class='big-title'>

📉 Customer Churn Prediction Dashboard

</div>

<div class='subtitle'>

Predict customer churn risk using Machine Learning trained on IBM Telco Dataset

</div>

""",unsafe_allow_html=True)

c1,c2,c3=st.columns(3)

with c1:

    st.markdown("""

    <div class='metric-box'>

    <h3>Goal</h3>

    Identify churn risk early

    </div>

    """,unsafe_allow_html=True)

with c2:

    st.markdown("""

    <div class='metric-box'>

    <h3>Model</h3>

    L1 Balanced Logistic Regression

    </div>

    """,unsafe_allow_html=True)

with c3:

    st.markdown("""

    <div class='metric-box'>

    <h3>Focus</h3>

    High Recall Retention Strategy

    </div>

    """,unsafe_allow_html=True)

st.write("")
st.header("Customer Information")

# -------------------------
# INPUTS
# -------------------------

left,right=st.columns(2)

with left:

    gender=st.selectbox(
        "Gender",
        ["Male","Female"]
    )

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
        "Tenure (Months)",
        0,
        72,
        12
    )

    monthly=st.number_input(
        "Monthly Charges",
        value=70.0
    )

with right:

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
# INPUT PROCESSING
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
(monthly>70)
&
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

input_scaled=scaler.transform(input_df)

# -------------------------
# PREDICT
# -------------------------

if st.button("Predict Churn Risk"):

    prediction=model.predict(
        input_scaled
    )[0]

    probability=model.predict_proba(
        input_scaled
    )[0][1]

    st.write("")

    if prediction=="Yes":

        st.markdown(

f"""

<div class='pred-box'
style='background:#fee2e2;color:#991b1b;'>

⚠ HIGH CHURN RISK

<br><br>

Probability: {probability:.2%}

</div>

""",

unsafe_allow_html=True

)

    else:

        st.markdown(

f"""

<div class='pred-box'
style='background:#dcfce7;color:#14532d;'>

✓ LOW CHURN RISK

<br><br>

Probability: {probability:.2%}

</div>

""",

unsafe_allow_html=True

)
