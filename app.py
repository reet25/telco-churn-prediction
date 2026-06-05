import streamlit as st
import joblib

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("Customer Churn Prediction App")
st.markdown("Predict whether a customer will churn using ML (Telco Dataset)")

st.sidebar.header("Customer Input Features")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges", 0, 150, 70)
total_charges = st.sidebar.slider("Total Charges", 0, 10000, 500)

senior = st.sidebar.selectbox("Senior Citizen", [0, 1])

partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])

internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment = st.sidebar.selectbox("Payment Method", 
                               ["Electronic check", "Mailed check", 
                                "Bank transfer (automatic)", "Credit card (automatic)"])

paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

service_usage = st.sidebar.slider("Service Usage (0–6 services)", 0, 6, 3)

new_customer = 1 if tenure <= 11 else 0

input_dict = {
    "SeniorCitizen": senior,
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Partner_Yes": 1 if partner == "Yes" else 0,
    "Dependents_Yes": 1 if dependents == "Yes" else 0,
    "PaperlessBilling_Yes": 1 if paperless == "Yes" else 0,
    "InternetService_Fiber optic": 1 if internet == "Fiber optic" else 0,
    "InternetService_No": 1 if internet == "No" else 0,
    "Contract_One year": 1 if contract == "One year" else 0,
    "Contract_Two year": 1 if contract == "Two year" else 0,
    "PaymentMethod_Electronic check": 1 if payment == "Electronic check" else 0,
    "service_usage": service_usage,
    "high_charge_new_customer": new_customer
}

input_df = pd.DataFrame([input_dict])

input_df = input_df.reindex(columns=columns, fill_value=0)

input_scaled = scaler.transform(input_df)

if st.button("Predict Churn"):
    prob = model.predict_proba(input_scaled)[0][1]
    pred = model.predict(input_scaled)[0]

    st.subheader("Prediction Result")

    if pred == "Yes":
        st.error(f"⚠ Customer WILL CHURN (Probability: {prob:.2f})")
    else:
        st.success(f" Customer will NOT churn (Probability: {prob:.2f})")

    st.progress(float(prob))
