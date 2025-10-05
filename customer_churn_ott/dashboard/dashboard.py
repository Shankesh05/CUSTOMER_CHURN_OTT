# dashboard.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ==============================
# Paths
# ==============================
PRED_PATH = "../data/netflix_predictions_with_actuals.csv"
TOP_PATH = "../data/top_churn_users.csv"
ENCODERS_PATH = "../models/encoders.pkl"
MODEL_PATH = "../models/churn_model.pkl"
FEATURE_COLS_PATH = "../models/feature_columns.pkl"
SCALER_PATH = "../models/scaler.pkl"

# ==============================
# Load model, encoders, scaler, features
# ==============================
model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODERS_PATH)
feature_cols = joblib.load(FEATURE_COLS_PATH)
scaler = joblib.load(SCALER_PATH)

# ==============================
# Streamlit page config
# ==============================
st.set_page_config(page_title="Netflix Churn Dashboard", layout="wide")
st.title("📊 Netflix User Churn Dashboard")

# ==============================
# Load prediction data
# ==============================
df = pd.read_csv(PRED_PATH)
top_churn = pd.read_csv(TOP_PATH)

# Sort top_churn by user_id ascending
top_churn = top_churn.sort_values('user_id')

# ==============================
# Sidebar filters
# ==============================
st.sidebar.header("Filters")

def safe_multiselect(col_name):
    if col_name in df.columns:
        return st.sidebar.multiselect(
            f"Select {col_name.replace('_',' ').title()}",
            options=df[col_name].unique(),
            default=df[col_name].unique()
        )
    return None

country_filter = safe_multiselect("country")
subscription_filter = safe_multiselect("subscription_plan")

filtered_df = df.copy()
if country_filter is not None:
    filtered_df = filtered_df[filtered_df['country'].isin(country_filter)]
if subscription_filter is not None:
    filtered_df = filtered_df[filtered_df['subscription_plan'].isin(subscription_filter)]

# ==============================
# Metrics summary
# ==============================
st.subheader("Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Users", filtered_df.shape[0])
col2.metric("Actual Churned", filtered_df['churned_actual'].sum())
col3.metric("Predicted Churned", filtered_df['churn_prediction'].sum())

# ==============================
# Churn probability histogram
# ==============================
st.subheader("Churn Probability Distribution")
fig, ax = plt.subplots(figsize=(8,4))
ax.hist(filtered_df['churn_probability'], bins=20, color='skyblue', edgecolor='black')
ax.set_xlabel("Churn Probability")
ax.set_ylabel("Number of Users")
st.pyplot(fig)

# ==============================
# Scatter: Avg watch vs churn probability
# ==============================
if 'avg_watch_minutes' in filtered_df.columns:
    st.subheader("Average Watch Minutes vs Churn Probability")
    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.scatter(filtered_df['avg_watch_minutes'], filtered_df['churn_probability'], alpha=0.5)
    ax2.set_xlabel("Average Watch Minutes")
    ax2.set_ylabel("Churn Probability")
    st.pyplot(fig2)

# ==============================
# Threshold slider for high-risk users
# ==============================
st.subheader("Users Above Churn Probability Threshold")
threshold = st.slider("Churn Probability Threshold", 0.0, 1.0, 0.5, 0.05)
high_risk = filtered_df[filtered_df['churn_probability'] >= threshold].sort_values(
    by='churn_probability', ascending=False
)
st.dataframe(high_risk[['user_id','churned_actual','churn_prediction','churn_probability']].reset_index(drop=True))

# ==============================
# Top 100 likely churn users
# ==============================
st.subheader("Top 100 Likely Churn Users")
st.dataframe(top_churn.reset_index(drop=True))

# ==============================
# Manual churn prediction
# ==============================
st.sidebar.header("Manual Churn Prediction")
st.sidebar.markdown("Test a hypothetical/new user")

# Input fields
input_values = {}
for col in ['age','gender','subscription_plan','avg_watch_minutes','total_sessions','country']:
    if col == 'age':
        input_values[col] = st.sidebar.number_input("Age", 18, 100, 35)
    elif col == 'gender':
        input_values[col] = st.sidebar.selectbox("Gender", ["Male","Female","Other"])
    elif col == 'subscription_plan':
        input_values[col] = st.sidebar.selectbox("Subscription Plan", ["Basic","Standard","Premium"])
    elif col == 'avg_watch_minutes':
        input_values[col] = st.sidebar.number_input("Avg Watch Minutes", 0.0, 500.0, 50.0)
    elif col == 'total_sessions':
        input_values[col] = st.sidebar.number_input("Total Sessions", 0, 100, 10)
    elif col == 'country':
        input_values[col] = st.sidebar.text_input("Country", "USA")

# Safe label encoding
def safe_transform(le, val):
    val = str(val)
    if val in le.classes_:
        return le.transform([val])[0]
    else:
        return le.transform([le.classes_[0]])[0]

if st.sidebar.button("Predict Churn"):
    # Create full DataFrame with all feature columns
    new_user_full = pd.DataFrame([{col: 0 for col in feature_cols}])
    
    # Fill user inputs
    for col, val in input_values.items():
        if col in new_user_full.columns:
            new_user_full[col] = val
    
    # Apply encoders safely
    for col, le in encoders.items():
        if col in new_user_full.columns:
            new_user_full[col] = new_user_full[col].apply(lambda x: safe_transform(le, x))
    
    # Ensure all numeric columns expected by scaler exist
    numeric_cols = scaler.feature_names_in_  # all features used during training
    for col in numeric_cols:
        if col not in new_user_full.columns:
            new_user_full[col] = 0  # default 0
    
    # Apply scaler
    new_user_full[numeric_cols] = scaler.transform(new_user_full[numeric_cols])
    
    # Predict
    pred = model.predict(new_user_full)[0]
    prob = model.predict_proba(new_user_full)[0][1]
    
    # Risk categorization
    if prob < 0.3:
        risk = "✅ Low Risk"
    elif prob < 0.7:
        risk = "⚠️ Medium Risk"
    else:
        risk = "❌ High Risk"
    
    st.subheader("Manual Prediction Result")
    st.success(f"Churn Prediction: {pred}")
    st.info(f"Churn Probability: {prob:.2f} ({risk})")
