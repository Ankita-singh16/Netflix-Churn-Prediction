import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('churn_model.pkl')

# Page setup
st.set_page_config(page_title="Netflix Churn Predictor", page_icon="🎬")
st.title("Netflix Churn Predictor")
st.markdown("Enter customer details below to predict if they will churn or not!")
st.divider()

# Input fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=70, value=30)
    watch_hours = st.number_input("Total Watch Hours", min_value=0.0, value=10.0)
    last_login_days = st.number_input("Days Since Last Login", min_value=0, value=5)
    monthly_fee = st.selectbox("Monthly Fee ($)", [8.99, 13.99, 17.99])
    number_of_profiles = st.slider("Number of Profiles", 1, 5, 2)

with col2:
    avg_watch_time = st.number_input("Avg Watch Time Per Day (hrs)", min_value=0.0, value=0.5)
    subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    payment_method = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "PayPal", "Crypto", "Gift Card"])
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    device = st.selectbox("Device", ["Mobile", "TV", "Laptop", "Desktop", "Tablet"])

col3, col4 = st.columns(2)
with col3:
    region = st.selectbox("Region", ["Asia", "Europe", "Africa", "North America", "South America", "Oceania"])
with col4:
    favorite_genre = st.selectbox("Favorite Genre", ["Action", "Drama", "Comedy", "Horror", "Romance", "Sci-Fi", "Documentary"])

st.divider()

st.markdown("""
    <style>
    .stButton > button {
        background-color: #4287f5;
        color: white;
        border-radius: 8px;
        font-size: 16px;
        height: 50px;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
    .stSlider > div > div > div {
        background-color: #4287f5 !important;
    }
    .stSlider > div > div > div > div {
        background-color: #E50914 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Prediction function
def prepare_input():
    sub_map = {"Basic": 0, "Premium": 1, "Standard": 2}
    pay_map = {"Credit Card": 0, "Crypto": 1, "Debit Card": 2, "Gift Card": 3, "PayPal": 4}

    data = {
        "age": age,
        "watch_hours": watch_hours,
        "last_login_days": last_login_days,
        "monthly_fee": monthly_fee,
        "number_of_profiles": number_of_profiles,
        "avg_watch_time_per_day": avg_watch_time,
        "subscription_type": sub_map[subscription_type],
        "payment_method": pay_map[payment_method],
        "is_inactive": 1 if last_login_days > 30 else 0,
        "fee_per_profile": monthly_fee / number_of_profiles,
        "watch_intensity": watch_hours / (avg_watch_time + 0.01),
        "gender_Male": 1 if gender == "Male" else 0,
        "gender_Other": 1 if gender == "Other" else 0,
        "region_Asia": 1 if region == "Asia" else 0,
        "region_Europe": 1 if region == "Europe" else 0,
        "region_North America": 1 if region == "North America" else 0,
        "region_Oceania": 1 if region == "Oceania" else 0,
        "region_South America": 1 if region == "South America" else 0,
        "device_Laptop": 1 if device == "Laptop" else 0,
        "device_Mobile": 1 if device == "Mobile" else 0,
        "device_TV": 1 if device == "TV" else 0,
        "device_Tablet": 1 if device == "Tablet" else 0,
        "favorite_genre_Comedy": 1 if favorite_genre == "Comedy" else 0,
        "favorite_genre_Documentary": 1 if favorite_genre == "Documentary" else 0,
        "favorite_genre_Drama": 1 if favorite_genre == "Drama" else 0,
        "favorite_genre_Horror": 1 if favorite_genre == "Horror" else 0,
        "favorite_genre_Romance": 1 if favorite_genre == "Romance" else 0,
        "favorite_genre_Sci-Fi": 1 if favorite_genre == "Sci-Fi" else 0,
    }
    return pd.DataFrame([data])

# Predict button
if st.button("Predict Churn", type="primary", use_container_width=True):
    input_df = prepare_input()
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    churn_prob = model.predict_proba(input_df)[:, 1][0]
    churn_pred = model.predict(input_df)[0]

    st.divider()

    if churn_pred == 1:
        st.error("This customer is likely to CHURN!")
    else:
        st.success("This customer is likely to STAY!")

    st.metric("Churn Probability", f"{round(churn_prob * 100, 1)}%")
    st.progress(float(churn_prob))

    st.subheader("Recommended Action:")
    if churn_prob >= 0.7:
        st.warning("HIGH RISK — Send a discount offer immediately!")
    elif churn_prob >= 0.4:
        st.info("MEDIUM RISK — Send personalized recommendations")
    else:
        st.success("LOW RISK — User is happy, no action needed")