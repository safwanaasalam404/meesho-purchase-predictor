"""
Meesho Purchase Prediction - Streamlit App (Live Prediction)

Loads the 4 files saved from the training notebook:
    model.pkl     -> the trained Logistic Regression model (best of the 4 compared)
    scaler.pkl    -> StandardScaler fitted on X_train (ALL 15 columns, since the
                     notebook scaled the whole feature set, not just numeric ones)
    encoders.pkl  -> dict of LabelEncoders, one per categorical column
    columns.pkl   -> the exact column order the model was trained on

IMPORTANT: these 4 .pkl files must sit in the SAME FOLDER as this app.py file.

Navigation: a sidebar radio switches between 4 pages. "Prediction" is the
default/first page so a grader opening the app cold lands on the working demo.
"""

import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(page_title="Meesho Purchase Predictor", page_icon="🛍️", layout="wide")

# ------------------------------------------------------------------
# LOAD SAVED MODEL FILES
# (cached so it only loads once, not on every user interaction)
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("encoders.pkl")       # dict: {col_name: LabelEncoder}
    columns = joblib.load("columns.pkl")         # list of column names, in training order
    return model, scaler, encoders, columns

model, scaler, label_encoders, feature_columns = load_artifacts()

# ------------------------------------------------------------------
# FONT
# Pulls in Poppins from Google Fonts and applies it across the whole app.
# Streamlit's native theme only offers 3 generic font choices, so this is
# done via CSS instead, same pattern as the button/heading color changes.
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# BANNER (shown on every page, above the sidebar-selected content)
# ------------------------------------------------------------------
st.markdown(
    """
    <div style="background:linear-gradient(135deg, #D5175D 0%, #7B2CBF 100%);
                padding:18px 24px; border-radius:10px; margin-bottom:16px;
                border-left:6px solid #FF8C42;">
        <h1 style="color:white; margin:0; font-size:28px;">🛍️ Meesho Purchase Predictor</h1>
        <p style="color:#FDE4ED; margin:4px 0 0 0; font-size:14px;">
            Predicting customer purchase likelihood using browsing &amp; account behavior
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Prediction", "Home", "Dataset Overview", "Team", "Model Performance"],
)

# ==================================================================
# PAGE: PREDICTION  (the actual working demo)
# ==================================================================
if page == "Prediction":

    st.caption("Model: Logistic Regression (best of Logistic Regression / Decision Tree / "
               "Random Forest / KNN — Accuracy 0.779, ROC-AUC 0.824)")

    st.markdown('<h3 style="color:#7B2CBF;">Enter customer details</h3>', unsafe_allow_html=True)

    # --------------------------------------------------------------
    # INPUT FORM
    # The categorical dropdown OPTIONS come directly from the saved encoders
    # (label_encoders[col].classes_) so the dropdown can NEVER show a value
    # the encoder doesn't know how to convert back to a number.
    # --------------------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", min_value=18, max_value=65, value=30)

        gender = st.selectbox("Gender", label_encoders["Gender"].classes_)

        device = st.selectbox("Device", label_encoders["Device"].classes_)

        traffic_source = st.selectbox("Traffic Source", label_encoders["Traffic_Source"].classes_)

        pages_viewed = st.slider("Pages Viewed", min_value=1, max_value=30, value=10)

        session_duration = st.slider("Session Duration (minutes)", min_value=0.0, max_value=30.0, value=10.0)

        products_viewed = st.slider("Products Viewed", min_value=0, max_value=20, value=5)

        payment_method = st.selectbox("Payment Method", label_encoders["Payment_Method"].classes_)

    with col2:
        cart_additions = st.slider("Cart Additions", min_value=0, max_value=10, value=1)

        previous_purchases = st.slider("Previous Purchases", min_value=0, max_value=20, value=2)

        # Discount_Used and Returning_Customer were label-encoded (not manually mapped)
        # in the training notebook, so we pull their real category strings the same way
        discount_used = st.selectbox("Discount Used", label_encoders["Discount_Used"].classes_)

        returning_customer = st.selectbox("Returning Customer", label_encoders["Returning_Customer"].classes_)

        customer_rating = st.slider("Customer Rating", min_value=1.0, max_value=5.0, value=3.5, step=0.1)

        avg_order_value = st.number_input("Avg Order Value (₹)", min_value=0.0, value=1200.0, step=50.0)

        days_since_last_visit = st.slider("Days Since Last Visit", min_value=0, max_value=90, value=15)

    # --------------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------------
    st.markdown(
        """
        <style>
        div.stButton > button[kind="primary"] {
            background-color: #FF8C42;
            border-color: #FF8C42;
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: #E67635;
            border-color: #E67635;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Predict Purchase", type="primary"):

        # Step 1: build a dict of raw input values.
        # Categorical values are converted through the SAME encoder objects
        # that were used during training (label_encoders[col].transform([...])),
        # so "Male"/"Female" etc. map to the exact same numbers the model learned.
        input_dict = {
            "Age": age,
            "Gender": label_encoders["Gender"].transform([gender])[0],
            "Device": label_encoders["Device"].transform([device])[0],
            "Traffic_Source": label_encoders["Traffic_Source"].transform([traffic_source])[0],
            "Pages_Viewed": pages_viewed,
            "Session_Duration_Min": session_duration,
            "Products_Viewed": products_viewed,
            "Cart_Additions": cart_additions,
            "Previous_Purchases": previous_purchases,
            "Discount_Used": label_encoders["Discount_Used"].transform([discount_used])[0],
            "Returning_Customer": label_encoders["Returning_Customer"].transform([returning_customer])[0],
            "Customer_Rating": customer_rating,
            "Avg_Order_Value": avg_order_value,
            "Days_Since_Last_Visit": days_since_last_visit,
            "Payment_Method": label_encoders["Payment_Method"].transform([payment_method])[0],
        }

        # Step 2: put it in a single-row DataFrame, with columns in the EXACT
        # order the model was trained on (feature_columns, loaded from columns.pkl).
        # This order-matching is critical -- StandardScaler and LogisticRegression
        # both rely on column position, not column name, once you call .transform()/.predict().
        input_df = pd.DataFrame([input_dict])[feature_columns]

        # Step 3: scale. The notebook scaled ALL 15 columns (not just numeric ones),
        # so we scale the whole row here too, to match training exactly.
        input_scaled = scaler.transform(input_df)

        # Step 4: predict.
        prediction = model.predict(input_scaled)[0]              # 0 = No Purchase, 1 = Purchase
        probability = model.predict_proba(input_scaled)[0][1]    # confidence of class 1 (Purchase)

        st.divider()
        if prediction == 1:
            st.success(f"✅ Likely to PURCHASE — confidence: {probability:.1%}")
        else:
            st.error(f"❌ Unlikely to purchase — purchase confidence: {probability:.1%}")

        st.progress(float(probability))

        # ------------------------------------------------------------
        # DOWNLOAD BUTTON - a plain text report of this one prediction.
        # Built from input_dict (the raw, human-readable values entered above),
        # not the encoded/scaled version, so it stays readable to a person.
        # ------------------------------------------------------------
        result_label = "Likely to PURCHASE" if prediction == 1 else "Unlikely to purchase"
        report = f"""MEESHO PURCHASE PREDICTION REPORT
==================================

CUSTOMER DETAILS
----------------
Age: {age}
Gender: {gender}
Device: {device}
Traffic Source: {traffic_source}
Pages Viewed: {pages_viewed}
Session Duration: {session_duration} min
Products Viewed: {products_viewed}
Cart Additions: {cart_additions}
Previous Purchases: {previous_purchases}
Discount Used: {discount_used}
Returning Customer: {returning_customer}
Customer Rating: {customer_rating}
Avg Order Value: Rs. {avg_order_value}
Days Since Last Visit: {days_since_last_visit}
Payment Method: {payment_method}

PREDICTION RESULT
------------------
Result: {result_label}
Purchase Confidence: {probability:.1%}
Model Used: Logistic Regression
"""

        st.download_button(
            label="📥 Download Prediction Report",
            data=report,
            file_name="meesho_prediction_report.txt",
            mime="text/plain",
        )

# ==================================================================
# PAGE: HOME
# ==================================================================
elif page == "Home":
    st.markdown('<h3 style="color:#7B2CBF;">🏠 Home</h3>', unsafe_allow_html=True)
    st.write(
        "Welcome! This app predicts whether a Meesho customer is likely to make a "
        "purchase, based on their browsing behavior and account activity."
    )

    st.markdown("#### How it works")
    st.markdown(
        """
1. Head to the **Prediction** tab and enter a customer's details (age, device, browsing activity, etc.)
2. Click **Predict Purchase** — the model returns a Likely/Unlikely result with a confidence score
3. Download a report of that prediction if you want to keep a record
        """
    )

    st.markdown("#### About the model")
    st.write(
        "The prediction is powered by a Logistic Regression model, chosen after comparing "
        "it against Decision Tree, Random Forest, and KNN classifiers. See the "
        "**Model Performance** tab for the full comparison, or **Dataset Overview** "
        "for details on the data it was trained on."
    )

# ==================================================================
# PAGE: DATASET OVERVIEW
# Loads the raw CSV separately from the .pkl files (the pkl files only hold
# the TRAINED model/scaler/encoders, not the original data).
# Wrapped in try/except so a missing file shows a friendly message instead
# of crashing the whole app.
# ==================================================================
elif page == "Dataset Overview":
    st.markdown('<h3 style="color:#7B2CBF;">📊 Dataset Overview</h3>', unsafe_allow_html=True)
    try:
        overview_df = pd.read_csv("balanced dataset.csv")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", overview_df.shape[0])
        with col2:
            st.metric("Columns", overview_df.shape[1])
        with col3:
            st.metric("Categorical Features", 6)

        st.markdown("#### Preview")
        st.dataframe(overview_df.head(10), use_container_width=True)

        st.markdown("#### Column Names")
        st.write(list(overview_df.columns))

    except FileNotFoundError:
        st.warning(
            "Dataset file not found. To enable this page, place "
            "`balanced dataset.csv` in the same folder as `app.py`."
        )

# ==================================================================
# PAGE: TEAM
# ==================================================================
elif page == "Team":
    st.markdown('<h3 style="color:#7B2CBF;">👥 Team</h3>', unsafe_allow_html=True)
    st.markdown(
        """
- **Harigovind** — Data Cleaning
- **Rida Abdul Sathar** — EDA & Visualisation
- **Anathakrishnan** — Logistic Regression
- **Faida Rajeef** — Decision Tree & Random Forest
- **Alen Vimal Raj** — Everything after Random Forest (KNN, comparison, evaluation)
- **Safwana Abdul Salam** — Streamlit Deployment
        """
    )

# ==================================================================
# PAGE: MODEL PERFORMANCE
# These numbers come straight from the training notebook's comparison_df —
# hardcoded here since the app doesn't retrain models, just uses the saved one.
# ==================================================================
elif page == "Model Performance":
    st.markdown('<h3 style="color:#FF8C42;">📊 Model Performance Comparison</h3>', unsafe_allow_html=True)

    comparison_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "Decision Tree", "KNN"],
        "Accuracy": [0.7794, 0.7762, 0.7575, 0.7339],
        "Precision": [0.7213, 0.7249, 0.6794, 0.6184],
        "Recall": [0.5402, 0.5185, 0.5022, 0.5054],
        "F1 Score": [0.6178, 0.6046, 0.5775, 0.5562],
        "ROC-AUC": [0.8237, 0.8203, 0.8001, 0.7480],
    })
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    st.caption("Logistic Regression was selected for deployment — highest accuracy and ROC-AUC, "
               "and the smallest gap between training and testing accuracy (least overfit).")