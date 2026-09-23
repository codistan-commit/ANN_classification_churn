import pandas as pd
import pickle
import tensorflow as tf
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        color: white;
    }

    /* Remove top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Main Heading */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        color: #38bdf8;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #cbd5e1;
        font-size: 17px;
        margin-bottom: 35px;
    }

    /* Cards */
    .card {
        background: rgba(30, 41, 59, 0.9);
        padding: 25px;
        border-radius: 18px;
        border: 1px solid rgba(148, 163, 184, 0.2);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.25);
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 23px;
        font-weight: 700;
        color: #7dd3fc;
        margin-bottom: 20px;
    }

    /* Labels */
    label {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }

    /* Input Fields */
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background-color: #334155 !important;
        border-radius: 10px !important;
        border: 1px solid #475569 !important;
    }

    input {
        color: white !important;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        height: 52px;
        border-radius: 12px;
        border: none;
        background: linear-gradient(90deg, #0284c7, #2563eb);
        color: white;
        font-size: 18px;
        font-weight: 700;
        transition: 0.3s;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #0369a1, #1d4ed8);
        transform: scale(1.02);
    }

    /* Prediction Result */
    .prediction-card {
        background: linear-gradient(135deg, #164e63, #075985);
        padding: 30px;
        border-radius: 18px;
        text-align: center;
        margin-top: 25px;
        border: 1px solid #38bdf8;
    }

    .probability {
        font-size: 45px;
        font-weight: 800;
        color: #7dd3fc;
    }

    .result-text {
        font-size: 22px;
        font-weight: 700;
        margin-top: 10px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #020617;
    }

    /* Divider */
    hr {
        border-color: #475569;
    }

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODEL, ENCODERS AND SCALER
# --------------------------------------------------

model = tf.keras.models.load_model("model.h5")

with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("ohe_geo.pkl", "rb") as file:
    label_encoder_geo = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered customer retention analysis using an Artificial Neural Network'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 📊 Churn AI")
    st.markdown("---")

    st.markdown("### About the Model")
    st.write(
        "This application uses a trained Artificial Neural Network "
        "to estimate the probability of customer churn."
    )

    st.markdown("---")

    st.markdown("### Prediction Guide")
    st.write("**Probability > 0.50:** Customer likely to churn")
    st.write("**Probability ≤ 0.50:** Customer unlikely to churn")

    st.markdown("---")
    st.caption("Built with TensorFlow and Streamlit")


# --------------------------------------------------
# USER INPUT FORM
# --------------------------------------------------

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">👤 Customer Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    geography = st.selectbox(
        "🌍 Geography",
        label_encoder_geo.categories_[0]
    )

    gender = st.selectbox(
        "⚥ Gender",
        label_encoder_gender.classes_
    )

    age = st.slider(
        "🎂 Age",
        18,
        92,
        35
    )

with col2:
    credit_score = st.number_input(
        "💳 Credit Score",
        min_value=0,
        max_value=1000,
        value=650
    )

    balance = st.number_input(
        "💰 Balance",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    estimated_salary = st.number_input(
        "💵 Estimated Salary",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

with col3:
    tenure = st.slider(
        "📅 Tenure",
        0,
        10,
        5
    )

    num_of_products = st.slider(
        "📦 Number of Products",
        1,
        4,
        1
    )

    has_cr_card = st.selectbox(
        "💳 Has Credit Card",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    is_active_member = st.selectbox(
        "✅ Is Active Member",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)

predict_button = st.button("🔍 Predict Customer Churn")


if predict_button:

    # ----------------------------------------------
    # PREPARE INPUT DATA
    # ----------------------------------------------

    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": [
            label_encoder_gender.transform([gender])[0]
        ],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary]
    })

    # ----------------------------------------------
    # ONE-HOT ENCODE GEOGRAPHY
    # ----------------------------------------------

    geo_encoded = label_encoder_geo.transform(
        [[geography]]
    ).toarray()

    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=label_encoder_geo.get_feature_names_out(
            ["Geography"]
        )
    )

    # ----------------------------------------------
    # COMBINE INPUT DATA
    # ----------------------------------------------

    input_data = pd.concat(
        [
            input_data.reset_index(drop=True),
            geo_encoded_df
        ],
        axis=1
    )

    # ----------------------------------------------
    # SCALE INPUT DATA
    # ----------------------------------------------

    input_data_scaled = scaler.transform(input_data)

    # ----------------------------------------------
    # PREDICTION
    # ----------------------------------------------

    prediction = model.predict(
        input_data_scaled,
        verbose=0
    )

    prediction_probab = prediction[0][0]

    # ----------------------------------------------
    # DISPLAY RESULT
    # ----------------------------------------------

    st.markdown(
        '<div class="prediction-card">',
        unsafe_allow_html=True
    )

    st.markdown("## 📈 Prediction Result")

    st.markdown(
        f'<div class="probability">'
        f'{prediction_probab:.2%}'
        f'</div>',
        unsafe_allow_html=True
    )

    st.write("Estimated Churn Probability")

    if prediction_probab > 0.5:

        st.markdown(
            '<div class="result-text">'
            '⚠️ The customer is likely to churn'
            '</div>',
            unsafe_allow_html=True
        )

        st.warning(
            "This customer may require retention-focused action."
        )

    else:

        st.markdown(
            '<div class="result-text">'
            '✅ The customer is not likely to churn'
            '</div>',
            unsafe_allow_html=True
        )

        st.success(
            "This customer currently shows a lower churn probability."
        )

    st.markdown("</div>", unsafe_allow_html=True)