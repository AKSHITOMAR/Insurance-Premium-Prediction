import streamlit as st
import requests
import pandas as pd
import csv
import io
from datetime import datetime
from textwrap import dedent

from model_evaluation import evaluation_results


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Insurance AI | Premium Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONFIGURATION
# ============================================================

API_BASE_URL = "http://13.235.19.194:8000"

PREDICT_URL = f"{API_BASE_URL}/predict"
FEATURE_URL = f"{API_BASE_URL}/feature-importance"
MODEL_INFO_URL = f"{API_BASE_URL}/model-info"


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "active_section" not in st.session_state:
    st.session_state.active_section = None


# ============================================================
# HELPER FUNCTION FOR HTML
# ============================================================

def render_html(html_content):
    """
    Safely render custom HTML without Markdown
    interpreting it as plain text.
    """
    st.html(dedent(html_content))


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    dedent("""
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(109, 40, 217, 0.18),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(6, 182, 212, 0.12),
                transparent 25%
            ),
            linear-gradient(
                135deg,
                #070b18 0%,
                #0b1020 45%,
                #111827 100%
            );
        color: #f8fafc;
    }

    .main .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #090d1a 0%,
                #0d1324 50%,
                #111827 100%
            );
        border-right: 1px solid rgba(148, 163, 184, 0.12);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }

    .sidebar-logo {
        padding: 12px 5px 22px 5px;
    }

    .sidebar-brand {
        font-size: 23px;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.5px;
    }

    .sidebar-subtitle {
        font-size: 12px;
        color: #8f9ab5;
        margin-top: 5px;
    }

    .sidebar-divider {
        height: 1px;
        background: rgba(148, 163, 184, 0.13);
        margin: 5px 0 20px 0;
    }

    .sidebar-heading {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #64748b;
        font-weight: 700;
        margin-top: 22px;
        margin-bottom: 8px;
    }

    .sidebar-text {
        font-size: 13px;
        line-height: 1.65;
        color: #9ca8bf;
    }

    .sidebar-highlight {
        color: #c4b5fd;
        font-weight: 600;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero-wrapper {
        padding: 8px 0 30px 0;
    }

    .hero-badge {
        display: inline-block;
        padding: 7px 13px;
        border-radius: 999px;
        background: rgba(124, 58, 237, 0.14);
        border: 1px solid rgba(167, 139, 250, 0.28);
        color: #c4b5fd;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.1px;
        margin-bottom: 15px;
    }

    .main-title {
        font-size: 43px;
        line-height: 1.15;
        font-weight: 850;
        letter-spacing: -1.7px;
        color: #ffffff;
    }

    .subtitle {
        font-size: 16px;
        line-height: 1.6;
        color: #94a3b8;
        margin-top: 10px;
        max-width: 720px;
    }


    /* ========================================================
       STATUS
       ======================================================== */

    .status-card {
        padding: 14px 18px;
        border-radius: 14px;
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.13);
        margin-bottom: 20px;
    }

    .status-online {
        color: #4ade80;
        font-weight: 700;
    }

    .status-offline {
        color: #f87171;
        font-weight: 700;
    }


    /* ========================================================
       INPUT SECTION
       ======================================================== */

    .section-title {
        font-size: 22px;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 8px;
        margin-bottom: 4px;
    }

    .section-description {
        color: #8491aa;
        font-size: 13px;
        margin-bottom: 20px;
    }

    div[data-testid="stTextInput"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        font-size: 13px !important;
    }

    div[data-baseweb="input"],
    div[data-baseweb="select"] > div {
        background: rgba(15, 23, 42, 0.78) !important;
        border: 1px solid rgba(148, 163, 184, 0.18) !important;
        border-radius: 11px !important;
    }

    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="select"] > div:focus-within {
        border-color: rgba(139, 92, 246, 0.75) !important;
        box-shadow: 0 0 0 1px rgba(139, 92, 246, 0.2) !important;
    }

    input {
        color: #f8fafc !important;
    }

    div[data-baseweb="select"] * {
        color: #f8fafc !important;
    }


    /* ========================================================
       BMI CARD
       ======================================================== */

    .bmi-card {
        background:
            linear-gradient(
                135deg,
                rgba(124, 58, 237, 0.14),
                rgba(6, 182, 212, 0.08)
            );
        border: 1px solid rgba(139, 92, 246, 0.22);
        border-radius: 16px;
        padding: 17px 20px;
        margin-top: 20px;
        margin-bottom: 22px;
    }

    .bmi-label {
        color: #94a3b8;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
    }

    .bmi-value {
        font-size: 29px;
        font-weight: 850;
        color: #ffffff;
        margin-top: 4px;
    }

    .bmi-category {
        color: #a78bfa;
        font-size: 13px;
        font-weight: 700;
        margin-top: 3px;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    div.stButton > button {
        border-radius: 11px;
        border: 1px solid rgba(148, 163, 184, 0.16);
        background: rgba(15, 23, 42, 0.72);
        color: #cbd5e1;
        font-weight: 650;
        min-height: 43px;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        border-color: rgba(139, 92, 246, 0.55);
        color: #ffffff;
        background: rgba(124, 58, 237, 0.14);
    }

    div.stButton > button[kind="primary"] {
        border: none !important;
        background:
            linear-gradient(
                135deg,
                #7c3aed,
                #06b6d4
            ) !important;
        color: white !important;
        font-size: 15px !important;
        font-weight: 800 !important;
        min-height: 52px;
        box-shadow:
            0 8px 28px rgba(124, 58, 237, 0.22);
    }

    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow:
            0 12px 34px rgba(124, 58, 237, 0.3);
    }


    /* ========================================================
       RESULT CARD
       ======================================================== */

    .premium-result-card {
        margin-top: 28px;
        padding: 28px;
        border-radius: 20px;
        background:
            linear-gradient(
                145deg,
                rgba(15, 23, 42, 0.94),
                rgba(30, 41, 59, 0.78)
            );
        border: 1px solid rgba(148, 163, 184, 0.16);
        box-shadow:
            0 20px 55px rgba(0, 0, 0, 0.25);
    }

    .result-low {
        border-left: 4px solid #22c55e;
    }

    .result-medium {
        border-left: 4px solid #eab308;
    }

    .result-high {
        border-left: 4px solid #ef4444;
    }

    .result-small-label {
        color: #7f8ca5;
        text-transform: uppercase;
        letter-spacing: 1.3px;
        font-size: 11px;
        font-weight: 800;
    }

    .result-icon {
        font-size: 30px;
        margin-top: 8px;
    }

    .result-category {
        font-size: 35px;
        font-weight: 900;
        color: white;
        margin-top: 3px;
    }

    .result-message {
        color: #aab5c8;
        font-size: 14px;
        line-height: 1.6;
        margin-top: 8px;
    }

    .result-confidence {
        margin-top: 18px;
        padding: 11px 14px;
        border-radius: 10px;
        background: rgba(255,255,255,0.04);
        color: #cbd5e1;
        font-size: 13px;
    }


    /* ========================================================
       METRICS
       ======================================================== */

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.12);
        padding: 16px;
        border-radius: 15px;
    }

    div[data-testid="stMetricLabel"] {
        color: #8491aa !important;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 800 !important;
    }


    /* ========================================================
       SECTION CARDS
       ======================================================== */

    .section-card {
        padding: 24px;
        border-radius: 18px;
        background: rgba(15, 23, 42, 0.66);
        border: 1px solid rgba(148, 163, 184, 0.13);
        margin-top: 20px;
        margin-bottom: 20px;
    }

    .info-card {
        padding: 20px;
        border-radius: 16px;
        background: rgba(15, 23, 42, 0.62);
        border: 1px solid rgba(148, 163, 184, 0.12);
        height: 100%;
    }

    .info-card-title {
        color: #f8fafc;
        font-size: 15px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .info-card-text {
        color: #94a3b8;
        font-size: 13px;
        line-height: 1.65;
    }


    /* ========================================================
       EXPANDERS
       ======================================================== */

    div[data-testid="stExpander"] {
        background: rgba(15, 23, 42, 0.58);
        border: 1px solid rgba(148, 163, 184, 0.13);
        border-radius: 14px;
    }


    /* ========================================================
       ALERTS
       ======================================================== */

    div[data-testid="stAlert"] {
        border-radius: 13px;
    }


    /* ========================================================
       DATAFRAME
       ======================================================== */

    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
    }


    /* ========================================================
       DIVIDER
       ======================================================== */

    hr {
        border-color: rgba(148, 163, 184, 0.10) !important;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {
        text-align: center;
        padding: 35px 0 10px 0;
        color: #59667e;
        font-size: 12px;
    }

    .footer strong {
        color: #8b5cf6;
    }


    /* ========================================================
       HIDE STREAMLIT DEFAULT ELEMENTS
       ======================================================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """),
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    render_html("""
    <div class="sidebar-logo">
        <div class="sidebar-brand">
            🛡️ Insurance AI
        </div>

        <div class="sidebar-subtitle">
            ML-Powered Premium Analytics
        </div>
    </div>

    <div class="sidebar-divider"></div>
    """)

    st.markdown(
        '<div class="sidebar-heading">About Project</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-text">
            An AI-powered insurance premium category prediction
            system that uses machine learning to classify
            customers into <span class="sidebar-highlight">
            Low, Medium, or High</span> premium risk categories.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-heading">Technology</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-text">
            <b>Frontend:</b> Streamlit<br>
            <b>Backend:</b> FastAPI<br>
            <b>Model:</b> Random Forest<br>
            <b>ML:</b> Scikit-learn<br>
            <b>Deployment:</b> Docker / AWS
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-heading">Input Features</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-text">
            • Age<br>
            • Weight & Height<br>
            • Annual Income<br>
            • Smoking Status<br>
            • City<br>
            • Occupation
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-heading">Disclaimer</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-text">
            This application provides an AI-based prediction
            for educational and demonstration purposes.
            It is not a substitute for professional insurance
            or financial advice.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HERO SECTION
# ============================================================

render_html("""
<div class="hero-wrapper">

    <div class="hero-badge">
        ✦ AI-POWERED INSURANCE ANALYTICS
    </div>

    <div class="main-title">
        🛡️ Insurance Premium Predictor
    </div>

    <div class="subtitle">
        Intelligent premium category prediction using
        machine learning, health indicators, lifestyle
        factors and customer information.
    </div>

</div>
""")


# ============================================================
# API STATUS
# ============================================================

try:
    response = requests.get(
        f"{API_BASE_URL}/docs",
        timeout=3
    )

    if response.status_code == 200:
        render_html("""
        <div class="status-card">
            ⚡ API Status:
            <span class="status-online">
                Online
            </span>
            &nbsp;•&nbsp;
            FastAPI backend is connected
        </div>
        """)
    else:
        render_html("""
        <div class="status-card">
            ⚡ API Status:
            <span class="status-offline">
                Unavailable
            </span>
        </div>
        """)

except Exception:
    render_html("""
    <div class="status-card">
        ⚡ API Status:
        <span class="status-offline">
            Offline
        </span>
        &nbsp;•&nbsp;
        Start the FastAPI server to enable predictions.
    </div>
    """)


# ============================================================
# MAIN INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">Enter Customer Details</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Provide the customer information below to generate an AI-powered premium category prediction.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# INPUT ROW 1
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=30,
        step=1
    )

with col2:
    weight = st.number_input(
        "Weight (kg)",
        min_value=20.0,
        max_value=200.0,
        value=65.0,
        step=0.5
    )

with col3:
    height = st.number_input(
        "Height (m)",
        min_value=0.5,
        max_value=2.5,
        value=1.65,
        step=0.01
    )


# ============================================================
# INPUT ROW 2
# ============================================================

col4, col5, col6 = st.columns(3)

with col4:
    income_lpa = st.number_input(
        "Annual Income (LPA)",
        min_value=0.0,
        max_value=1000.0,
        value=8.0,
        step=0.5
    )

with col5:
    smoker = st.selectbox(
        "Are you a smoker?",
        [False, True],
        format_func=lambda x: "Yes" if x else "No"
    )

with col6:
    city = st.selectbox(
        "City",
        [
            "Mumbai",
            "Delhi",
            "Bangalore",
            "Hyderabad",
            "Chennai",
            "Kolkata",
            "Pune",
            "Ahmedabad",
            "Jaipur",
            "Lucknow",
            "Kanpur",
            "Nagpur",
            "Indore",
            "Bhopal",
            "Patna"
        ]
    )


# ============================================================
# INPUT ROW 3
# ============================================================

occupation = st.selectbox(
    "Occupation",
    [
        "private_job",
        "government_job",
        "business_owner",
        "student",
        "freelancer",
        "retired",
        "unemployed"
    ],
    format_func=lambda x: {
        "private_job": "Private Job",
        "government_job": "Government Job",
        "business_owner": "Business Owner",
        "student": "Student",
        "freelancer": "Freelancer",
        "retired": "Retired",
        "unemployed": "Unemployed"
    }[x]
)


# ============================================================
# BMI CALCULATION
# ============================================================

bmi = weight / (height ** 2)

if bmi < 18.5:
    bmi_category = "Underweight"
elif bmi < 25:
    bmi_category = "Normal"
elif bmi < 30:
    bmi_category = "Overweight"
else:
    bmi_category = "Obese"


render_html(f"""
<div class="bmi-card">

    <div class="bmi-label">
        Calculated Health Indicator
    </div>

    <div class="bmi-value">
        BMI {bmi:.2f}
    </div>

    <div class="bmi-category">
        {bmi_category}
    </div>

</div>
""")


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_col1, predict_col2, predict_col3 = st.columns(
    [1, 2, 1]
)

with predict_col2:

    predict_clicked = st.button(
        "🚀 Predict Premium Category",
        type="primary",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if predict_clicked:

    payload = {
        "age": age,
        "weight": weight,
        "height": height,
        "income_lpa": income_lpa,
        "smoker": smoker,
        "city": city,
        "occupation": occupation
    }

    try:

        with st.spinner("AI model is analyzing the customer profile..."):

            response = requests.post(
                PREDICT_URL,
                json=payload,
                timeout=15
            )

        if response.status_code == 200:

            prediction = response.json()

            category = prediction.get(
                "predicted_category",
                "Unknown"
            )

            confidence = prediction.get(
                "confidence",
                0
            )

            probabilities = prediction.get(
                "class_probabilities",
                {}
            )

            # ------------------------------------------------
            # SAVE LAST PREDICTION
            # ------------------------------------------------

            st.session_state.last_prediction = prediction

            # ------------------------------------------------
            # PREDICTION MESSAGE
            # ------------------------------------------------

            if category.lower() == "low":

                result_icon = "🟢"

                result_message = (
                    "The customer profile indicates a relatively "
                    "low insurance premium risk based on the "
                    "model's learned patterns."
                )

                result_class = "result-low"

            elif category.lower() == "medium":

                result_icon = "🟡"

                result_message = (
                    "The customer profile indicates a moderate "
                    "insurance premium risk. Some lifestyle or "
                    "health-related factors may influence the "
                    "predicted premium category."
                )

                result_class = "result-medium"

            else:

                result_icon = "🔴"

                result_message = (
                    "The customer profile indicates a relatively "
                    "high insurance premium risk based on the "
                    "model's learned patterns."
                )

                result_class = "result-high"


            # ------------------------------------------------
            # SAVE HISTORY
            # ------------------------------------------------

            history_record = {
                "Timestamp": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "Age": age,
                "BMI": round(bmi, 2),
                "Income (LPA)": income_lpa,
                "Smoker": "Yes" if smoker else "No",
                "City": city,
                "Occupation": {
                    "private_job": "Private Job",
                    "government_job": "Government Job",
                    "business_owner": "Business Owner",
                    "student": "Student",
                    "freelancer": "Freelancer",
                    "retired": "Retired",
                    "unemployed": "Unemployed"
                }.get(occupation, occupation),
                "Prediction": category,
                "Confidence": round(
                    float(confidence) * 100,
                    2
                )
            }

            st.session_state.prediction_history.append(
                history_record
            )


            # ------------------------------------------------
            # RESULT CARD
            # ------------------------------------------------

            render_html(f"""
            <div class="premium-result-card {result_class}">

                <div class="result-small-label">
                    AI Prediction Result
                </div>

                <div class="result-icon">
                    {result_icon}
                </div>

                <div class="result-category">
                    {category}
                </div>

                <div class="result-message">
                    {result_message}
                </div>

                <div class="result-confidence">
                    <b>Model Confidence:</b>
                    {float(confidence) * 100:.2f}%
                </div>

            </div>
            """)


            # ------------------------------------------------
            # CLASS PROBABILITIES
            # ------------------------------------------------

            st.markdown("---")

            st.markdown(
                '<div class="section-title">📊 Prediction Probabilities</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-description">'
                'Probability assigned by the machine learning model to each premium category.'
                '</div>',
                unsafe_allow_html=True
            )

            prob_cols = st.columns(
                len(probabilities)
            )

            for i, (label, value) in enumerate(
                probabilities.items()
            ):

                with prob_cols[i]:

                    st.metric(
                        label,
                        f"{float(value) * 100:.2f}%"
                    )

                    st.progress(
                        min(
                            max(float(value), 0.0),
                            1.0
                        )
                    )


            # ------------------------------------------------
            # WHY THIS PREDICTION
            # ------------------------------------------------

            st.markdown("---")

            st.markdown(
                '<div class="section-title">🧠 Why this prediction?</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-description">'
                'Important customer factors considered by the prediction pipeline.'
                '</div>',
                unsafe_allow_html=True
            )

            why_col1, why_col2, why_col3, why_col4 = st.columns(4)

            with why_col1:
                st.metric(
                    "Age",
                    f"{age} years"
                )

            with why_col2:
                st.metric(
                    "BMI",
                    f"{bmi:.2f}"
                )

            with why_col3:
                st.metric(
                    "Income",
                    f"{income_lpa:.1f} LPA"
                )

            with why_col4:
                st.metric(
                    "Smoker",
                    "Yes" if smoker else "No"
                )


        else:

            try:
                error_data = response.json()
            except Exception:
                error_data = response.text

            st.error(
                f"Prediction failed "
                f"(HTTP {response.status_code}): "
                f"{error_data}"
            )


    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Could not connect to the FastAPI backend. "
            "Please start it using:"
        )

        st.code(
            "uvicorn app:app --reload",
            language="powershell"
        )


    except requests.exceptions.Timeout:

        st.error(
            "⏱️ The API request timed out. "
            "Please make sure the FastAPI server is running correctly."
        )


    except Exception as e:

        st.error(
            f"❌ Unexpected error: {str(e)}"
        )


# ============================================================
# NAVIGATION
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">Explore Model Analytics</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Open a section below to explore the model, health analysis, performance and architecture.'
    '</div>',
    unsafe_allow_html=True
)


nav1, nav2, nav3, nav4, nav5 = st.columns(5)


with nav1:
    if st.button(
        "🩺 Health Summary",
        use_container_width=True
    ):
        st.session_state.active_section = "health"
        st.rerun()


with nav2:
    if st.button(
        "🧠 Model Insights",
        use_container_width=True
    ):
        st.session_state.active_section = "insights"
        st.rerun()


with nav3:
    if st.button(
        "📈 Model Performance",
        use_container_width=True
    ):
        st.session_state.active_section = "performance"
        st.rerun()


with nav4:
    if st.button(
        "🗂️ Prediction History",
        use_container_width=True
    ):
        st.session_state.active_section = "history"
        st.rerun()


with nav5:
    if st.button(
        "🏗️ Architecture",
        use_container_width=True
    ):
        st.session_state.active_section = "architecture"
        st.rerun()


# ============================================================
# HEALTH SUMMARY
# ============================================================

if st.session_state.active_section == "health":

    st.markdown("---")

    st.markdown(
        '<div class="section-title">🩺 Health Summary</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'A quick overview of the health-related inputs used by the system.'
        '</div>',
        unsafe_allow_html=True
    )

    h1, h2, h3, h4 = st.columns(4)

    with h1:
        st.metric(
            "BMI",
            f"{bmi:.2f}"
        )

    with h2:
        st.metric(
            "BMI Category",
            bmi_category
        )

    with h3:
        st.metric(
            "Age",
            f"{age} years"
        )

    with h4:
        st.metric(
            "Smoking",
            "Yes" if smoker else "No"
        )

    st.markdown("")

    render_html("""
    <div class="info-card">

        <div class="info-card-title">
            📌 Health Interpretation
        </div>

        <div class="info-card-text">
            BMI is calculated using the customer's weight and height.
            The system combines BMI with smoking status and age-related
            information to derive lifestyle and health-related features
            before sending the data to the machine learning model.
        </div>

    </div>
    """)


# ============================================================
# MODEL INSIGHTS
# ============================================================

elif st.session_state.active_section == "insights":

    st.markdown("---")

    st.markdown(
        '<div class="section-title">🧠 Model Insights</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Explore the trained model configuration and feature importance.'
        '</div>',
        unsafe_allow_html=True
    )

    try:

        model_info_response = requests.get(
            MODEL_INFO_URL,
            timeout=10
        )

        feature_response = requests.get(
            FEATURE_URL,
            timeout=10
        )

        if model_info_response.status_code == 200:

            model_info = model_info_response.json()

            m1, m2, m3, m4 = st.columns(4)

            with m1:
                st.metric(
                    "Model",
                    model_info.get(
                        "model_type",
                        "Random Forest"
                    )
                )

            with m2:
                st.metric(
                    "Features",
                    model_info.get(
                        "number_of_features",
                        "N/A"
                    )
                )

            with m3:
                st.metric(
                    "Version",
                    model_info.get(
                        "model_version",
                        "N/A"
                    )
                )

            with m4:
                classes = model_info.get(
                    "classes",
                    []
                )

                st.metric(
                    "Classes",
                    len(classes)
                )


            st.markdown("---")

            st.markdown(
                '<div class="section-title">⚙️ Model Configuration</div>',
                unsafe_allow_html=True
            )

            st.json(model_info)


        if feature_response.status_code == 200:

            feature_data = feature_response.json()

            st.markdown("---")

            st.markdown(
                '<div class="section-title">📊 Feature Importance</div>',
                unsafe_allow_html=True
            )

            if isinstance(feature_data, dict):

                importance_df = pd.DataFrame(
                    {
                        "Feature": list(
                            feature_data.keys()
                        ),
                        "Importance": list(
                            feature_data.values()
                        )
                    }
                )

            elif isinstance(feature_data, list):

                importance_df = pd.DataFrame(
                    feature_data
                )

            else:

                importance_df = pd.DataFrame()


            if not importance_df.empty:

                importance_df = importance_df.sort_values(
                    "Importance",
                    ascending=False
                )

                st.bar_chart(
                    importance_df.set_index(
                        "Feature"
                    )
                )

                st.markdown(
                    '<div class="section-title">'
                    'All Feature Importance'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.dataframe(
                    importance_df,
                    use_container_width=True,
                    hide_index=True
                )

        else:

            st.warning(
                "Feature importance endpoint is unavailable."
            )


    except requests.exceptions.ConnectionError:

        st.error(
            "Could not connect to FastAPI. "
            "Start the backend first."
        )

    except Exception as e:

        st.error(
            f"Unable to load model information: {str(e)}"
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif st.session_state.active_section == "performance":

    st.markdown("---")

    st.markdown(
        '<div class="section-title">📈 Model Performance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Evaluation results using cross-validation and a held-out test split.'
        '</div>',
        unsafe_allow_html=True
    )

    # ========================================================
    # READ EVALUATION RESULTS
    # ========================================================

    dataset_size = evaluation_results.get(
        "dataset_size",
        0
    )

    training_size = evaluation_results.get(
        "training_size",
        0
    )

    testing_size = evaluation_results.get(
        "testing_size",
        0
    )

    cv_accuracy = evaluation_results.get(
        "cv_accuracy",
        0
    )

    cv_std = evaluation_results.get(
        "cv_std",
        0
    )

    test_accuracy = evaluation_results.get(
        "test_accuracy",
        0
    )

    test_precision = evaluation_results.get(
        "test_precision",
        0
    )

    test_recall = evaluation_results.get(
        "test_recall",
        0
    )

    test_f1 = evaluation_results.get(
        "test_f1",
        0
    )

    cm = evaluation_results.get(
        "confusion_matrix",
        None
    )

    labels = evaluation_results.get(
        "labels",
        ["High", "Low", "Medium"]
    )

    report = evaluation_results.get(
        "classification_report",
        {}
    )


    # ========================================================
    # DATASET INFORMATION
    # ========================================================

    dataset_col1, dataset_col2, dataset_col3 = st.columns(3)

    with dataset_col1:

        st.metric(
            "Dataset Size",
            dataset_size
        )

    with dataset_col2:

        st.metric(
            "Training Samples",
            training_size
        )

    with dataset_col3:

        st.metric(
            "Test Samples",
            testing_size
        )


    st.markdown("")


    # ========================================================
    # CROSS VALIDATION
    # ========================================================

    st.markdown(
        '<div class="section-title">🔄 Cross-Validation</div>',
        unsafe_allow_html=True
    )

    cv1, cv2 = st.columns(2)

    with cv1:

        st.metric(
            "5-Fold CV Accuracy",
            f"{float(cv_accuracy) * 100:.2f}%"
        )

    with cv2:

        st.metric(
            "Standard Deviation",
            f"{float(cv_std) * 100:.2f}%"
        )


    st.markdown("---")


    # ========================================================
    # TEST SET PERFORMANCE
    # ========================================================

    st.markdown(
        '<div class="section-title">🎯 Test Set Performance</div>',
        unsafe_allow_html=True
    )

    test1, test2, test3, test4 = st.columns(4)

    with test1:

        st.metric(
            "Accuracy",
            f"{float(test_accuracy) * 100:.2f}%"
        )

    with test2:

        st.metric(
            "Precision",
            f"{float(test_precision) * 100:.2f}%"
        )

    with test3:

        st.metric(
            "Recall",
            f"{float(test_recall) * 100:.2f}%"
        )

    with test4:

        st.metric(
            "F1 Score",
            f"{float(test_f1) * 100:.2f}%"
        )


    st.markdown("---")


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.markdown(
        '<div class="section-title">🧩 Confusion Matrix</div>',
        unsafe_allow_html=True
    )

    # IMPORTANT:
    # Do NOT use:
    #
    # if cm:
    #
    # because cm is a NumPy array.

    if cm is not None and len(cm) > 0:

        try:

            cm_df = pd.DataFrame(
                cm,
                index=labels,
                columns=labels
            )

            cm_df.index.name = "Actual"
            cm_df.columns.name = "Predicted"

            st.dataframe(
                cm_df,
                use_container_width=True
            )

        except Exception as e:

            st.error(
                f"Unable to display confusion matrix: {e}"
            )

    else:

        st.info(
            "Confusion matrix is not available."
        )


    st.markdown("---")


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    st.markdown(
        '<div class="section-title">📋 Classification Report</div>',
        unsafe_allow_html=True
    )

    if isinstance(report, dict):

        report_rows = []

        # Display classes in the same order as the model labels
        for label in labels:

            if label not in report:
                continue

            metrics = report[label]

            if not isinstance(metrics, dict):
                continue

            report_rows.append(
                {
                    "Class": label,

                    "Precision": (
                        f"{float(metrics.get('precision', 0)) * 100:.2f}%"
                    ),

                    "Recall": (
                        f"{float(metrics.get('recall', 0)) * 100:.2f}%"
                    ),

                    "F1 Score": (
                        f"{float(metrics.get('f1-score', 0)) * 100:.2f}%"
                    ),

                    "Support": int(
                        metrics.get("support", 0)
                    )
                }
            )


        if report_rows:

            report_df = pd.DataFrame(
                report_rows
            )

            st.dataframe(
                report_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "Classification report is not available."
            )

    else:

        st.info(
            "Classification report is not available."
        )


    # ========================================================
    # PERFORMANCE SUMMARY
    # ========================================================

    st.markdown("---")

    render_html(f"""
    <div class="info-card">

        <div class="info-card-title">
            💡 Performance Summary
        </div>

        <div class="info-card-text">

            The model achieved a
            <b>{float(cv_accuracy) * 100:.2f}%</b>
            average accuracy across 5-fold cross-validation.

            The held-out test set achieved an accuracy of
            <b>{float(test_accuracy) * 100:.2f}%</b>.

            The weighted precision was
            <b>{float(test_precision) * 100:.2f}%</b>,

            while the weighted recall was
            <b>{float(test_recall) * 100:.2f}%</b>.

            The weighted F1 score was
            <b>{float(test_f1) * 100:.2f}%</b>,

            indicating the overall balance between precision
            and recall across the three premium categories.

        </div>

    </div>
    """)


# ============================================================
# PREDICTION HISTORY
# ============================================================

elif st.session_state.active_section == "history":

    st.markdown("---")

    st.markdown(
        '<div class="section-title">🗂️ Prediction History</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Review predictions generated during the current Streamlit session.'
        '</div>',
        unsafe_allow_html=True
    )


    if st.session_state.prediction_history:

        history_df = pd.DataFrame(
            st.session_state.prediction_history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )


        # ----------------------------------------------------
        # CSV DOWNLOAD
        # ----------------------------------------------------

        csv_buffer = io.StringIO()

        writer = csv.DictWriter(
            csv_buffer,
            fieldnames=history_df.columns
        )

        writer.writeheader()

        writer.writerows(
            st.session_state.prediction_history
        )

        download_col1, download_col2 = st.columns(
            [1, 1]
        )

        with download_col1:

            st.download_button(
                label="⬇️ Download History as CSV",
                data=csv_buffer.getvalue(),
                file_name="insurance_prediction_history.csv",
                mime="text/csv",
                use_container_width=True
            )

        with download_col2:

            if st.button(
                "🗑️ Clear Prediction History",
                use_container_width=True
            ):

                st.session_state.prediction_history = []

                st.rerun()


    else:

        render_html("""
        <div class="info-card">

            <div class="info-card-title">
                📭 No predictions yet
            </div>

            <div class="info-card-text">
                Generate your first prediction using the
                prediction button above. Your predictions will
                automatically appear here during this session.
            </div>

        </div>
        """)


# ============================================================
# ARCHITECTURE
# ============================================================

elif st.session_state.active_section == "architecture":

    st.markdown("---")

    st.markdown(
        '<div class="section-title">🏗️ Project Architecture</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'End-to-end architecture of the Insurance AI prediction system.'
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # ARCHITECTURE CARDS
    # --------------------------------------------------------

    arch1, arch2 = st.columns(2)

    with arch1:

        render_html("""
        <div class="info-card">

            <div class="info-card-title">
                🎨 1. Streamlit Frontend
            </div>

            <div class="info-card-text">
                Collects customer information such as age, weight,
                height, income, smoking status, city and occupation.
                It also displays predictions, probabilities,
                model insights and evaluation results.
            </div>

        </div>
        """)


    with arch2:

        render_html("""
        <div class="info-card">

            <div class="info-card-title">
                ⚡ 2. FastAPI Backend
            </div>

            <div class="info-card-text">
                Receives customer information through REST API
                endpoints, validates the request using Pydantic
                and sends engineered features to the trained
                machine learning pipeline.
            </div>

        </div>
        """)


    st.markdown("")

    arch3, arch4 = st.columns(2)

    with arch3:

        render_html("""
        <div class="info-card">

            <div class="info-card-title">
                🤖 3. Machine Learning Model
            </div>

            <div class="info-card-text">
                A Random Forest classification pipeline processes
                engineered features including BMI, age group,
                lifestyle risk, city tier, income and occupation.
                The model predicts Low, Medium or High premium category.
            </div>

        </div>
        """)


    with arch4:

        render_html("""
        <div class="info-card">

            <div class="info-card-title">
                ☁️ 4. Deployment
            </div>

            <div class="info-card-text">
                The application is containerized using Docker and
                can be deployed to cloud infrastructure such as AWS.
                This provides a scalable path from local development
                to production deployment.
            </div>

        </div>
        """)


    # --------------------------------------------------------
    # FLOW
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown(
        '<div class="section-title">🔄 Prediction Flow</div>',
        unsafe_allow_html=True
    )

    render_html("""
    <div class="info-card">

        <div class="info-card-text"
             style="font-size:15px; line-height:2.1;">

            👤 Customer Input
            &nbsp; → &nbsp;
            🎨 Streamlit
            &nbsp; → &nbsp;
            ⚡ FastAPI
            &nbsp; → &nbsp;
            🧮 Feature Engineering
            &nbsp; → &nbsp;
            🤖 Random Forest
            &nbsp; → &nbsp;
            📊 Premium Category
            &nbsp; → &nbsp;
            📱 Streamlit Result

        </div>

    </div>
    """)


    # --------------------------------------------------------
    # TECHNOLOGY STACK
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown(
        '<div class="section-title">🧰 Technology Stack</div>',
        unsafe_allow_html=True
    )

    tech_df = pd.DataFrame(
        {
            "Layer": [
                "Frontend",
                "Backend",
                "Machine Learning",
                "Data Processing",
                "Model Serialization",
                "Containerization",
                "Deployment"
            ],
            "Technology": [
                "Streamlit",
                "FastAPI",
                "Scikit-learn Random Forest",
                "Pandas / NumPy",
                "Joblib",
                "Docker",
                "AWS"
            ]
        }
    )

    st.dataframe(
        tech_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

render_html("""
<div class="footer">

    Built with ❤️ using
    <strong>Python</strong>,
    <strong>FastAPI</strong>,
    <strong>Streamlit</strong>
    and
    <strong>Machine Learning</strong>.

    <br><br>

    Insurance AI • Premium Prediction System

</div>
""")