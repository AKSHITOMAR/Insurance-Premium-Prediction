import streamlit as st
import requests
import csv
import io
from datetime import datetime


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Insurance Premium Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONFIGURATION
# ============================================================

API_BASE_URL = "http://127.0.0.1:8000"

PREDICT_URL = f"{API_BASE_URL}/predict"
FEATURE_URL = f"{API_BASE_URL}/feature-importance"


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
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #777;
        margin-bottom: 25px;
    }

    .result-box {
        padding: 28px;
        border-radius: 18px;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
        border: 1px solid #ddd;
        background-color: #f8f9fa;
    }

    .category {
        font-size: 40px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .confidence {
        font-size: 18px;
        color: #555;
        margin-top: 5px;
    }

    .metric-card {
        padding: 18px;
        border: 1px solid #e0e0e0;
        border-radius: 15px;
        background-color: #fafafa;
        text-align: center;
        min-height: 110px;
    }

    .metric-title {
        font-size: 14px;
        color: #777;
        margin-bottom: 7px;
    }

    .metric-value {
        font-size: 26px;
        font-weight: 800;
    }

    .section-card {
        padding: 20px;
        border: 1px solid #e2e2e2;
        border-radius: 15px;
        background-color: #fafafa;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    .footer {
        text-align: center;
        color: #888;
        font-size: 13px;
        margin-top: 40px;
        padding: 20px;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🛡️ Insurance AI")

    st.caption("Machine Learning Prediction System")

    st.markdown("---")

    # ABOUT
    with st.expander("📖 About Project"):

        st.write(
            "This AI-powered application predicts an "
            "insurance premium category using health, "
            "lifestyle, income and demographic information."
        )

        st.markdown("**Model:** Machine Learning Classification")

        st.markdown("**Frontend:** Streamlit")

        st.markdown("**Backend:** FastAPI")

        st.markdown("**ML:** Scikit-learn")

        st.markdown("**Deployment:** Docker / AWS")


    # FEATURES
    with st.expander("📊 Input Features"):

        st.markdown(
            """
            • Age  
            • Weight  
            • Height  
            • Annual Income  
            • Smoking Status  
            • City  
            • Occupation
            """
        )


    # DISCLAIMER
    with st.expander("⚠️ Disclaimer"):

        st.caption(
            "This project is for educational and "
            "demonstration purposes only. Predictions "
            "should not be considered professional "
            "insurance advice."
        )


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🛡️ Insurance Premium Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered insurance premium category prediction'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# API STATUS
# ============================================================

try:

    api_response = requests.get(
        f"{API_BASE_URL}/docs",
        timeout=2
    )

    if api_response.status_code == 200:
        st.success("🟢 FastAPI Backend Online")

    else:
        st.warning("🟡 FastAPI Backend Responded Unexpectedly")

except requests.exceptions.RequestException:

    st.error(
        "🔴 FastAPI Backend Offline — "
        "Start Uvicorn before making a prediction."
    )


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown("### 👤 Enter Your Details")

input_col1, input_col2 = st.columns(2)


with input_col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=31,
        step=1
    )

    weight = st.number_input(
        "Weight (kg)",
        min_value=1.0,
        max_value=300.0,
        value=80.0,
        step=0.5
    )

    height = st.number_input(
        "Height (m)",
        min_value=0.5,
        max_value=2.5,
        value=1.83,
        step=0.01
    )


with input_col2:

    income_lpa = st.number_input(
        "Annual Income (LPA)",
        min_value=0.0,
        max_value=1000.0,
        value=10.0,
        step=0.5
    )

    smoker = st.selectbox(
        "Are you a smoker?",
        [False, True],
        format_func=lambda x: "Yes" if x else "No"
    )

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
            "Other"
        ]
    )

    occupation = st.selectbox(
        "Occupation",
        [
            "private_job",
            "government_job",
            "business_owner",
            "student",
            "retired"
        ],
        format_func=lambda x: x.replace("_", " ").title()
    )


# ============================================================
# BMI
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


# ============================================================
# BMI PREVIEW
# ============================================================

st.markdown("---")

bmi_col1, bmi_col2, bmi_col3 = st.columns(3)

with bmi_col1:

    st.metric(
        "⚖️ BMI",
        f"{bmi:.2f}"
    )

with bmi_col2:

    st.metric(
        "BMI Category",
        bmi_category
    )

with bmi_col3:

    st.metric(
        "🚬 Smoker",
        "Yes" if smoker else "No"
    )


# ============================================================
# PREDICT BUTTON
# ============================================================

st.markdown("###")

predict_button = st.button(
    "🔮 Predict Premium Category",
    use_container_width=True,
    type="primary"
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

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

        with st.spinner("🤖 Analyzing your details..."):

            response = requests.post(
                PREDICT_URL,
                json=payload,
                timeout=10
            )

        if response.status_code == 200:

            prediction = response.json()

            category = prediction["predicted_category"]
            confidence = prediction["confidence"]
            probabilities = prediction["class_probabilities"]

            st.session_state.last_prediction = prediction

            # Save history
            st.session_state.prediction_history.append(
                {
                    "Time": datetime.now().strftime(
                        "%d-%m-%Y %H:%M"
                    ),
                    "Age": age,
                    "BMI": round(bmi, 2),
                    "Income (LPA)": income_lpa,
                    "Smoker": "Yes" if smoker else "No",
                    "City": city,
                    "Occupation": occupation.replace(
                        "_", " "
                    ).title(),
                    "Prediction": category,
                    "Confidence": f"{confidence:.0%}"
                }
            )


            # ================================================
            # CATEGORY
            # ================================================

            if category.lower() == "low":

                result_icon = "🟢"
                result_message = "Lower premium category"

            elif category.lower() == "medium":

                result_icon = "🟡"
                result_message = "Moderate premium category"

            else:

                result_icon = "🔴"
                result_message = "Higher premium category"


            # ================================================
            # RESULT
            # ================================================

            st.markdown("---")

            st.markdown("### 🎯 Prediction Result")

            if category.lower() == "low":

                st.success(
                    f"🟢 **{category}** — {result_message}"
                )

            elif category.lower() == "medium":

                st.warning(
                    f"🟡 **{category}** — {result_message}"
                )

            else:

                st.error(
                    f"🔴 **{category}** — {result_message}"
                )


            # ================================================
            # RESULT METRICS
            # ================================================

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Predicted Category",
                    category
                )

            with col2:

                st.metric(
                    "Confidence",
                    f"{confidence:.0%}"
                )

            with col3:

                st.metric(
                    "BMI Category",
                    bmi_category
                )


            # ================================================
            # PROBABILITIES
            # ================================================

            st.markdown("---")

            st.subheader("📊 Prediction Probabilities")

            probability_columns = st.columns(
                len(probabilities)
            )

            for index, (label, probability) in enumerate(
                probabilities.items()
            ):

                with probability_columns[index]:

                    st.metric(
                        label,
                        f"{probability:.1%}"
                    )

                    st.progress(
                        min(
                            max(
                                float(probability),
                                0.0
                            ),
                            1.0
                        )
                    )


            # ================================================
            # WHY THIS PREDICTION
            # ================================================

            st.markdown("---")

            st.subheader("💡 Why this prediction?")

            reasons = []


            if smoker:

                reasons.append(
                    "🚬 Smoking status may increase "
                    "the premium category."
                )

            else:

                reasons.append(
                    "🚭 Non-smoker status is generally "
                    "a positive factor."
                )


            if bmi < 18.5:

                reasons.append(
                    "⚖️ BMI is in the underweight range."
                )

            elif bmi < 25:

                reasons.append(
                    "⚖️ BMI is in the normal range."
                )

            elif bmi < 30:

                reasons.append(
                    "⚖️ BMI is in the overweight range."
                )

            else:

                reasons.append(
                    "⚖️ BMI is in the obese range."
                )


            if age < 30:

                reasons.append(
                    "👤 Age is relatively low."
                )

            elif age < 50:

                reasons.append(
                    "👤 Age falls within the middle-age range."
                )

            else:

                reasons.append(
                    "👤 Higher age can contribute to "
                    "a higher premium category."
                )


            reasons.append(
                f"💰 Annual income: ₹{income_lpa:.1f} LPA"
            )

            reasons.append(
                f"📍 City: {city}"
            )

            reasons.append(
                f"💼 Occupation: "
                f"{occupation.replace('_', ' ').title()}"
            )


            reason_col1, reason_col2 = st.columns(2)

            for index, reason in enumerate(reasons):

                if index % 2 == 0:

                    with reason_col1:
                        st.info(reason)

                else:

                    with reason_col2:
                        st.info(reason)


        else:

            st.error(
                f"❌ API Error: {response.status_code}"
            )

            try:
                st.json(response.json())

            except Exception:
                st.write(response.text)


    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Could not connect to FastAPI."
        )

        st.info(
            "Run: `uvicorn app:app --reload`"
        )


    except requests.exceptions.Timeout:

        st.error(
            "⏱️ API request timed out."
        )


    except Exception as e:

        st.error(
            f"❌ Something went wrong: {str(e)}"
        )


# ============================================================
# NAVIGATION BUTTONS
# ============================================================

st.markdown("---")

st.markdown("### 📌 Explore More")


nav1, nav2, nav3, nav4 = st.columns(4)


with nav1:

    if st.button(
        "👤 Health Summary",
        use_container_width=True
    ):

        if st.session_state.active_section == "health":
            st.session_state.active_section = None
        else:
            st.session_state.active_section = "health"

        st.rerun()


with nav2:

    if st.button(
        "📊 Model Insights",
        use_container_width=True
    ):

        if st.session_state.active_section == "model":
            st.session_state.active_section = None
        else:
            st.session_state.active_section = "model"

        st.rerun()


with nav3:

    if st.button(
        "📜 Prediction History",
        use_container_width=True
    ):

        if st.session_state.active_section == "history":
            st.session_state.active_section = None
        else:
            st.session_state.active_section = "history"

        st.rerun()


with nav4:

    if st.button(
        "🏗️ Architecture",
        use_container_width=True
    ):

        if st.session_state.active_section == "architecture":
            st.session_state.active_section = None
        else:
            st.session_state.active_section = "architecture"

        st.rerun()


# ============================================================
# HEALTH SUMMARY
# ============================================================

if st.session_state.active_section == "health":

    st.markdown("---")

    st.subheader("👤 Health Summary")

    st.markdown(
        '<div class="section-card">',
        unsafe_allow_html=True
    )

    health1, health2, health3, health4 = st.columns(4)

    with health1:

        st.metric(
            "Age",
            age
        )

    with health2:

        st.metric(
            "Weight",
            f"{weight:.1f} kg"
        )

    with health3:

        st.metric(
            "Height",
            f"{height:.2f} m"
        )

    with health4:

        st.metric(
            "BMI",
            f"{bmi:.2f}"
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.info(
        f"Your BMI is **{bmi:.2f}**, which falls under "
        f"the **{bmi_category}** category."
    )


# ============================================================
# MODEL INSIGHTS
# ============================================================

if st.session_state.active_section == "model":

    st.markdown("---")

    st.subheader("📊 Model Insights")

    # ========================================================
    # MODEL OVERVIEW
    # ========================================================

    try:

        model_info_response = requests.get(
            f"{API_BASE_URL}/model-info",
            timeout=5
        )

        if model_info_response.status_code == 200:

            model_info = model_info_response.json()

            info_col1, info_col2, info_col3, info_col4 = st.columns(4)

            with info_col1:
                st.metric(
                    "🤖 Model",
                    model_info["model_type"]
                )

            with info_col2:
                st.metric(
                    "🔢 Features",
                    model_info["number_of_features"]
                )

            with info_col3:
                st.metric(
                    "📦 Version",
                    model_info["model_version"]
                )

            with info_col4:
                st.metric(
                    "🎯 Classes",
                    len(model_info["classes"])
                )

            st.markdown("#### ⚙️ Model Pipeline")

            pipeline_text = " → ".join(
                step["type"]
                for step in model_info["pipeline_steps"]
            )

            st.info(pipeline_text)

        else:

            st.warning(
                "Could not load model information."
            )

    except requests.exceptions.RequestException:

        st.warning(
            "FastAPI is not available."
        )

    except Exception as e:

        st.warning(
            f"Could not load model information: {str(e)}"
        )

    try:

        importance_response = requests.get(
            FEATURE_URL,
            timeout=5
        )

        if importance_response.status_code == 200:

            importance_data = importance_response.json()

            feature_importance = (
                importance_data["feature_importance"]
            )

            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )

            st.write(
                "The chart below shows the features that "
                "contribute most to the model."
            )

            top_features = dict(
                sorted_features[:10]
            )

            chart_data = {
                feature.replace("num__", "")
                       .replace("cat__", "")
                       .replace("_", " ")
                       .title(): importance
                for feature, importance
                in top_features.items()
            }

            st.bar_chart(
                chart_data,
                horizontal=True
            )


            with st.expander(
                "🔍 View All Feature Importance"
            ):

                for feature, importance in sorted_features:

                    display_name = (
                        feature
                        .replace("num__", "")
                        .replace("cat__", "")
                        .replace("_", " ")
                        .title()
                    )

                    st.write(
                        f"**{display_name}** — "
                        f"{importance:.1%}"
                    )

                    st.progress(
                        min(
                            max(
                                float(importance),
                                0.0
                            ),
                            1.0
                        )
                    )

        else:

            st.warning(
                "Could not load feature importance."
            )

    except requests.exceptions.RequestException:

        st.warning(
            "FastAPI is not available."
        )

    except Exception as e:

        st.warning(
            f"Could not load model insights: {str(e)}"
        )


# ============================================================
# PREDICTION HISTORY
# ============================================================

if st.session_state.active_section == "history":

    st.markdown("---")

    st.subheader("📜 Prediction History")

    if st.session_state.prediction_history:

        st.caption(
            f"{len(st.session_state.prediction_history)} "
            "prediction(s) recorded in this session."
        )

        st.dataframe(
            st.session_state.prediction_history,
            use_container_width=True,
            hide_index=True
        )


        # CSV
        csv_buffer = io.StringIO()

        fieldnames = list(
            st.session_state.prediction_history[0].keys()
        )

        writer = csv.DictWriter(
            csv_buffer,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            st.session_state.prediction_history
        )


        download_col, clear_col = st.columns(2)


        with download_col:

            st.download_button(
                "📥 Download CSV",
                data=csv_buffer.getvalue(),
                file_name="insurance_prediction_history.csv",
                mime="text/csv",
                use_container_width=True
            )


        with clear_col:

            if st.button(
                "🗑️ Clear History",
                use_container_width=True
            ):

                st.session_state.prediction_history = []

                st.rerun()

    else:

        st.info(
            "📭 No prediction history yet. "
            "Make a prediction first."
        )

# ============================================================
# PROJECT ARCHITECTURE
# ============================================================

if st.session_state.active_section == "architecture":

    st.markdown("---")

    st.subheader("🏗️ Project Architecture")

    st.write(
        "The application follows an end-to-end machine learning "
        "deployment architecture."
    )

    # ========================================================
    # ARCHITECTURE COMPONENTS
    # ========================================================

    architecture_col1, architecture_col2 = st.columns(2)

    with architecture_col1:

        st.info(
            """
            ### 🖥️ 1. Streamlit Frontend

            Provides the interactive user interface.

            • Collects user inputs
            • Displays BMI analysis
            • Shows predictions and probabilities
            • Displays prediction history
            • Provides model insights
            """
        )

        st.info(
            """
            ### 🤖 3. Machine Learning Model

            A Scikit-learn classification pipeline processes
            the engineered features and predicts the insurance
            premium category.

            **Output: Low / Medium / High**
            """
        )

    with architecture_col2:

        st.info(
            """
            ### ⚡ 2. FastAPI Backend

            Provides REST API services between the frontend
            and machine learning model.

            **POST /predict**

            **GET /health**

            **GET /model-info**

            **GET /feature-importance**
            """
        )

        st.info(
            """
            ### 🐳 4. Deployment

            The application is containerized using Docker
            and can be deployed to cloud infrastructure
            such as AWS.
            """
        )

    # ========================================================
    # DATA FLOW
    # ========================================================

    st.markdown("### 🔄 End-to-End Data Flow")

    st.code(
        """
User
  ↓
Streamlit Frontend
  ↓
FastAPI REST API
  ↓
Input Validation & Feature Engineering
  ↓
Preprocessing Pipeline
  ↓
Random Forest Classifier
  ↓
Prediction + Probabilities
  ↓
FastAPI Response
  ↓
Streamlit Dashboard
        """,
        language="text"
    )



# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🛡️ Insurance Premium Prediction |
        Machine Learning Project |
        Built with Python, Streamlit & FastAPI
        <br><br>
        ⚠️ For educational and demonstration purposes only.
    </div>
    """,
    unsafe_allow_html=True
)
