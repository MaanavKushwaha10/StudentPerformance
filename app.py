import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')
 
# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Exam Score Predictor",
    page_icon="🎓",
    layout="wide",
)
 
# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1e3a5f;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background: linear-gradient(135deg, #1e3a5f, #2e86de);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin-top: 1.5rem;
    }
    .result-score {
        font-size: 4rem;
        font-weight: 800;
        line-height: 1;
    }
    .result-label {
        font-size: 1.1rem;
        opacity: 0.85;
        margin-top: 0.5rem;
    }
    .grade-badge {
        display: inline-block;
        padding: 0.3rem 1.2rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.3rem;
        margin-top: 1rem;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1e3a5f;
        border-left: 4px solid #2e86de;
        padding-left: 0.6rem;
        margin-bottom: 0.8rem;
        margin-top: 1.2rem;
    }
    .info-card {
        background: #f0f4fa;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
        font-size: 0.9rem;
        color: #333;
    }
    div[data-testid="stVerticalBlock"] > div:has(> div.result-box) {
        padding: 0;
    }
</style>
""", unsafe_allow_html=True)
 
 
# ─── Model Loading ───────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the pre-trained pipeline from model.pkl if available."""
    model_path = "model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f), True
    return None, False
 
 
model_pipeline, model_loaded = load_model()
 
 
# ─── Prediction Helper ───────────────────────────────────────────────────────
def predict_score(inputs: dict):
    """Run prediction using the loaded model or a simple heuristic fallback."""
    df_input = pd.DataFrame([inputs])
    if model_loaded:
        pred = model_pipeline.predict(df_input)[0]
    else:
        # Lightweight heuristic (demo only – replace with real model)
        score = 50
        score += inputs["Hours_Studied"] * 0.3
        score += (inputs["Attendance"] - 60) * 0.1
        score += inputs["Previous_Scores"] * 0.15
        score += inputs["Tutoring_Sessions"] * 1.2
        score += inputs["Sleep_Hours"] * 0.4
        score += inputs["Physical_Activity"] * 0.3
        level_map = {"Low": 0, "Medium": 2, "High": 4}
        score += level_map.get(inputs["Parental_Involvement"], 0)
        score += level_map.get(inputs["Access_to_Resources"], 0)
        score += level_map.get(inputs["Motivation_Level"], 0)
        score += level_map.get(inputs["Family_Income"], 0)
        score += level_map.get(inputs["Teacher_Quality"], 0)
        peer_map = {"Negative": -2, "Neutral": 0, "Positive": 2}
        score += peer_map.get(inputs["Peer_Influence"], 0)
        if inputs["Internet_Access"] == "Yes":
            score += 1
        if inputs["Extracurricular_Activities"] == "Yes":
            score += 1
        if inputs["Learning_Disabilities"] == "Yes":
            score -= 3
        edu_map = {"High School": 0, "College": 1, "Postgraduate": 2}
        score += edu_map.get(inputs["Parental_Education_Level"], 0)
        dist_map = {"Near": 1, "Moderate": 0, "Far": -1}
        score += dist_map.get(inputs["Distance_from_Home"], 0)
        pred = np.clip(score, 55, 99)
    return round(float(pred), 2)
 
 
def get_grade(score):
    if score >= 90:
        return "A+", "#27ae60"
    elif score >= 80:
        return "A", "#2ecc71"
    elif score >= 70:
        return "B", "#3498db"
    elif score >= 60:
        return "C", "#f39c12"
    else:
        return "D", "#e74c3c"
 
 
# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🎓 Student Exam Score Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Enter student details below to predict the expected exam score using ML</div>', unsafe_allow_html=True)
 
if not model_loaded:
    st.warning("⚠️ **model.pkl not found.** Place your trained `model.pkl` in the same directory to enable real predictions. A heuristic fallback is active for demo purposes.")
 
# ─── Layout ─────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([2, 1], gap="large")
 
with left_col:
    # ── Section 1: Academic ──────────────────────────────────────────────────
    st.markdown('<div class="section-title">📚 Academic Performance</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        hours_studied = st.number_input("Hours Studied (per week)", min_value=1, max_value=44, value=20, step=1)
    with col2:
        attendance = st.slider("Attendance (%)", min_value=60, max_value=100, value=80)
    with col3:
        previous_scores = st.slider("Previous Scores", min_value=50, max_value=100, value=75)
 
    col4, col5 = st.columns(2)
    with col4:
        tutoring_sessions = st.number_input("Tutoring Sessions (per month)", min_value=0, max_value=8, value=1, step=1)
    with col5:
        sleep_hours = st.slider("Sleep Hours (per night)", min_value=4, max_value=10, value=7)
 
    # ── Section 2: Personal Factors ──────────────────────────────────────────
    st.markdown('<div class="section-title">🧑 Personal Factors</div>', unsafe_allow_html=True)
    col6, col7, col8 = st.columns(3)
    with col6:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col7:
        motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
    with col8:
        learning_disabilities = st.selectbox("Learning Disabilities", ["No", "Yes"])
 
    col9, col10 = st.columns(2)
    with col9:
        physical_activity = st.slider("Physical Activity (hrs/week)", min_value=0, max_value=6, value=3)
    with col10:
        extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])
 
    # ── Section 3: Home & Family ─────────────────────────────────────────────
    st.markdown('<div class="section-title">🏠 Home & Family</div>', unsafe_allow_html=True)
    col11, col12, col13 = st.columns(3)
    with col11:
        parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
    with col12:
        family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])
    with col13:
        parental_education = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"])
 
    col14, col15 = st.columns(2)
    with col14:
        distance_from_home = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"])
    with col15:
        internet_access = st.selectbox("Internet Access", ["Yes", "No"])
 
    # ── Section 4: School Environment ────────────────────────────────────────
    st.markdown('<div class="section-title">🏫 School Environment</div>', unsafe_allow_html=True)
    col16, col17, col18 = st.columns(3)
    with col16:
        access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
    with col17:
        teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"])
    with col18:
        school_type = st.selectbox("School Type", ["Public", "Private"])
 
    col19, col20 = st.columns(2)
    with col19:
        peer_influence = st.selectbox("Peer Influence", ["Positive", "Neutral", "Negative"])
 
    # ── Predict Button ───────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict Exam Score", use_container_width=True, type="primary")
 
 
# ─── Results Panel ───────────────────────────────────────────────────────────
with right_col:
    st.markdown('<div class="section-title">📊 Prediction Result</div>', unsafe_allow_html=True)
 
    if predict_btn:
        user_inputs = {
            "Hours_Studied": hours_studied,
            "Attendance": attendance,
            "Parental_Involvement": parental_involvement,
            "Access_to_Resources": access_to_resources,
            "Extracurricular_Activities": extracurricular,
            "Sleep_Hours": sleep_hours,
            "Previous_Scores": previous_scores,
            "Motivation_Level": motivation_level,
            "Internet_Access": internet_access,
            "Tutoring_Sessions": tutoring_sessions,
            "Family_Income": family_income,
            "Teacher_Quality": teacher_quality,
            "School_Type": school_type,
            "Peer_Influence": peer_influence,
            "Physical_Activity": physical_activity,
            "Learning_Disabilities": learning_disabilities,
            "Parental_Education_Level": parental_education,
            "Distance_from_Home": distance_from_home,
            "Gender": gender,
        }
 
        with st.spinner("Predicting..."):
            score = predict_score(user_inputs)
 
        grade, color = get_grade(score)
 
        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Predicted Exam Score</div>
            <div class="result-score">{score}</div>
            <div>
                <span class="grade-badge" style="background:{color}; color:white;">
                    Grade: {grade}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
 
        # Key driver summary
        st.markdown('<div class="section-title" style="margin-top:1.5rem;">🔍 Input Summary</div>', unsafe_allow_html=True)
        summary_df = pd.DataFrame({
            "Feature": [
                "Hours Studied", "Attendance", "Previous Scores",
                "Tutoring Sessions", "Sleep Hours", "Motivation",
                "Peer Influence", "Teacher Quality"
            ],
            "Value": [
                f"{hours_studied} hrs", f"{attendance}%", str(previous_scores),
                str(tutoring_sessions), f"{sleep_hours} hrs", motivation_level,
                peer_influence, teacher_quality
            ]
        })
        st.dataframe(summary_df, hide_index=True, use_container_width=True)
 
        # Score interpretation
        interp = {
            "A+": "🌟 Exceptional performance expected!",
            "A":  "✅ Strong performance predicted.",
            "B":  "👍 Good performance predicted.",
            "C":  "📈 Average – room for improvement.",
            "D":  "⚠️ Below average – consider extra support.",
        }
        st.info(interp[grade])
 
    else:
        st.markdown("""
        <div class="info-card">
            <b>How to use:</b><br>
            Fill in all the student details on the left panel and click
            <b>Predict Exam Score</b> to get a predicted score along with
            a grade and performance summary.
            <br><br>
            <b>Features used:</b><br>
            Hours studied, attendance, previous scores, sleep, tutoring,
            parental involvement, school environment, and more.
        </div>
        """, unsafe_allow_html=True)
 
# ─── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#aaa; font-size:0.85rem;'>Student Performance Predictor · Linear Regression + sklearn Pipeline</center>",
    unsafe_allow_html=True,
)