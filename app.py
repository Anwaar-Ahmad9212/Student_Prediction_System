import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime

st.set_page_config(page_title="Grade Prediction Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

* { font-family: 'Poppins', sans-serif; }

.stApp { background: #ecf7ff; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3a5f 0%, #3b82f6 100%);
    padding-top: 2rem;
}

[data-testid="stSidebar"] * { color: white !important; }

[data-testid="stSidebar"] .stRadio label {
    color: white !important;
    font-weight: 600;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.profile-section { text-align: center; padding: 20px; margin-bottom: 30px; border-bottom: 1px solid rgba(255,255,255,0.2); }
.profile-icon { width: 100px; height: 100px; border-radius: 50%; background: white; margin: 0 auto 15px; display: flex; align-items: center; justify-content: center; font-size: 48px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
.profile-name { font-size: 24px; font-weight: 700; color: white; margin: 10px 0 5px; }
.profile-email { font-size: 12px; color: rgba(255,255,255,0.7); font-weight: 300; }

.stat-card {
    background: white;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border-left: 4px solid #3b82f6;
    transition: transform 0.2s, box-shadow 0.2s;
    height: 100%;
}

.stat-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.12);
}

.stat-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
.stat-label { font-size: 13px; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
.stat-icon { font-size: 24px; opacity: 0.7; }
.stat-value { font-size: 32px; font-weight: 700; color: #1e3a5f; line-height: 1; }

.dashboard-header {
    background: white;
    padding: 24px 32px;
    border-radius: 12px;
    margin-bottom: 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.dashboard-title { font-size: 28px; font-weight: 700; color: #1e3a5f; margin: 0; }
.dashboard-subtitle { font-size: 14px; color: #64748b; margin: 4px 0 0 0; }

.result-card {
    background: linear-gradient(135deg, #1e3a5f 0%, #3b82f6 80%, #22c55e 100%);
    border-radius: 16px;
    padding: 40px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(30, 58, 95, 0.3);
    margin: 20px 0;
}

.result-label { font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 600; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 12px; }
.result-value { font-size: 64px; font-weight: 900; color: white; text-shadow: 0 4px 20px rgba(0,0,0,0.2); line-height: 1; }

.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #22c55e 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 16px 32px;
    font-size: 16px;
    font-weight: 700;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
    transition: all 0.3s ease;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6);
}

/* MERGED SECTION CLASS */
.section {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    margin-bottom: 20px;
    font-size: 18px;
    font-weight: 700;
    color: #1e3a5f;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display:none;}
[data-testid="stSidebar"] .stRadio > div { gap: 8px; }
</style>

""", unsafe_allow_html=True)

# -----------------------------
# 3. MODEL CONFIGURATION
# -----------------------------
MODEL_FEATURES = {
    "RQ1": {"Simple": ["Quiz_Pct"], "Multi": ["Quiz_Pct", "Assign_Pct"], "Dummy": ["Quiz_Pct", "Assign_Pct"]},
    "RQ2": {"Simple": ["Mid1_Pct"], "Multi": ["Quiz_Pct", "Assign_Pct", "Mid1_Pct"], "Dummy": ["Quiz_Pct", "Assign_Pct", "Mid1_Pct"]},
    "RQ3": {"Simple": ["Mid2_Pct"], "Multi": ["Quiz_Pct", "Assign_Pct", "Mid1_Pct", "Mid2_Pct"], "Dummy": ["Quiz_Pct", "Assign_Pct", "Mid1_Pct", "Mid2_Pct"]}
}

MODEL_FILES = {
    "RQ1": {"Simple": "RQ1_Simple_2.pkl", "Multi": "RQ1_Linear_Multi.pkl", "Dummy": "RQ1_Dummy_2.pkl"},
    "RQ2": {"Simple": "RQ2_Simple_2.pkl", "Multi": "RQ2_Linear_Multi.pkl", "Dummy": "RQ2_Dummy_2.pkl"},
    "RQ3": {"Simple": "RQ3_Simple_2.pkl", "Multi": "RQ3_Linear_Multi.pkl", "Dummy": "RQ3_Dummy_2.pkl"}
}

# -----------------------------
# 4. DATA LOADING
# -----------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("universal_student_dataset.csv") 
        return df
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'universal_student_dataset.csv' is available.")
        return pd.DataFrame()

df = load_data()

@st.cache_resource
def load_models():
    models = {}
    for rq, model_map in MODEL_FILES.items():
        models[rq] = {}
        for key, path in model_map.items():
            try:
                models[rq][key] = joblib.load(path)
            except FileNotFoundError:
                st.error(f"Model file not found: {path}. Cannot run application.")
                return None  
    return models

loaded_models = load_models()
if loaded_models is None or df.empty:
    st.stop()

# -----------------------------
# 5. SIDEBAR - NAVIGATION ONLY
# -----------------------------
logo_path = os.path.join(os.getcwd(), "logo.jpeg")



with st.sidebar:
    st.markdown('<div style="text-align:center; margin-bottom:20px;">', unsafe_allow_html=True)
    st.image(logo_path, width=100)
    st.markdown("""
        <div style="font-size:24px; font-weight:700; color:white; margin-top:10px;">STUDENT SCORE</div>
        <div style="font-size:12px; color:rgba(255,255,255,0.7); font-weight:300;">PREDICTION SYSTEM</div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Navigation")
    page = st.radio("Select Page", ["Home", "Prediction", "Analytics", "Data", "Workflow","Contact"], label_visibility="collapsed")

# -----------------------------
# 6. MAIN CONTENT AREA
# -----------------------------
st.markdown(f"""
<div class="dashboard-header">
    <div>
        <div class="dashboard-title">Grade Prediction Dashboard</div>
        <div class="dashboard-subtitle">Regression Model based Academic Forecasting System</div>
    </div>
    <div style="color: #64748b; font-size: 14px;">
        {datetime.now().strftime("%B %d, %Y")}
    </div>
</div>
""", unsafe_allow_html=True)

# ====================================================
# PAGE: DASHBOARD (Overview Only)
# ====================================================
if "Home" in page:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="stat-card"><div class="stat-header"><span class="stat-label">Total Records</span></div><div class="stat-value">{len(df):,}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="stat-card"><div class="stat-header"><span class="stat-label">Active Models</span></div><div class="stat-value">3</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="stat-card"><div class="stat-header"><span class="stat-label">Best Model</span></div><div class="stat-value" style="font-size: 20px;">Multiple Linear Regression</div></div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="section">System Overview</div>', unsafe_allow_html=True)
    
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown("""
        #### Model Information  
        The system provides prediction using three trained regression models:  
        - Simple Linear Regression → uses one key feature  
        - Multiple Linear Regression (Best) → uses all required features  
        - Dummy Baseline → used only for comparison  
        """)
        
        stats_data = {
            "Metric": ["Total Students", "Features Used", "Targets Predicted"],
            "Value": [len(df), "4 (Assignment, Quiz, Mid1, Mid2)", "3 (Mid1, Mid2, Final)"]
        }
        st.dataframe(pd.DataFrame(stats_data), hide_index=True, use_container_width=True)
    
    with col_info2:
        st.markdown("""
        #### Available Predictions  
        """)
        st.markdown("""
        <div style="background:white;border-radius:10px;padding:15px;margin-bottom:10px;box-shadow:0 2px 6px rgba(0,0,0,0.08);">
        <b>RQ1 – Midterm 1 Prediction</b><br>
        Using: Quiz + Assignment  
        Output: Mid 1 Score  
        </div>
        <div style="background:white;border-radius:10px;padding:15px;margin-bottom:10px;box-shadow:0 2px 6px rgba(0,0,0,0.08);">
        <b>RQ2 – Midterm 2 Prediction</b><br>
        Using: Quiz + Assignment + Mid1  
        Output: Mid 2 Score  
        </div>
        <div style="background:white;border-radius:10px;padding:15px;box-shadow:0 2px 6px rgba(0,0,0,0.08);">
        <b>RQ3 – Final Exam Prediction</b><br>
        Using: Quiz + Assignment + Mid1 + Mid2  
        Output: Final Exam Score  
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    # ---------------------------------------------------------
# Model Evaluation & Insights Section for HOME DASHBOARD
# ---------------------------------------------------------
    
    st.write("")
    st.markdown('<div class="section">Model Evaluation Overview</div>', unsafe_allow_html=True)
    
    # High-level highlight cards
    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown('<div class="stat-card"><div class="stat-header"><span class="stat-label">Best Overall Model</span></div><div class="stat-value">Multi Linear (RQ3)</div></div>', unsafe_allow_html=True)
    with colB:
        st.markdown('<div class="stat-card"><div class="stat-header"><span class="stat-label">Highest R² Score</span></div><div class="stat-value">0.67 in RQ3</div></div>', unsafe_allow_html=True)
    with colC:
        st.markdown('<div class="stat-card"><div class="stat-header"><span class="stat-label">Lowest MAE</span></div><div class="stat-value">~9.13 in RQ3</div></div>', unsafe_allow_html=True)
    
    st.write("")
    
    # Load dataframe based on your experimental results
    results_data = [
        ("RQ1", "Multi Linear", 10.994, 0.305, "Stable, Moderate Fit"),
        ("RQ2", "Multi Linear", 12.224, 0.525, "Better than RQ1, Improved Feature Signal"),
        ("RQ3", "Multi Linear", 9.137, 0.669, "Best Model, Strong Predictive Power"),
    ]
    home_df = pd.DataFrame(results_data, columns=["RQ", "Model", "MAE", "R²", "Interpretation"])
    
    st.markdown("#### Summary Performance Table")
    st.dataframe(home_df, use_container_width=True, hide_index=True)
    
    st.write("")
    st.markdown("#### Quick Interpretation Notes")
    st.markdown("""
    - RQ3 model is best and the most reliable for final marks prediction.  
    - Performance improves gradually from RQ1 → RQ3 as more features contribute.  
    - Multi Linear consistently beats Dummy baseline, confirming real learning.  
    - RQ1 is useful for early prediction but accuracy is moderate.  
    - RQ2 strengthens prediction with Mid1 added, lower error than RQ1.  
    """)
    
  
    

# ====================================================
# PAGE: PREDICTION
# ====================================================
elif "Prediction" in page:
    st.markdown('<div class="section">Prediction Settings</div>', unsafe_allow_html=True)
    
    col_settings1, col_settings2 = st.columns(2)
    with col_settings1:
        rq_choice = st.selectbox("Target", ["RQ1: Midterm 1", "RQ2: Midterm 2", "RQ3: Final Exam"], index=2)
    with col_settings2:
        model_type = st.radio("Model Type", ["Simple", "Multi", "Dummy"], horizontal=True)

    if "RQ1" in rq_choice: curr_rq, target_name = "RQ1", "Midterm 1"
    elif "RQ2" in rq_choice: curr_rq, target_name = "RQ2", "Midterm 2"
    else: curr_rq, target_name = "RQ3", "Final Exam"

    curr_model_key = model_type
    curr_model = loaded_models[curr_rq][curr_model_key]
    required_features = MODEL_FEATURES[curr_rq][curr_model_key]

    col_left, col_right = st.columns([2, 3])
    
    with col_left:
        st.markdown('<div class="section">Input Parameters</div>', unsafe_allow_html=True)
        def create_feature_slider(feature_name):
            feature_config = {'Assign_Pct': {'label': 'Assignment Percentage', 'default': 80}, 'Quiz_Pct': {'label': 'Quiz Percentage', 'default': 70}, 'Mid1_Pct': {'label': 'Midterm 1 Percentage', 'default': 75}, 'Mid2_Pct': {'label': 'Midterm 2 Percentage', 'default': 70}}
            config = feature_config.get(feature_name, {'label': feature_name, 'default': 50})
            return st.slider(config['label'], 0, 100, config['default'], key=f"{curr_rq}_{curr_model_key}_{feature_name}")
        
        if curr_model_key == "Dummy":
            input_data = {feature: [0] for feature in required_features}
            input_df = pd.DataFrame(input_data, columns=required_features)
        else:
            all_inputs = {feature: create_feature_slider(feature) for feature in required_features}
            input_data = {feature: [all_inputs[feature]] for feature in required_features}
            input_df = pd.DataFrame(input_data, columns=required_features)
    
    with col_right:
        st.markdown('<div class="section">Prediction Result</div>', unsafe_allow_html=True)
        if st.button("GENERATE PREDICTION", use_container_width=True):
            try:
                prediction = curr_model.predict(input_df)[0]
                prediction = np.clip(prediction, 0, 100)
                st.markdown(f'<div class="result-card"><div class="result-label">Predicted {target_name}</div><div class="result-value">{prediction:.1f}%</div></div>', unsafe_allow_html=True)
                st.success("Prediction completed successfully!")
                if curr_model_key == "Simple":
                    st.info(f"Using: {required_features[0]}")
                elif curr_model_key == "Multi":
                    st.info(f"Using {len(required_features)} features")
                else:
                    st.warning("Baseline prediction")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ====================================================
# PAGE: ANALYTICS
# ====================================================
elif "Analytics" in page:
    st.markdown('<div class="section">Data Analytics</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Correlation Matrix")
        if not df.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            core_features = ['Assign_Pct', 'Quiz_Pct', 'Mid1_Pct', 'Mid2_Pct', 'Final_Pct']
            available = [f for f in core_features if f in df.columns]
            if available:
                sns.heatmap(df[available].corr(), annot=True, cmap="RdYlBu_r", fmt=".2f", ax=ax, square=True)
                st.pyplot(fig)
    with col2:
        st.markdown("#### Grade Distribution")
        target = st.selectbox("Select", ["Mid1_Pct", "Mid2_Pct", "Final_Pct"])
        if target in df.columns:
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            sns.histplot(df[target].dropna(), kde=True, color="#2c5282", ax=ax2, bins=30)
            ax2.set_title(f"{target} Distribution")
            st.pyplot(fig2)

        st.write("")
    st.markdown("#### Feature Relationship Pairplot")
    
    selected_features = st.multiselect(
        "Select Features to Visualize",
        ["Assign_Pct", "Quiz_Pct", "Mid1_Pct", "Mid2_Pct", "Final_Pct"],
        default=["Assign_Pct", "Quiz_Pct", "Mid1_Pct", "Mid2_Pct", "Final_Pct"]
    )
    
    if len(selected_features) >= 2:
        pair_fig = sns.pairplot(df[selected_features], corner=True, diag_kind="kde")
        st.pyplot(pair_fig.fig)
    else:
        st.info("Select at least two features to generate Pairplot.")


# ====================================================
# PAGE: DATA
# ====================================================
elif "Data" in page:
    st.markdown('<div class="section">Dataset Explorer</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Columns", len(df.columns))
    col3.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    st.dataframe(df, use_container_width=True, height=400)
    st.markdown("### Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)



# ====================================================
# PAGE: WORKFLOW
# ====================================================
# ====================================================
# PAGE: WORKFLOW
# ====================================================
if page == "Workflow":
    st.markdown('<div class="section">Workflow / Pipeline Diagram</div>', unsafe_allow_html=True)
    
    steps = [
        "RAW SHEETS (6)", 
        "Cleaning & Standardization", 
        "Combined Universal Dataset", 
        "Research Questions (RQ1, RQ2, RQ3)",
        "Train/Test Split (80/20)",
        "Models (LR, MLR, Dummy)",
        "Evaluation (MAE, RMSE, R², CI)",
        "Streamlit Dashboard"
    ]
    
    step_details = {
        "Cleaning & Standardization": [
            "Renaming Columns",
            "Converting to Percentages",
            "Handling Missing Values"
        ]
    }

    arrow_style = "font-size: 32px; text-align: center; color: #64748b; margin: -10px 0 10px 0;"
    card_style = """
        background: linear-gradient(135deg, #1e3a5f 0%, #3b82f6 80%, #22c55e 100%);
        color: white;
        padding: 24px;
        border-radius: 16px;
        text-align: center;
        font-weight: 700;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        margin-bottom: 12px;
        font-size: 16px;
    """
    
    for step in steps:
        st.markdown(f'<div style="{card_style}">{step}</div>', unsafe_allow_html=True)
        if step in step_details:
            for sub in step_details[step]:
                st.markdown(f'<div style="{card_style}; font-size: 14px; padding:16px;">{sub}</div>', unsafe_allow_html=True)
        if step != steps[-1]:
            st.markdown(f'<div style="{arrow_style}">▼</div>', unsafe_allow_html=True)



elif page == "Contact":
    st.markdown('<div class="section">Contact Information</div>', unsafe_allow_html=True)
    
    # First person
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-header">
                <span class="stat-label">Name</span>
            </div>
            <div class="stat-value" style="font-size: 16px;">Muhammd Anwaar Ahmad</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-header">
                <span class="stat-label">Email</span>
            </div>
            <div class="stat-value" style="font-size: 16px;">f230540@cfd.nu.edu.pk</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("")  # Space between rows

    # Second person
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-header">
                <span class="stat-label">Name</span>
            </div>
            <div class="stat-value" style="font-size: 16px;">Rania Shoaib</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-header">
                <span class="stat-label">Email</span>
            </div>
            <div class="stat-value" style="font-size: 16px;">f230650@cfd.nu.edu.pk</div>
        </div>
        """, unsafe_allow_html=True)
