import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Page Config ---
st.set_page_config(page_title="CVD Risk Drivers Dashboard", layout="wide")

# Custom CSS for tighter spacing and visual polish
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 32px; font-weight: 600; }
    [data-testid="stMetric"] { padding: 0.5rem 0; }
    .stTabs [data-baseweb="tab-list"] button { min-width: 110px; font-size: 14px; }
    .block-container { padding-top: 1rem; padding-bottom: 0.5rem; }
    h2 { margin-bottom: 0.3rem; }
    h3 { margin-top: 0; margin-bottom: 0.5rem; }
    .risk-gauge { text-align: center; }
    .risk-low { color: #2ecc71; }
    .risk-medium { color: #f39c12; }
    .risk-high { color: #e74c3c; }
</style>
""", unsafe_allow_html=True)

st.title("🏥 What Drives Cardiovascular Disease Risk?")
st.markdown("*Explore which factors matter most and predict your risk profile*")

# --- 1. Data Loading & Preprocessing ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("cardio_train.csv", delimiter=";")
    except FileNotFoundError:
        st.info("Using sample dataset. Download 'cardio_train.csv' from Kaggle for full analysis.")
        np.random.seed(42)
        n = 5000
        df = pd.DataFrame({
            "age": np.random.randint(12000, 24000, n),
            "gender": np.random.choice([1, 2], n),
            "height": np.random.randint(150, 190, n),
            "weight": np.random.randint(50, 110, n),
            "ap_hi": np.random.randint(90, 180, n),
            "cholesterol": np.random.choice([1, 2, 3], n),
            "gluc": np.random.choice([1, 2, 3], n),
            "active": np.random.choice([0, 1], n),
            "cardio": np.random.choice([0, 1], n)
        })
    
    # Feature transformation
    df["age_years"] = df["age"] / 365.25
    df["Gender Label"] = df["gender"].map({1: "Female", 2: "Male"})
    df["Activity Level"] = df["active"].map({0: "Sedentary", 1: "Active"})
    df["Cholesterol Level"] = df["cholesterol"].map({1: "Normal", 2: "Above Normal", 3: "High"})
    df["Glucose Level"] = df["gluc"].map({1: "Normal", 2: "Above Normal", 3: "High"})
    df["BMI"] = df["weight"] / ((df["height"] / 100) ** 2)
    
    # Systolic BP category
    def categorize_bp(bp):
        if bp < 120:
            return "Normal"
        elif 120 <= bp < 130:
            return "Elevated"
        elif 130 <= bp < 140:
            return "Stage 1 HTN"
        else:
            return "Stage 2 HTN"
    
    df["BP Category"] = df["ap_hi"].apply(categorize_bp)
    
    # Outlier filter
    df_clean = df[
        (df["ap_hi"] > 80)
        & (df["ap_hi"] < 220)
        & (df["weight"] > 40)
        & (df["BMI"] < 50)
    ].copy()
    
    return df_clean

df = load_data()

# --- Calculate Factor Importance Metrics ---
@st.cache_data
def calculate_factor_importance(data):
    """Calculate correlation and CVD prevalence differences for each factor"""
    
    # Numeric conversions for correlation
    data_numeric = data.copy()
    data_numeric["gender_numeric"] = data_numeric["Gender Label"].map({"Female": 0, "Male": 1})
    data_numeric["activity_numeric"] = data_numeric["Activity Level"].map({"Sedentary": 0, "Active": 1})
    data_numeric["cholesterol_numeric"] = data_numeric["Cholesterol Level"].map(
        {"Normal": 1, "Above Normal": 2, "High": 3}
    )
    data_numeric["glucose_numeric"] = data_numeric["Glucose Level"].map(
        {"Normal": 1, "Above Normal": 2, "High": 3}
    )
    
    factors = {
        "Age": data_numeric["age_years"],
        "Systolic BP": data_numeric["ap_hi"],
        "Cholesterol": data_numeric["cholesterol_numeric"],
        "Glucose": data_numeric["glucose_numeric"],
        "BMI": data_numeric["BMI"],
        "Male Gender": data_numeric["gender_numeric"],
        "Physical Activity": data_numeric["activity_numeric"]
    }
    
    importance = {}
    for factor_name, factor_vals in factors.items():
        corr = abs(factor_vals.corr(data_numeric["cardio"]))
        importance[factor_name] = corr
    
    return pd.DataFrame(list(importance.items()), columns=["Factor", "Importance"]).sort_values("Importance", ascending=True)

factor_importance = calculate_factor_importance(df)

# --- SIDEBAR ---
st.sidebar.markdown("### 📋 Filter Population")

gender_filter = st.sidebar.multiselect(
    "Gender",
    options=["Female", "Male"],
    default=["Female", "Male"]
)

activity_filter = st.sidebar.multiselect(
    "Activity",
    options=["Active", "Sedentary"],
    default=["Active", "Sedentary"]
)

age_range = st.sidebar.slider(
    "Age Range",
    min_value=int(df["age_years"].min()),
    max_value=int(df["age_years"].max()),
    value=(int(df["age_years"].min()), int(df["age_years"].max())),
    step=1
)

# Apply filters
df_filtered = df[
    (df["Gender Label"].isin(gender_filter)) &
    (df["Activity Level"].isin(activity_filter)) &
    (df["age_years"] >= age_range[0]) &
    (df["age_years"] <= age_range[1])
].copy()

if len(df_filtered) == 0:
    st.error("⚠️ No patients match filters. Broaden selection.")
    st.stop()

# --- Helper: Estimate CVD risk based on user profile ---
def estimate_risk(age, gender, bp, cholesterol, glucose, activity, bmi):
    """Simple Bayesian risk estimator based on population"""
    
    # Filter to similar population segment
    similar = df[
        (df["age_years"] >= age - 2) & (df["age_years"] <= age + 2)
    ]
    
    if len(similar) == 0:
        return 0.5
    
    # Apply factor weights
    mask = similar["cardio"] == 1
    base_rate = mask.sum() / len(similar)
    
    # Adjust for factors (simplified)
    adjustments = []
    
    # Age factor
    if age > 55:
        adjustments.append(0.15)
    
    # BP factor
    if bp >= 140:
        adjustments.append(0.20)
    elif bp >= 130:
        adjustments.append(0.10)
    
    # Cholesterol factor
    if cholesterol == "High":
        adjustments.append(0.15)
    elif cholesterol == "Above Normal":
        adjustments.append(0.05)
    
    # Glucose factor
    if glucose == "High":
        adjustments.append(0.12)
    elif glucose == "Above Normal":
        adjustments.append(0.04)
    
    # Activity (protective)
    if activity == "Active":
        adjustments.append(-0.08)
    
    # BMI factor
    if bmi >= 30:
        adjustments.append(0.10)
    elif bmi >= 25:
        adjustments.append(0.05)
    
    # Gender (males higher risk)
    if gender == "Male":
        adjustments.append(0.08)
    
    risk = base_rate + sum(adjustments)
    return np.clip(risk, 0, 0.95)

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Risk Calculator", "📊 Factor Importance", "🔗 Interactions", "👥 Risk Profiles"])

# ============= TAB 1: INTERACTIVE RISK CALCULATOR =============
with tab1:
    st.markdown("### Your CVD Risk Profile")
    st.markdown("*Adjust your characteristics to see how your risk changes*")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Demographics")
        calc_age = st.slider("Age (years)", 35, 70, 50, key="calc_age")
        calc_gender = st.radio("Gender", ["Female", "Male"], key="calc_gender", horizontal=True)
    
    with col2:
        st.markdown("#### Vitals")
        calc_bp = st.slider("Systolic BP (mmHg)", 80, 200, 120, key="calc_bp")
        calc_bmi = st.slider("BMI", 15.0, 50.0, 25.0, step=0.1, key="calc_bmi")
    
    with col3:
        st.markdown("#### Labs & Lifestyle")
        calc_chol = st.selectbox("Cholesterol", ["Normal", "Above Normal", "High"], key="calc_chol")
        calc_gluc = st.selectbox("Glucose", ["Normal", "Above Normal", "High"], key="calc_gluc")
        calc_activity = st.radio("Activity", ["Sedentary", "Active"], key="calc_activity", horizontal=True)
    
    # Calculate risk
    estimated_risk = estimate_risk(calc_age, calc_gender, calc_bp, calc_chol, calc_gluc, calc_activity, calc_bmi)
    
    # Display risk prominently
    st.markdown("---")
    risk_pct = estimated_risk * 100
    
    col_risk1, col_risk2, col_risk3 = st.columns([1, 2, 1])
    
    with col_risk2:
        if risk_pct < 20:
            risk_color = "🟢 LOW"
        elif risk_pct < 40:
            risk_color = "🟡 MODERATE"
        else:
            risk_color = "🔴 HIGH"
        
        st.markdown(f"## {risk_color}")
        st.metric("Estimated CVD Risk", f"{risk_pct:.1f}%")
    
    # Show which factors are pushing risk up/down
    st.markdown("---")
    st.markdown("### Factor Contribution to Your Risk")
    
    factor_contrib = []
    
    if calc_age > 55:
        factor_contrib.append(("Age (55+)", 0.15, "↑"))
    if calc_bp >= 140:
        factor_contrib.append(("Systolic BP (≥140)", 0.20, "↑"))
    elif calc_bp >= 130:
        factor_contrib.append(("Systolic BP (130-139)", 0.10, "↑"))
    if calc_chol == "High":
        factor_contrib.append(("High Cholesterol", 0.15, "↑"))
    elif calc_chol == "Above Normal":
        factor_contrib.append(("Above Normal Cholesterol", 0.05, "↑"))
    if calc_gluc == "High":
        factor_contrib.append(("High Glucose", 0.12, "↑"))
    elif calc_gluc == "Above Normal":
        factor_contrib.append(("Above Normal Glucose", 0.04, "↑"))
    if calc_bmi >= 30:
        factor_contrib.append(("High BMI (≥30)", 0.10, "↑"))
    elif calc_bmi >= 25:
        factor_contrib.append(("Overweight (BMI 25-29)", 0.05, "↑"))
    if calc_activity == "Active":
        factor_contrib.append(("Regular Activity", -0.08, "↓"))
    if calc_gender == "Male":
        factor_contrib.append(("Male Gender", 0.08, "↑"))
    
    if factor_contrib:
        contrib_df = pd.DataFrame(factor_contrib, columns=["Factor", "Impact", "Direction"])
        contrib_df = contrib_df.sort_values("Impact", ascending=True)
        
        fig_contrib = px.bar(
            contrib_df,
            y="Factor",
            x="Impact",
            color="Direction",
            color_discrete_map={"↑": "#e74c3c", "↓": "#2ecc71"},
            title="Which Factors Drive Your Risk?",
            labels={"Impact": "Risk Contribution", "Factor": ""},
            height=350
        )
        fig_contrib.update_layout(margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
        st.plotly_chart(fig_contrib, use_container_width=True)
    else:
        st.info("✅ No major risk factors detected in your profile!")

# ============= TAB 2: FACTOR IMPORTANCE =============
with tab2:
    st.markdown("### Which Factors Matter Most for CVD Risk?")
    
    col_imp1, col_imp2 = st.columns([1, 1.2])
    
    with col_imp1:
        # Factor importance ranking
        fig_importance = px.barh(
            factor_importance,
            x="Importance",
            y="Factor",
            color="Importance",
            color_continuous_scale="RdYlGn_r",
            title="Factor Importance Ranking",
            height=350,
            labels={"Importance": "Correlation with CVD"}
        )
        fig_importance.update_layout(margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
        st.plotly_chart(fig_importance, use_container_width=True)
    
    with col_imp2:
        st.markdown("### CVD Prevalence by Factor Level")
        
        # Create factor breakdown tabs
        factor_to_show = st.radio(
            "Select factor to explore:",
            options=["Age", "Systolic BP", "Cholesterol", "Glucose", "BMI", "Activity"],
            key="factor_explorer"
        )
        
        if factor_to_show == "Age":
            df_filtered["age_group"] = pd.cut(df_filtered["age_years"], bins=[35, 45, 55, 65, 75], 
                                               labels=["35-45", "45-55", "55-65", "65+"])
            factor_breakdown = df_filtered.groupby("age_group", observed=False)["cardio"].agg(['mean', 'count']).reset_index()
            factor_breakdown.columns = ["Level", "CVD Rate", "N"]
            x_col = "Level"
        
        elif factor_to_show == "Systolic BP":
            factor_breakdown = df_filtered.groupby("BP Category", observed=False)["cardio"].agg(['mean', 'count']).reset_index()
            factor_breakdown.columns = ["Level", "CVD Rate", "N"]
            x_col = "Level"
        
        elif factor_to_show == "Cholesterol":
            factor_breakdown = df_filtered.groupby("Cholesterol Level", observed=False)["cardio"].agg(['mean', 'count']).reset_index()
            factor_breakdown.columns = ["Level", "CVD Rate", "N"]
            x_col = "Level"
        
        elif factor_to_show == "Glucose":
            factor_breakdown = df_filtered.groupby("Glucose Level", observed=False)["cardio"].agg(['mean', 'count']).reset_index()
            factor_breakdown.columns = ["Level", "CVD Rate", "N"]
            x_col = "Level"
        
        elif factor_to_show == "BMI":
            df_filtered["BMI Category"] = pd.cut(df_filtered["BMI"], 
                                                  bins=[0, 18.5, 25, 30, 100],
                                                  labels=["Underweight", "Normal", "Overweight", "Obese"])
            factor_breakdown = df_filtered.groupby("BMI Category", observed=False)["cardio"].agg(['mean', 'count']).reset_index()
            factor_breakdown.columns = ["Level", "CVD Rate", "N"]
            x_col = "Level"
        
        else:  # Activity
            factor_breakdown = df_filtered.groupby("Activity Level", observed=False)["cardio"].agg(['mean', 'count']).reset_index()
            factor_breakdown.columns = ["Level", "CVD Rate", "N"]
            x_col = "Level"
        
        factor_breakdown["CVD Rate %"] = factor_breakdown["CVD Rate"] * 100
        
        fig_factor = px.bar(
            factor_breakdown,
            x=x_col,
            y="CVD Rate %",
            color="CVD Rate %",
            color_continuous_scale="RdYlGn_r",
            title=f"CVD Risk by {factor_to_show}",
            height=350,
            text="N",
            labels={x_col: factor_to_show, "CVD Rate %": "CVD Prevalence (%)"}
        )
        fig_factor.update_layout(margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
        fig_factor.update_traces(textposition="outside")
        st.plotly_chart(fig_factor, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Distribution Comparison: Healthy vs CVD")
    
    col_dist1, col_dist2 = st.columns(2)
    
    with col_dist1:
        # Age distribution
        fig_age_dist = px.violin(
            df_filtered,
            x="Gender Label",
            y="age_years",
            color="cardio",
            color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
            category_orders={"cardio": [0, 1]},
            labels={"cardio": "CVD Status", "0": "Healthy", "1": "CVD Present"},
            title="Age Distribution by CVD Status",
            height=350
        )
        fig_age_dist.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_age_dist, use_container_width=True)
    
    with col_dist2:
        # BP distribution
        fig_bp_dist = px.box(
            df_filtered,
            x="cardio",
            y="ap_hi",
            color="Gender Label",
            points="outliers",
            category_orders={"cardio": [0, 1]},
            labels={"cardio": "CVD Status"},
            title="Systolic BP Distribution by CVD Status",
            height=350,
            color_discrete_map={"Female": "#e34a33", "Male": "#2b8cbe"}
        )
        fig_bp_dist.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_bp_dist, use_container_width=True)

# ============= TAB 3: FACTOR INTERACTIONS =============
with tab3:
    st.markdown("### How Do Risk Factors Combine?")
    st.markdown("*Explore interactions between the biggest risk drivers*")
    
    col_int1, col_int2 = st.columns(2)
    
    with col_int1:
        # BP × Cholesterol interaction
        df_filtered["age_group"] = pd.cut(df_filtered["age_years"], bins=[35, 45, 55, 65, 75])
        
        bp_chol_cvd = df_filtered.groupby(["BP Category", "Cholesterol Level"], observed=False)["cardio"].agg(['mean', 'count']).reset_index()
        bp_chol_cvd["CVD Rate %"] = bp_chol_cvd["mean"] * 100
        
        bp_order = ["Normal", "Elevated", "Stage 1 HTN", "Stage 2 HTN"]
        chol_order = ["Normal", "Above Normal", "High"]
        
        fig_int1 = px.density_heatmap(
            bp_chol_cvd,
            x="BP Category",
            y="Cholesterol Level",
            z="CVD Rate %",
            nbinsx=4,
            nbinsy=3,
            color_continuous_scale="RdYlGn_r",
            text_auto=True,
            title="BP × Cholesterol Risk Matrix",
            height=350,
            category_orders={"BP Category": bp_order, "Cholesterol Level": chol_order}
        )
        fig_int1.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_int1, use_container_width=True)
    
    with col_int2:
        # Age × Activity interaction
        df_filtered["age_group_5"] = pd.cut(df_filtered["age_years"], bins=np.arange(35, 75, 5))
        
        age_activity_cvd = df_filtered.groupby(["age_group_5", "Activity Level"], observed=False)["cardio"].mean().reset_index()
        age_activity_cvd["CVD Rate %"] = age_activity_cvd["cardio"] * 100
        age_activity_cvd["Age Range"] = age_activity_cvd["age_group_5"].astype(str)
        
        fig_int2 = px.line(
            age_activity_cvd,
            x="Age Range",
            y="CVD Rate %",
            color="Activity Level",
            markers=True,
            title="Age × Activity Level Risk",
            color_discrete_map={"Active": "#2ecc71", "Sedentary": "#e74c3c"},
            height=350,
            labels={"Age Range": "Age Group"}
        )
        fig_int2.update_yaxes(range=[0, 100])
        fig_int2.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_int2, use_container_width=True)
    
    st.markdown("---")
    
    col_int3, col_int4 = st.columns(2)
    
    with col_int3:
        # Glucose × BMI
        df_filtered["BMI Category"] = pd.cut(df_filtered["BMI"], 
                                              bins=[0, 18.5, 25, 30, 100],
                                              labels=["Underweight", "Normal", "Overweight", "Obese"])
        
        glucose_bmi_cvd = df_filtered.groupby(["Glucose Level", "BMI Category"], observed=False)["cardio"].agg(['mean', 'count']).reset_index()
        glucose_bmi_cvd["CVD Rate %"] = glucose_bmi_cvd["mean"] * 100
        glucose_bmi_cvd = glucose_bmi_cvd[glucose_bmi_cvd["count"] >= 5]
        
        fig_int3 = px.scatter(
            glucose_bmi_cvd,
            x="BMI Category",
            y="Glucose Level",
            size="count",
            color="CVD Rate %",
            color_continuous_scale="RdYlGn_r",
            title="Glucose × BMI Risk Profile",
            height=350,
            labels={"count": "Sample Size"}
        )
        fig_int3.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_int3, use_container_width=True)
    
    with col_int4:
        # Gender × Age interaction
        gender_age_cvd = df_filtered.groupby(["Gender Label", df_filtered["age_years"]//5*5], observed=False)["cardio"].agg(['mean', 'count']).reset_index()
        gender_age_cvd.columns = ["Gender Label", "Age Decade", "CVD Rate", "Count"]
        gender_age_cvd["CVD Rate %"] = gender_age_cvd["CVD Rate"] * 100
        gender_age_cvd = gender_age_cvd[gender_age_cvd["Count"] >= 10]
        
        fig_int4 = px.line(
            gender_age_cvd,
            x="Age Decade",
            y="CVD Rate %",
            color="Gender Label",
            markers=True,
            title="Gender × Age Divergence",
            color_discrete_map={"Female": "#e34a33", "Male": "#2b8cbe"},
            height=350
        )
        fig_int4.update_yaxes(range=[0, 100])
        fig_int4.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_int4, use_container_width=True)

# ============= TAB 4: RISK PROFILES / SEGMENTS =============
with tab4:
    st.markdown("### CVD Risk Profiles in Your Population")
    st.markdown("*Identify distinct patient segments by their risk factor combinations*")
    
    # Define risk profiles based on factor combinations
    df_filtered["Risk Profile"] = "Unknown"
    
    # High risk: High BP + High Cholesterol
    high_risk_mask = (
        (df_filtered["ap_hi"] >= 140) & 
        (df_filtered["Cholesterol Level"] == "High")
    )
    df_filtered.loc[high_risk_mask, "Risk Profile"] = "🔴 High BP + High Chol"
    
    # Very high risk: multiple factors
    very_high_mask = (
        (df_filtered["ap_hi"] >= 140) & 
        ((df_filtered["Cholesterol Level"] == "High") | (df_filtered["Glucose Level"] == "High"))
    )
    df_filtered.loc[very_high_mask, "Risk Profile"] = "🔴🔴 Multi-Factor High"
    
    # Metabolic syndrome: High BP + High Glucose + High BMI
    metabolic_mask = (
        (df_filtered["ap_hi"] >= 130) &
        (df_filtered["Glucose Level"] == "High") &
        (df_filtered["BMI"] >= 30)
    )
    df_filtered.loc[metabolic_mask, "Risk Profile"] = "🟡 Metabolic Syndrome"
    
    # Sedentary + overweight
    sedentary_mask = (
        (df_filtered["Activity Level"] == "Sedentary") &
        (df_filtered["BMI"] >= 25)
    )
    df_filtered.loc[sedentary_mask & (df_filtered["Risk Profile"] == "Unknown"), "Risk Profile"] = "🟡 Sedentary + Overweight"
    
    # Age-related risk (older, but controlled)
    age_risk_mask = (
        (df_filtered["age_years"] >= 60) &
        (df_filtered["ap_hi"] < 140) &
        (df_filtered["Cholesterol Level"] != "High")
    )
    df_filtered.loc[age_risk_mask & (df_filtered["Risk Profile"] == "Unknown"), "Risk Profile"] = "🟡 Age-Related"
    
    # Healthy profile
    df_filtered.loc[df_filtered["Risk Profile"] == "Unknown", "Risk Profile"] = "🟢 Low Risk"
    
    # Count profiles
    profile_counts = df_filtered["Risk Profile"].value_counts()
    
    col_prof1, col_prof2 = st.columns([1, 1.2])
    
    with col_prof1:
        fig_profiles = px.pie(
            values=profile_counts.values,
            names=profile_counts.index,
            title="Population Distribution by Risk Profile",
            height=350,
            color_discrete_sequence=["#2ecc71", "#f39c12", "#e74c3c", "#c0392b", "#95a5a6"]
        )
        fig_profiles.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_profiles, use_container_width=True)
    
    with col_prof2:
        # CVD rate by profile
        profile_cvd = df_filtered.groupby("Risk Profile")["cardio"].agg(['mean', 'count']).reset_index()
        profile_cvd["CVD Rate %"] = profile_cvd["mean"] * 100
        profile_cvd = profile_cvd.sort_values("CVD Rate %", ascending=True)
        
        fig_profile_cvd = px.barh(
            profile_cvd,
            x="CVD Rate %",
            y="Risk Profile",
            color="CVD Rate %",
            color_continuous_scale="RdYlGn_r",
            title="Actual CVD Rate by Profile",
            height=350,
            text="count",
            labels={"count": "N"}
        )
        fig_profile_cvd.update_layout(margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
        fig_profile_cvd.update_traces(textposition="outside")
        st.plotly_chart(fig_profile_cvd, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Profile Characteristics")
    
    selected_profile = st.selectbox(
        "Select a profile to explore",
        options=df_filtered["Risk Profile"].unique(),
        key="profile_explorer"
    )
    
    profile_data = df_filtered[df_filtered["Risk Profile"] == selected_profile]
    
    col_prof_stats1, col_prof_stats2, col_prof_stats3, col_prof_stats4 = st.columns(4)
    
    with col_prof_stats1:
        st.metric("Patients", len(profile_data))
    
    with col_prof_stats2:
        cvd_rate = (profile_data["cardio"].sum() / len(profile_data) * 100)
        st.metric("CVD Rate", f"{cvd_rate:.1f}%")
    
    with col_prof_stats3:
        avg_age = profile_data["age_years"].mean()
        st.metric("Avg Age", f"{avg_age:.1f} yrs")
    
    with col_prof_stats4:
        avg_bp = profile_data["ap_hi"].mean()
        st.metric("Avg Systolic BP", f"{avg_bp:.0f} mmHg")
    
    # Factor breakdown for selected profile
    col_prof_char1, col_prof_char2 = st.columns(2)
    
    with col_prof_char1:
        profile_gender = profile_data["Gender Label"].value_counts(normalize=True) * 100
        fig_gender = px.pie(
            values=profile_gender.values,
            names=profile_gender.index,
            title="Gender Distribution",
            height=300,
            color_discrete_map={"Female": "#e34a33", "Male": "#2b8cbe"}
        )
        fig_gender.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_gender, use_container_width=True)
    
    with col_prof_char2:
        profile_activity = profile_data["Activity Level"].value_counts(normalize=True) * 100
        fig_activity = px.pie(
            values=profile_activity.values,
            names=profile_activity.index,
            title="Activity Distribution",
            height=300,
            color_discrete_map={"Active": "#2ecc71", "Sedentary": "#e74c3c"}
        )
        fig_activity.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_activity, use_container_width=True)

# --- FOOTER ---
st.markdown("---")
st.sidebar.markdown("### 📊 Dataset Summary")
st.sidebar.metric("Total Patients", f"{len(df):,}")
st.sidebar.metric("CVD Cases", f"{(df['cardio'] == 1).sum():,}")
st.sidebar.metric("Overall Prevalence", f"{(df['cardio'].sum() / len(df) * 100):.1f}%")
st.sidebar.markdown(f"*Filtered view: {len(df_filtered):,} patients*")