import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Page Config ---
st.set_page_config(page_title="CVD Risk Dashboard", layout="wide")
st.title("🏥 Cardiovascular Disease Risk Dashboard")
st.markdown("*Analyzing when cardiovascular disease becomes noticeably more common*")

# --- 1. Data Loading & Preprocessing ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("cardio_train.csv", delimiter=";")
    except FileNotFoundError:
        st.warning("'cardio_train.csv' not found. Using dummy dataset...")
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
    
    # Feature transformation logic
    df["age_years"] = df["age"] / 365.25
    df["CVD Status"] = df["cardio"].map({0: "Healthy", 1: "CVD Present"})
    df["Gender Label"] = df["gender"].map({1: "Women", 2: "Men"})
    df["Activity Level"] = df["active"].map({0: "Sedentary", 1: "Active"})
    
    # Categorical mapping
    df["Cholesterol Level"] = df["cholesterol"].map({1: "Normal", 2: "Above Normal", 3: "High"})
    df["Glucose Level"] = df["gluc"].map({1: "Normal", 2: "Above Normal", 3: "High"})
    df["BMI"] = df["weight"] / ((df["height"] / 100) ** 2)
    
    # Outlier filter
    df_clean = df[
        (df["ap_hi"] > 80)
        & (df["ap_hi"] < 220)
        & (df["weight"] > 40)
        & (df["BMI"] < 50)
    ].copy()
    
    return df_clean

df_raw = load_data()

# --- SIDEBAR INTERACTIVE FILTERS ---
st.sidebar.header("🎛️ Patient Subpopulation Filters")
st.sidebar.markdown("Filter the whole dataset to analyze custom risk profiles:")

# Lifestyle Segment Filters
activity_filter = st.sidebar.multiselect(
    "Physical Activity Status",
    options=["Active", "Sedentary"],
    default=["Active", "Sedentary"]
)

chol_filter = st.sidebar.multiselect(
    "Cholesterol Baseline",
    options=["Normal", "Above Normal", "High"],
    default=["Normal", "Above Normal", "High"]
)

# Advanced Clinical Range Filters
bmi_range = st.sidebar.slider(
    "Body Mass Index (BMI) Range",
    min_value=int(df_raw["BMI"].min()),
    max_value=int(df_raw["BMI"].max()),
    value=(int(df_raw["BMI"].min()), int(df_raw["BMI"].max()))
)

# Dynamically apply filters to the active dataframe
df_clean = df_raw[
    (df_raw["Activity Level"].isin(activity_filter)) &
    (df_raw["Cholesterol Level"].isin(chol_filter)) &
    (df_raw["BMI"].between(bmi_range[0], bmi_range[1]))
].copy()


# --- Helper function to find age at risk threshold ---
def find_age_at_risk(data, gender, risk_threshold):
    """Find the age where CVD prevalence first exceeds the risk threshold"""
    gender_data = data[data["Gender Label"] == gender].copy()
    gender_data["age_bin"] = pd.cut(gender_data["age_years"], bins=np.arange(35, 69, 1), right=False)
    
    risk_by_age = (
        gender_data.groupby("age_bin", observed=False)["cardio"]
        .agg(['mean', 'count'])
        .reset_index()
    )
    risk_by_age["risk_pct"] = risk_by_age["mean"] * 100
    risk_by_age = risk_by_age[risk_by_age["count"] >= 5]  # Lower sample size bound for filtered sets
    
    exceeds = risk_by_age[risk_by_age["risk_pct"] >= risk_threshold]
    if len(exceeds) > 0:
        return int(exceeds.iloc[0]["age_bin"].left)
    return None

# --- SECTION 2: INTERACTIVE THRESHOLD EXPLORER ---
st.markdown("## 🎚️ Explore Risk Thresholds")
risk_threshold = st.slider(
    "Select Target CVD Prevalence Cross-Section (%)",
    min_value=10,
    max_value=90,
    value=50,
    step=5
)

# --- SECTION 1: DYNAMIC KEY INSIGHTS ---
st.markdown("## 🎯 Cohort Analysis Insights")

col1, col2, col3, col4 = st.columns(4)

with col1:
    women_pct = find_age_at_risk(df_clean, "Women", risk_threshold)
    st.metric(f"Women hit {risk_threshold}% risk", f"~{women_pct} yrs" if women_pct else "N/A")

with col2:
    men_pct = find_age_at_risk(df_clean, "Men", risk_threshold)
    st.metric(f"Men hit {risk_threshold}% risk", f"~{men_pct} yrs" if men_pct else "N/A")

with col3:
    if len(df_clean) > 0:
        overall_cvd = (df_clean["cardio"].sum() / len(df_clean) * 100)
        st.metric("Cohort CVD Rate", f"{overall_cvd:.1f}%")
    else:
        st.metric("Cohort CVD Rate", "0.0%")

with col4:
    age_diff = (women_pct - men_pct) if (women_pct and men_pct) else None
    st.metric("Gender Age Gap", f"{abs(age_diff)} years" if age_diff else "N/A", 
              delta="Women older" if (age_diff and age_diff >= 0) else "Women younger" if age_diff else None)

if len(df_clean) == 0:
    st.error("⚠️ No patients match the current sidebar filter criteria. Please broaden your selection.")
    st.stop()

st.markdown("---")

# --- SECTION 3: MAIN VISUALIZATIONS ---
st.markdown("## 📊 Cohort Breakdown Visualizations")

# --- Plot 1: The Tipping Point (ENHANCED) ---
st.subheader("1️⃣ CVD Risk Over Time by Gender")

df_clean["age_bin"] = pd.cut(df_clean["age_years"], bins=np.arange(35, 69, 2), right=False)
bin_data = (
    df_clean.groupby(["age_bin", "Gender Label"], observed=False)["cardio"]
    .mean()
    .reset_index()
)
bin_data["CVD Prevalence (%)"] = bin_data["cardio"] * 100
bin_data["Age Group (Years)"] = bin_data["age_bin"].astype(str)

fig1 = px.line(
    bin_data,
    x="Age Group (Years)",
    y="CVD Prevalence (%)",
    color="Gender Label",
    markers=True,
    title=f"Cardiovascular Risk Cross-Section (Filtered Cohort)",
    color_discrete_sequence=["#e34a33", "#2b8cbe"],
)
fig1.add_hline(y=risk_threshold, line_dash="dash", annotation_text=f"{risk_threshold}% Target", line_color="orange")
fig1.update_yaxes(range=[0, 100])
fig1.update_layout(hovermode="x unified")
st.plotly_chart(fig1, use_container_width=True)

# --- Plot 2: Multivariate Risk Matrix ---
st.subheader("2️⃣ Risk Matrix: Cholesterol & Glucose Impact")
df_clean["age_5yr"] = pd.cut(df_clean["age_years"], bins=np.arange(35, 70, 5), right=False)
prob_grid = (
    df_clean.groupby(["age_5yr", "Cholesterol Level", "Glucose Level"], observed=False)["cardio"]
    .mean()
    .reset_index()
)
prob_grid["CVD Probability"] = prob_grid["cardio"]
prob_grid["Age Bracket"] = prob_grid["age_5yr"].astype(str)

fig2 = px.line(
    prob_grid,
    x="Age Bracket",
    y="CVD Probability",
    color="Cholesterol Level",
    facet_col="Glucose Level",
    title="How Cholesterol & Glucose Levels Compound Risk within Cohort",
    color_discrete_sequence=px.colors.qualitative.Set1,
    category_orders={"Cholesterol Level": ["Normal", "Above Normal", "High"], "Glucose Level": ["Normal", "Above Normal", "High"]}
)
fig2.update_yaxes(range=[0, 1])
st.plotly_chart(fig2, use_container_width=True)

# --- Plot 3: Blood Pressure & Cholesterol Heatmap ---
st.subheader("3️⃣ Blood Pressure × Cholesterol: Combined Risk")

def categorize_bp(bp):
    if bp < 120:
        return "Normal (<120)"
    elif 120 <= bp < 130:
        return "Elevated (120-129)"
    elif 130 <= bp < 140:
        return "Stage 1 (130-139)"
    else:
        return "Stage 2 (140+)"

df_clean["BP Bracket"] = df_clean["ap_hi"].apply(categorize_bp)

heatmap_df = (
    df_clean.groupby(["Cholesterol Level", "BP Bracket"], observed=False)["cardio"]
    .mean()
    .reset_index()
)
heatmap_df["CVD Prevalence (%)"] = round(heatmap_df["cardio"] * 100, 1)

bp_order = ["Normal (<120)", "Elevated (120-129)", "Stage 1 (130-139)", "Stage 2 (140+)"]
chol_order = ["Normal", "Above Normal", "High"]

fig3 = px.density_heatmap(
    heatmap_df,
    x="BP Bracket",
    y="Cholesterol Level",
    z="CVD Prevalence (%)",
    histfunc="sum",
    text_auto=True,
    color_continuous_scale="Reds",
    title="CVD Prevalence by BP & Cholesterol Matrix",
    category_orders={"BP Bracket": bp_order, "Cholesterol Level": chol_order}
)
st.plotly_chart(fig3, use_container_width=True)

# --- Plot 4: Lifestyle Impact ---
st.subheader("4️⃣ Distribution of Age Across CVD Outcomes")
fig4 = px.box(
    df_clean,
    x="CVD Status",
    y="age_years",
    color="Activity Level",
    facet_col="Gender Label",
    notched=True,
    title="Age Distribution for Healthy vs CVD Cohorts",
    color_discrete_sequence=["#7fcdbb", "#2c7fb8"],
    labels={"age_years": "Patient Age (Years)"},
    category_orders={"CVD Status": ["Healthy", "CVD Present"], "Activity Level": ["Sedentary", "Active"]}
)
st.plotly_chart(fig4, use_container_width=True)

# --- Sidebar Context Metadata ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Active Subgroup Stats")
st.sidebar.metric("Patients in View", f"{len(df_clean):,}")
st.sidebar.metric("CVD Cases in View", f"{(df_clean['cardio'] == 1).sum():,}")
st.sidebar.markdown(f"**Original Total Pool:** {len(df_raw):,} patients")