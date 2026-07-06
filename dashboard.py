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

df_clean = load_data()

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
    risk_by_age = risk_by_age[risk_by_age["count"] > 10]  # Filter for meaningful sample sizes
    
    exceeds = risk_by_age[risk_by_age["risk_pct"] >= risk_threshold]
    if len(exceeds) > 0:
        return int(exceeds.iloc[0]["age_bin"].left)
    return None

# --- SECTION 1: KEY INSIGHTS ---
st.markdown("---")
st.markdown("## 🎯 Key Findings")

col1, col2, col3, col4 = st.columns(4)

with col1:
    women_50pct = find_age_at_risk(df_clean, "Women", 50)
    st.metric("Women hit 50% risk", f"~{women_50pct} years" if women_50pct else "N/A")

with col2:
    men_50pct = find_age_at_risk(df_clean, "Men", 50)
    st.metric("Men hit 50% risk", f"~{men_50pct} years" if men_50pct else "N/A")

with col3:
    overall_cvd = (df_clean["cardio"].sum() / len(df_clean) * 100)
    st.metric("Overall CVD Rate", f"{overall_cvd:.1f}%")

with col4:
    age_diff = (women_50pct - men_50pct) if (women_50pct and men_50pct) else None
    st.metric("Gender Gap", f"{abs(age_diff)} years" if age_diff else "N/A", delta=f"Women {'' if age_diff >= 0 else 'older'}")

st.markdown("""
**📌 Summary:** Cardiovascular disease prevalence becomes noticeably common (50% threshold) at:
- **Women**: ~{} years old
- **Men**: ~{} years old

Men tend to develop CVD at slightly earlier ages than women.
""".format(women_50pct or "?", men_50pct or "?"))

st.markdown("---")

# --- SECTION 2: INTERACTIVE THRESHOLD EXPLORER ---
st.markdown("## 🎚️ Explore Risk Thresholds")
st.markdown("*Use the slider below to see at what age different CVD risk levels are reached*")

risk_threshold = st.slider(
    "Select CVD Prevalence Threshold (%)",
    min_value=10,
    max_value=90,
    value=50,
    step=5
)

col1, col2 = st.columns(2)

with col1:
    women_age = find_age_at_risk(df_clean, "Women", risk_threshold)
    if women_age:
        st.success(f"**Women:** CVD reaches {risk_threshold}% prevalence at age **~{women_age}**")
    else:
        st.warning(f"**Women:** Threshold of {risk_threshold}% not reached in dataset")

with col2:
    men_age = find_age_at_risk(df_clean, "Men", risk_threshold)
    if men_age:
        st.success(f"**Men:** CVD reaches {risk_threshold}% prevalence at age **~{men_age}**")
    else:
        st.warning(f"**Men:** Threshold of {risk_threshold}% not reached in dataset")

st.markdown("---")

# --- SECTION 3: MAIN VISUALIZATIONS ---
st.markdown("## 📊 Detailed Visualizations")

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
    title="Cardiovascular Risk Over Time by Gender",
    color_discrete_sequence=["#e34a33", "#2b8cbe"],
)
fig1.add_hline(y=50, line_dash="dash", annotation_text="50% Threshold", line_color="orange")
fig1.add_hline(y=risk_threshold, line_dash="dot", annotation_text=f"{risk_threshold}% Selected", line_color="green")
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
    title="How Cholesterol & Glucose Levels Compound Risk",
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
    title="CVD Prevalence by BP & Cholesterol (Higher = More Red)",
    category_orders={"BP Bracket": bp_order, "Cholesterol Level": chol_order}
)
st.plotly_chart(fig3, use_container_width=True)

# --- Plot 4: Lifestyle Impact ---
st.subheader("4️⃣ The Lifestyle Buffer: Physical Activity Impact")
fig4 = px.box(
    df_clean,
    x="CVD Status",
    y="age_years",
    color="Activity Level",
    facet_col="Gender Label",
    notched=True,
    title="Does Physical Activity Lower CVD Risk?",
    color_discrete_sequence=["#7fcdbb", "#2c7fb8"],
    labels={"age_years": "Patient Age (Years)"},
    category_orders={"CVD Status": ["Healthy", "CVD Present"], "Activity Level": ["Sedentary", "Active"]}
)
st.plotly_chart(fig4, use_container_width=True)

# --- SECTION 4: INSIGHTS & INTERPRETATION ---
st.markdown("---")
st.markdown("## 💡 What Does This Mean?")

st.info("""
**Key Takeaway:** Cardiovascular disease prevalence shows a clear age-related increase, with a notable "tipping point" around 55-60 years old.

- **Gender Differences:** Men tend to develop CVD slightly earlier than women
- **Risk Factors Stack:** High cholesterol + high blood pressure = dramatically higher risk
- **Lifestyle Matters:** Active individuals show better outcomes in younger groups
- **Screening:** These findings support cardiovascular screening programs starting in the 45-50 age range
""")

# --- Sidebar ---
st.sidebar.markdown("### 📊 Dataset Overview")
st.sidebar.metric("Total Patients", f"{len(df_clean):,}")
st.sidebar.metric("CVD Cases", f"{(df_clean['cardio'] == 1).sum():,}")
st.sidebar.metric("CVD Rate", f"{(df_clean['cardio'].mean() * 100):.1f}%")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**About This Dashboard:**
- Shows CVD prevalence patterns by age and gender
- Helps identify high-risk age groups
- Explores how lifestyle & health factors influence risk
""")