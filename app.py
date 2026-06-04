import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Student Impact Dashboard",
    page_icon="🎓",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main{
    background-color:#F8FAFC;
}

.metric-card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.1);
}

h1{
    color:#0F172A;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("data/ai_student_impact_dataset.csv")
    return df

df = load_data()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🎓 AI Student Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Academic Performance",
        "AI Usage",
        "Burnout Analysis",
        "Skill Retention",
        "Predictions"
    ]
)

# --------------------------------------------------
# CREATE GPA CHANGE
# --------------------------------------------------

if (
    "Pre_Semester_GPA" in df.columns and
    "Post_Semester_GPA" in df.columns
):
    df["GPA_Change"] = (
        df["Post_Semester_GPA"]
        - df["Pre_Semester_GPA"]
    )

# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.title("📊 AI Impact on Student Performance")

    col1,col2,col3,col4 = st.columns(4)

    with col1:
        st.metric(
            "Students",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "Avg AI Hours",
            round(
                df["Weekly_GenAI_Hours"].mean(),
                2
            )
        )

    with col3:
        st.metric(
            "Skill Retention",
            round(
                df["Skill_Retention_Score"].mean(),
                2
            )
        )

    with col4:
        st.metric(
            "Avg GPA Change",
            round(
                df["GPA_Change"].mean(),
                2
            )
        )

    st.markdown("---")

    c1,c2 = st.columns(2)

    with c1:

        fig = px.histogram(
            df,
            x="Weekly_GenAI_Hours",
            title="Weekly AI Usage"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with c2:

        fig = px.pie(
            df,
            names="Burnout_Risk_Level",
            title="Burnout Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# --------------------------------------------------
# ACADEMIC PERFORMANCE
# --------------------------------------------------

elif page == "Academic Performance":

    st.title("📚 Academic Performance")

    fig = px.scatter(
        df,
        x="Weekly_GenAI_Hours",
        y="Post_Semester_GPA",
        color="Burnout_Risk_Level",
        title="AI Usage vs GPA"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    if "Major" in df.columns:

        major_gpa = (
            df.groupby("Major")["Post_Semester_GPA"]
            .mean()
            .reset_index()
            .sort_values(
                by="Post_Semester_GPA",
                ascending=False
            )
        )

        fig = px.bar(
            major_gpa,
            x="Major",
            y="Post_Semester_GPA",
            title="Average GPA by Major"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# --------------------------------------------------
# AI USAGE
# --------------------------------------------------

elif page == "AI Usage":

    st.title("🤖 AI Usage Analysis")

    col1,col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            df,
            x="Weekly_GenAI_Hours",
            nbins=30,
            title="Distribution of AI Usage"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.box(
            df,
            x="Burnout_Risk_Level",
            y="Weekly_GenAI_Hours",
            title="AI Hours by Burnout"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    if "Primary_Use_Case" in df.columns:

        fig = px.pie(
            df,
            names="Primary_Use_Case",
            title="Primary AI Use Cases"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# --------------------------------------------------
# BURNOUT ANALYSIS
# --------------------------------------------------

elif page == "Burnout Analysis":

    st.title("🔥 Burnout Analysis")

    burnout_stats = (
        df.groupby("Burnout_Risk_Level")
        ["Weekly_GenAI_Hours"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        burnout_stats,
        x="Burnout_Risk_Level",
        y="Weekly_GenAI_Hours",
        title="Average AI Hours by Burnout"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    fig = px.scatter(
        df,
        x="Weekly_GenAI_Hours",
        y="Skill_Retention_Score",
        color="Burnout_Risk_Level",
        title="Burnout vs Skill Retention"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# --------------------------------------------------
# SKILL RETENTION
# --------------------------------------------------

elif page == "Skill Retention":

    st.title("🧠 Skill Retention")

    fig = px.scatter(
        df,
        x="Weekly_GenAI_Hours",
        y="Skill_Retention_Score",
        color="Prompt_Engineering_Skill",
        title="AI Usage vs Skill Retention"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    prompt_stats = (
        df.groupby("Prompt_Engineering_Skill")
        ["Skill_Retention_Score"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        prompt_stats,
        x="Prompt_Engineering_Skill",
        y="Skill_Retention_Score",
        title="Skill Retention by Prompt Skill"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# --------------------------------------------------
# PREDICTION PAGE
# --------------------------------------------------

elif page == "Predictions":

    st.title("🔮 Burnout Prediction")

    model_df = df.copy()

    features = [
        "Weekly_GenAI_Hours",
        "Skill_Retention_Score"
    ]

    target = "Burnout_Risk_Level"

    encoder = LabelEncoder()

    model_df[target] = encoder.fit_transform(
        model_df[target]
    )

    X = model_df[features]
    y = model_df[target]

    X_train,X_test,y_train,y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train,y_train)

    st.subheader("Enter Student Details")

    ai_hours = st.slider(
        "Weekly AI Hours",
        0.0,
        30.0,
        8.0
    )

    skill = st.slider(
        "Skill Retention Score",
        0,
        100,
        70
    )

    if st.button("Predict Burnout Risk"):

        prediction = model.predict(
            [[ai_hours,skill]]
        )[0]

        result = encoder.inverse_transform(
            [prediction]
        )[0]

        st.success(
            f"Predicted Burnout Risk: {result}"
        )

# --------------------------------------------------
# RAW DATA
# --------------------------------------------------

st.sidebar.markdown("---")

if st.sidebar.checkbox("Show Dataset"):

    st.dataframe(df)
