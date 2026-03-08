import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os

from src.utils import load_object
from src.pipeline.train_pipeline import TrainPipeline

# Page settings
st.set_page_config(
    page_title="AI Student Performance Dashboard",
    page_icon="🎓",
    layout="wide"
)

# Dark style
st.markdown("""
<style>
body{
    background-color:#0E1117;
}
.big-font {
    font-size:22px !important;
}
</style>
""", unsafe_allow_html=True)

# Train model if missing
if not os.path.exists("artifacts/model.pkl"):
    pipeline = TrainPipeline()
    pipeline.run_pipeline()

model = load_object("artifacts/model.pkl")

# Header
st.title("🎓 AI Student Performance Dashboard")
st.write("Machine Learning powered system to predict student exam results")

st.markdown("---")

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📚 Academic Inputs")

    study_hours = st.slider("Study Hours Per Day",0,12,4)
    attendance = st.slider("Attendance %",0,100,80)
    previous_score = st.slider("Previous Exam Score",0,100,70)

with col2:
    st.subheader("💡 Lifestyle Inputs")

    sleep_hours = st.slider("Sleep Hours",0,10,7)
    activities = st.selectbox("Extracurricular Activities",["No","Yes"])

    activities = 1 if activities=="Yes" else 0

st.markdown("---")

# Prediction
if st.button("🚀 Predict Performance"):

    data = np.array([[study_hours,attendance,previous_score,sleep_hours,activities]])

    prediction = model.predict(data)[0]

    score = round(prediction,2)

    st.success(f"Predicted Final Score: {score}")

    st.markdown("## 📊 Performance Meter")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text':"Student Performance"},
        gauge={
            'axis':{'range':[0,100]},
            'bar':{'color':"lightblue"},
            'steps':[
                {'range':[0,50],'color':"red"},
                {'range':[50,70],'color':"orange"},
                {'range':[70,85],'color':"yellow"},
                {'range':[85,100],'color':"green"}
            ]
        }
    ))

    st.plotly_chart(gauge,use_container_width=True)

    st.markdown("---")

    # Student Report Card
    st.markdown("## 👤 Student AI Report Card")

    report = pd.DataFrame({
        "Metric":[
            "Study Hours",
            "Attendance",
            "Previous Score",
            "Sleep Hours",
            "Activities",
            "Predicted Score"
        ],
        "Value":[
            study_hours,
            attendance,
            previous_score,
            sleep_hours,
            activities,
            score
        ]
    })

    st.table(report)

    st.markdown("---")

    # Feature Importance
    st.markdown("## 📈 Feature Importance")

    importance = model.feature_importances_

    features = [
        "Study Hours",
        "Attendance",
        "Previous Score",
        "Sleep Hours",
        "Activities"
    ]

    importance_df = pd.DataFrame({
        "Feature":features,
        "Importance":importance
    })

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Model Feature Importance"
    )

    st.plotly_chart(fig,use_container_width=True)

st.markdown("---")
st.caption("Built with ❤️ using Machine Learning + Streamlit")