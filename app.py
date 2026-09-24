import streamlit as st
import joblib
import pandas as pd

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌸",
    layout="wide"
)

# ----------------------------
# Load Model
# ----------------------------
model = joblib.load("iris_model.pkl")

# Class names
classes = ["Setosa", "Versicolor", "Virginica"]

# ----------------------------
# Title
# ----------------------------
st.title("🌸 Iris Flower Species Prediction")
st.write(
    """
This application predicts the **species of an Iris flower**
using a Machine Learning model trained with **Random Forest**.

Adjust the flower measurements from the sidebar and click **Predict**.
"""
)

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("Flower Measurements")

sepal_length = st.sidebar.slider(
    "Sepal Length (cm)",
    min_value=4.0,
    max_value=8.0,
    value=5.8,
    step=0.1
)

sepal_width = st.sidebar.slider(
    "Sepal Width (cm)",
    min_value=2.0,
    max_value=5.0,
    value=3.0,
    step=0.1
)

petal_length = st.sidebar.slider(
    "Petal Length (cm)",
    min_value=1.0,
    max_value=7.0,
    value=4.3,
    step=0.1
)

petal_width = st.sidebar.slider(
    "Petal Width (cm)",
    min_value=0.1,
    max_value=3.0,
    value=1.3,
    step=0.1
)

# ----------------------------
# Input Data
# ----------------------------
input_data = pd.DataFrame({
    "Sepal Length": [sepal_length],
    "Sepal Width": [sepal_width],
    "Petal Length": [petal_length],
    "Petal Width": [petal_width]
})

# ----------------------------
# Layout
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Values")
    st.dataframe(input_data, use_container_width=True)

with col2:
    st.subheader("Prediction")

    if st.button("Predict"):

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        st.success(f"Predicted Species: **{classes[prediction]}**")

        probability_df = pd.DataFrame({
            "Species": classes,
            "Probability": probabilities
        })

        st.subheader("Prediction Probabilities")
        st.bar_chart(
            probability_df.set_index("Species")
        )

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Built with Streamlit • Random Forest Classifier • Scikit-learn")