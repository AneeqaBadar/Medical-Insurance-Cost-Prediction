import pickle
import pandas as pd
import streamlit as st

MODEL_FILE = "insurance_cost_model_full_dataset.pkl"
COLUMNS = ["age", "bmi", "children", "sex", "smoker", "region"]

@st.cache_resource
def load_model():
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("Medical Insurance Costs Prediction")
st.caption("Educational model trained on 1,337 records. Not a real insurance quote.")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 64, 30)
    bmi = st.slider("BMI", 15.0, 54.0, 27.0, step=0.1)
    children = st.slider("Number of children", 0, 5, 0)
with col2:
    sex = st.selectbox("Sex", ["female", "male"])
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

if st.button("Predict charges", type="primary"):
    row = pd.DataFrame([[age, bmi, children, sex, smoker, region]], columns=COLUMNS)
    charges = model.predict(row)[0]
    st.success(f"Predicted yearly insurance charges: ${charges:,.2f}")