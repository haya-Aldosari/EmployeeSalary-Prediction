import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("salary_model.pkl")

# Load dataset for available categories
df = pd.read_csv("Employers_data.csv")

st.set_page_config(page_title="Salary Prediction", layout="centered")

st.title("Employee Salary Prediction")
st.write("Enter employee details to predict the salary.")

# Employee information form
with st.form("salary_form"):

    age = st.number_input("Age", min_value=18, max_value=70, value=25)

    experience = st.number_input("Experience Years", min_value=0, value=2, step=1)

    department = st.selectbox("Department", sorted(df["Department"].dropna().unique()))

    job_title = st.selectbox("Job Title", sorted(df["Job_Title"].dropna().unique()))

    education = st.selectbox(
        "Education Level", sorted(df["Education_Level"].dropna().unique())
    )

    location = st.selectbox("Location", sorted(df["Location"].dropna().unique()))

    submit = st.form_submit_button("Predict Salary")

# Generate prediction
if submit:

    new_employee = pd.DataFrame(
        [
            {
                "Age": age,
                "Experience_Years": experience,
                "Department": department,
                "Job_Title": job_title,
                "Education_Level": education,
                "Location": location,
            }
        ]
    )

    predicted_salary = model.predict(new_employee)[0]

    st.success(f"Predicted Salary: {predicted_salary:,.2f}")
