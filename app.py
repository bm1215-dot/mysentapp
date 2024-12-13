import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

st.markdown("# LinkedIn User Predictor")

st.markdown("#### Use this app to predict the probability someone is a Linkedin user:")

def data_model():
    s = pd.read_csv("social_media_usage.csv")

    def clean_sm(x):
        return np.where(x == 1, 1, 0)

    ss = pd.DataFrame({
    "sm_li": clean_sm(s["web1h"]),
    "income": np.where(s["income"] > 9, np.nan, s["income"]),
    "education": np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    "parent": np.where(s["par"] == 1, 1, 
               np.where(s["par"] == 2, 0, np.nan)),
    "married": np.where(s["marital"] == 1, 1, 
               np.where(s["marital"].isin([2, 3, 4, 5, 6]), 0, np.nan)),
    "female": np.where(s["gender"] == 2, 1, 
               np.where(s["gender"].isin([1, 3]), 0, np.nan)),
    "age": np.where(s["age"] > 97, np.nan, s["age"])})

    ss = ss.dropna()

    y = ss["sm_li"]
    X = ss[["income", "education", "parent", "married", "female", "age"]]

    X_train, X_test, y_train, y_test = train_test_split(X.values,
                                                    y,
                                                    test_size=0.2, #Chose not to balance classes since we are using a weighted regression
                                                    random_state=7235)

    lr = LogisticRegression(class_weight='balanced')

    lr.fit(X_train, y_train)

    return ss, lr

ss, lr = data_model()

income_labels = {
    1: "Less than $10k",
    2: "10K-20K",
    3: "20K-30K",
    4: "30K-40K",
    5: "40K-50K",
    6: "50K-75K",
    7: "75K-100K",
    8: "100K-150K",
    9: "More than 150K"
}

education_labels = {
    1: "Less than High School",
    2: "High School Incomplete",
    3: "High School",
    4: "Some College",
    5: "Associate’s Degree",
    6: "Bachelor's Degree",
    7: "Some Professional",
    8: "Postgraduate or Professional Degree"
}

age = st.slider("Age", min_value=13, max_value=97, step=1)
income = st.select_slider("Income", options = list(income_labels.keys()), format_func=lambda x: income_labels[x])
education = st.select_slider("Education", options=list(education_labels.keys()), format_func=lambda x: education_labels[x])
parent = st.radio("Parent", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
married = st.radio("Married", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
female = st.radio("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 1 else "Male")

user_data = np.array([[income, education, parent, married, female, age]])

prediction = lr.predict_proba(user_data)[0][1]

linkedin_user = "Yes" if prediction > 0.5 else "No"

st.markdown(f"### LinkedIn User: {linkedin_user}")
st.markdown(f"### Probability of being a LinkedIn User: {prediction * 100:.2f}%")