import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# -------------------------------
# Create Dummy Dataset (Training)
# -------------------------------
data = {
    "10th_marks": [60, 70, 80, 90, 85, 75, 65, 95],
    "12th_marks": [55, 65, 75, 85, 80, 70, 60, 90],
    "entrance_score": [40, 50, 60, 75, 70, 55, 45, 85],
    "category": [0, 1, 1, 2, 2, 1, 0, 2],
    "admission": [0, 0, 1, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[["10th_marks", "12th_marks", "entrance_score", "category"]]
y = df["admission"]

model = LogisticRegression()
model.fit(X, y)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="College Admission Predictor", layout="centered")

st.title("🎓 College Admission Predictor")
st.write("Predict your chances of getting admission")

# Inputs
tenth = st.slider("10th Marks", 0, 100, 75)
twelfth = st.slider("12th Marks", 0, 100, 70)
entrance = st.slider("Entrance Exam Score", 0, 100, 60)

category = st.selectbox(
    "Category",
    ("General", "OBC", "SC/ST")
)

# Encode category
cat_map = {"General": 0, "OBC": 1, "SC/ST": 2}
category_encoded = cat_map[category]

# Prediction
if st.button("Predict Admission"):
    input_data = np.array([[tenth, twelfth, entrance, category_encoded]])
    probability = model.predict_proba(input_data)[0][1]
    result = model.predict(input_data)[0]

    st.subheader("📊 Prediction Result")

    st.write(f"**Admission Probability:** {probability*100:.2f}%")

    if result == 1:
        st.success("✅ High Chance of Admission")
    else:
        st.error("❌ Low Chance of Admission")

st.markdown("---")
st.caption("Mini Project using Streamlit & Machine Learning")
