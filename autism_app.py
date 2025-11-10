import streamlit as st
import pandas as pd
from Autism import EnhancedAutismDetector  # make sure Autism.py is in the same folder

st.set_page_config(page_title="Autism Screening Tool", page_icon="🧩")

# --- Persistent model holder in session state ---
if "detector" not in st.session_state:
    st.session_state.detector = EnhancedAutismDetector()
detector = st.session_state.detector  # <--- THIS defines detector safely

st.title("🧠 Autism Screening Tool chat based")

# --- Optional model training button ---
if st.button("🚀 Initialize / Train Model"):
    with st.spinner("Training model... please wait (may take 20–30 seconds)..."):
        detector.run_complete_analysis(use_real_data=False)
    st.success("✅ Model trained and ready!")


# -------------------------
# Q-CHAT-10 Screening Section
# -------------------------
st.header("📋 Q-CHAT-10 Screening Questions")
st.write("Answer each question based on observed behavior. (0 = Often/Always, 1 = Sometimes/Rarely/Never)")

q_chat_questions = [
    ("A1_Score", "Does your child look at you when you call their name?"),
    ("A2_Score", "How easy is it for you to get eye contact with your child?"),
    ("A3_Score", "Does your child point to indicate that they want something?"),
    ("A4_Score", "Does your child point to share interest with you?"),
    ("A5_Score", "Does your child pretend during play (e.g., talk on a toy phone)?"),
    ("A6_Score", "Does your child follow where you're pointing?"),
    ("A7_Score", "If you turn to look at something, does your child look to see what you are looking at?"),
    ("A8_Score", "Does your child try to attract your attention to their own activity?"),
    ("A9_Score", "Does your child cuddle when you try to cuddle them?"),
    ("A10_Score", "Does your child respond to their name when called?")
]

responses = {}
for code, question_text in q_chat_questions:
    responses[code] = st.selectbox(
        question_text,
        [0, 1],
        format_func=lambda x: "0 = Often/Always" if x == 0 else "1 = Sometimes/Rarely/Never"
    )


# -------------------------
# Demographics Section
# -------------------------
st.header("👤 Demographic Information")

age = st.number_input("Age", min_value=1, max_value=100, value=25)
gender = st.selectbox("Gender", ["m", "f"])
ethnicity = st.selectbox("Ethnicity", ["White-European", "Latino", "Asian", "Black", "Others"])
jaundice = st.selectbox("History of jaundice at birth?", ["yes", "no"])
autism_family_history = st.selectbox("Family history of autism?", ["yes", "no"])
country_of_res = st.selectbox("Country of residence", ["United States", "Brazil", "Spain", "Egypt", "Others"])
used_app_before = st.selectbox("Used autism app before?", ["yes", "no"])
screening_type = st.selectbox("Screening type", [1, 2, 3])
relation = st.selectbox("Who is completing this form?", ["Self", "Parent", "Relative", "Health care professional"])

# -------------------------
# Prepare Input Data
# -------------------------
input_data = {
    **responses,
    "age": age,
    "gender": gender,
    "ethnicity": ethnicity,
    "jaundice": jaundice,
    "autism_family_history": autism_family_history,
    "country_of_res": country_of_res,
    "used_app_before": used_app_before,
    "screening_type": screening_type,
    "relation": relation,
}
input_data["total_score"] = sum(responses.values())

# -------------------------
# Predict Button
# -------------------------
if st.button("🔍 Predict Autism Likelihood"):
    if not hasattr(detector.best_model, "predict"):
        st.error("⚠️ Please train the model first using the '🚀 Initialize / Train Model' button.")
    else:
        with st.spinner("Analyzing responses..."):

            # Convert input_data to a DataFrame
            input_df = pd.DataFrame([input_data])

            # Use the same preprocessing logic as training
            processed_df = detector.preprocess_data(input_df)

            # Reorder columns to match training features
            for feature in detector.feature_names:
                if feature not in processed_df.columns:
                    processed_df[feature] = 0
            processed_df = processed_df[detector.feature_names]

            # Make prediction
            prediction = detector.best_model.predict(processed_df)[0]
            probability = detector.best_model.predict_proba(processed_df)[0, 1]

            # Display results
            st.subheader("🧩 Screening Results")
            st.metric("Predicted Outcome", "ASD Traits Likely" if prediction == 1 else "ASD Traits Unlikely")
            st.metric("Probability", f"{probability * 100:.1f}%")

            if probability > 0.7:
                st.warning("High Risk: Consider seeking a professional evaluation.")
            elif probability > 0.3:
                st.info("Medium Risk: Monitoring or further screening may be useful.")
            else:
                st.success("Low Risk: Screening suggests low likelihood of ASD.")


