import streamlit as st
import pandas as pd
import datetime
import joblib

# Load model
import os
if not os.path.exists("ensemble_topic_classifier.pkl"):
    st.error("Model file not found. Please upload it.")
else:
    model = joblib.load("ensemble_topic_classifier.pkl")

st.set_page_config(page_title="Participedia Topic Classifier", page_icon="🚀")
st.title("Participedia Topic Classifier")

# Topic mapping
topic_id_to_name = {
    0: "Government & Civic Engagement",
    1: "Budgeting & Financial Planning",
    2: "Social & Community Issues",
    3: "Environmental Policies",
    4: "Healthcare & Public Services",
    5: "Education & Student Engagement",
    6: "Social Justice & Community Development",
    7: "Technology & Digital Governance",
    8: "Transportation & Urban Planning",
    9: "Elections & Democratic Processes",
    10: "Labor Rights & Economic Policies",
    11: "Housing & Infrastructure",
    12: "Public Safety & Law Enforcement",
    13: "Arts, Culture & Media",
    14: "Participatory & Deliberative Democracy",
    15: "International Relations & Geopolitics",
    16: "Public Health & Pandemic Policy",
    17: "Rural Development & Agriculture",
    18: "Energy & Renewable Resources"
}

# Prediction function
def predict_label(text, longitude, latitude, budget,
                  pp_methods, target_audience, completeness,
                  geog_scope, purpose, category, facilitation):

    df_input = pd.DataFrame([{
        "CombinedText": text,
        "TopicProbability_HDBSCAN": 0.0,  # ← best constant value!
        "Longitude": longitude,
        "Latitude": latitude,
        "Budget(USD)": budget,
        "PPMethods": pp_methods,
        "TargetedAudience": target_audience,
        "Completeness": completeness,
        "GeogScope": geog_scope,
        "Purpose": purpose,
        "Category": category,
        "FacilitationType": facilitation
    }])

    pred_index = model.predict(df_input)[0]
    return topic_id_to_name.get(pred_index, "Unknown")

# Save feedback
def save_feedback(text, predicted, actual):
    feedback_file = "user_feedback.csv"
    entry = pd.DataFrame([{
        "timestamp": datetime.datetime.now(),
        "text": text,
        "predicted_label": predicted,
        "actual_label": actual
    }])
    try:
        existing = pd.read_csv(feedback_file)
        updated = pd.concat([existing, entry], ignore_index=True)
    except FileNotFoundError:
        updated = entry
    updated.to_csv(feedback_file, index=False)

# Streamlit UI
st.title(" Topic Classifier with Human Feedback")
st.markdown("Enter participation text and attributes below:")

# Initialize session state
if "predicted_label" not in st.session_state:
    st.session_state.predicted_label = None
if "text_input" not in st.session_state:
    st.session_state.text_input = ""

# Input fields
text_input = st.text_area("Combined Text")
longitude = st.number_input("Longitude", value=0.0)
latitude = st.number_input("Latitude", value=0.0)
budget = st.number_input("Budget (USD)", value=10000)

pp_methods = st.selectbox("PP Methods", ['Participatory Budgeting', 'Surveys', 'Protest', 'Online Consultations', 'Public Hearing', 'Petition', 'Other', 'Unknown'])
target_audience = st.selectbox("Targeted Audience", ['General Public', 'Elected Officials', 'Appointed Public Servants', 'Organized Groups', 'Media', 'Other', 'Unknown'])
completeness = st.selectbox("Completeness", [1, 2, 3])
geog_scope = st.selectbox("Geographic Scope", ['Local', 'Regional', 'National', 'International'])
purpose = st.selectbox("Purpose", [
    'Consulting', 
    'Public Decision', 
    'Develop Individual Capacity', 
    'Raise Public Awareness', 
    'Community Building', 
    'Protest', 
    'Co-Governance', 
    'Other', 
    'None'
])
category = st.selectbox("Category", [
    'Social Issues', 
    'Governance & Law',
    'Economic & Financial Issues',
    'Community & Environment',
    'Education & Knowledge', 
    'Arts, Culture & Media', 
    'Miscellaneous / Other', 
    'None'
])
facilitation = st.selectbox("Facilitation Type", ['Professional', 'Peer', 'No Facilitation', 'Other', 'Unknown'])

# Store text input
st.session_state.text_input = text_input

# Prediction
if st.button("Predict"):
    prediction = predict_label(
        text_input, longitude, latitude, budget,
        pp_methods, target_audience, completeness,
        geog_scope, purpose, category, facilitation
    )
    st.session_state.predicted_label = prediction
    st.success(f"###  Predicted Topic: **{prediction}**")

# Feedback section
if st.session_state.predicted_label:
    feedback = st.radio("Is this prediction correct?", ["Yes", "No"], key="feedback_radio")

    if feedback == "Yes":
        save_feedback(st.session_state.text_input, st.session_state.predicted_label, st.session_state.predicted_label)
        st.success(" Feedback saved. Thanks!")
    else:
        correct_label = st.selectbox("What is the correct topic?", list(topic_id_to_name.values()), key="correction_label")
        if st.button("Submit Correction"):
            save_feedback(st.session_state.text_input, st.session_state.predicted_label, correct_label)
            st.success(" Correction submitted. Thank you!")
