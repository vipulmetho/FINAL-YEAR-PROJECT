import streamlit as st
import pickle
import re

# load the saved model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# same cleaning function used during training
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# page setup
st.set_page_config(page_title="Spam Classifier", page_icon="📧")
st.title("📧 Spam Email Classifier")
st.write("This app uses Machine Learning (Logistic Regression + TF-IDF) to detect spam messages.")

# input area
msg = st.text_area("Enter the email or SMS message below:", height=150)

if st.button("Check Message"):
    if msg.strip() == "":
        st.warning("Please enter a message first.")
    else:
        cleaned = clean_text(msg)
        vec = vectorizer.transform([cleaned])
        result = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]

        if result == 1:
            st.error("🚨 This message looks like SPAM!")
            st.write(f"Confidence: {prob[1]*100:.1f}%")
        else:
            st.success("✅ This message looks SAFE (Not Spam)")
            st.write(f"Confidence: {prob[0]*100:.1f}%")

# small footer
st.markdown("---")
st.caption("BCA Final Year Project | Spam Email Classification using NLP and ML")
