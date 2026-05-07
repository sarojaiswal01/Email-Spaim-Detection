import streamlit as st
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📧 Email Spam Detection")

st.write("Enter a message to check if it's Spam or Not")

# Input
msg = st.text_area("Message")

if st.button("Check"):
    if msg.strip() == "":
        st.warning("Please enter a message")
    else:
        transformed = vectorizer.transform([msg])
        result = model.predict(transformed)

        if result[0] == 1:
            st.error("🚫 Spam Message")
        else:
            st.success("✅ Not Spam")