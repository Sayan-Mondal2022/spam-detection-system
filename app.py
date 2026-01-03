import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# -------------------------------------------------
# NLTK setup (run once)
# -------------------------------------------------
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

# -------------------------------------------------
# Text preprocessing function
# -------------------------------------------------
def transform_text(text: str):
    # remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # lowercase & tokenize
    tokens = nltk.word_tokenize(text.lower())

    # remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # stemming
    tokens = [ps.stem(word) for word in tokens]

    return " ".join(tokens)

# -------------------------------------------------
# Load model & vectorizer
# -------------------------------------------------
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.set_page_config(
    page_title="Spam Detection System",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Email / SMS Spam Detection System")
st.write("Enter a message below to check whether it is **Spam** or **Not Spam**.")

# User input
message = st.text_area(
    "Enter your message:",
    height=150,
    placeholder="Type your email or SMS here..."
)

# Predict button
if st.button("Detect Spam"):
    if message.strip() == "":
        st.warning("⚠️ Please enter a message before predicting.")
    else:
        # Preprocess text
        processed_msg = transform_text(message)

        # Vectorize
        vectorized_msg = vectorizer.transform([processed_msg])

        # Predict
        prediction = model.predict(vectorized_msg)[0]

        # Probability (if supported)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(vectorized_msg)[0]
            confidence = max(prob) * 100
        else:
            confidence = None

        # Output
        if prediction == 1:
            st.error("🚨 **SPAM MESSAGE DETECTED**")
        else:
            st.success("✅ **NOT A SPAM MESSAGE**")

        if confidence is not None:
            st.info(f"Confidence: **{confidence:.2f}%**")
