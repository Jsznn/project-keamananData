import streamlit as st
import joblib
import os
import nltk
import sys
import json

# Add the current directory to sys.path to ensure we can import from src
sys.path.append(os.getcwd())

from src.preprocessing import transform_text

# Download NLTK data (if not present)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load models and vectorizer
@st.cache_resource
def load_assets():
    # Check if models exist
    if not os.path.exists("models/tfidf_vectorizer.pkl"):
        st.error("Models not found. Please run 'python entrypoint/train.py' first.")
        return None, None

    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    models = {
        "Naive Bayes": joblib.load("models/naive_bayes.pkl"),
        "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
        "Linear SVM": joblib.load("models/linear_svc.pkl")
    }
    return tfidf, models

@st.cache_data
def load_metrics():
    if os.path.exists("models/metrics.json"):
        with open("models/metrics.json", "r") as f:
            return json.load(f)
    return None

def main():
    st.set_page_config(page_title="Spam Detection", page_icon="📧")
    
    st.title("📧 Email Spam Detection")
    st.markdown("""
    This application uses Machine Learning to classify emails or SMS messages as **Spam** or **Ham** (Not Spam).
    """)

    tfidf, models = load_assets()

    if tfidf is None or models is None:
        return

    # Sidebar for model selection
    st.sidebar.header("Settings")
    model_name = st.sidebar.selectbox("Select Model", list(models.keys()))

    # Load and display metrics
    metrics = load_metrics()
    if metrics:
        st.sidebar.divider()
        st.sidebar.subheader("Model Performance")
        
        # Find metrics for selected model
        model_metrics = next((m for m in metrics if m["Model"] == model_name), None)
        
        if model_metrics:
            st.sidebar.metric("Accuracy", f"{model_metrics['Accuracy']:.4f}")
            st.sidebar.metric("Precision", f"{model_metrics['Precision']:.4f}")
            st.sidebar.metric("Recall", f"{model_metrics['Recall']:.4f}")
            st.sidebar.metric("F1 Score", f"{model_metrics['F1']:.4f}")
        else:
            st.sidebar.warning("No metrics found for this model.")

    # Input text
    st.subheader("Enter Message")
    default_text = "Congratulations! You've won a $1000 Walmart gift card. Click here to claim your prize now!"
    input_text = st.text_area("Type or paste your message here...", value=default_text, height=150)

    col1, col2 = st.columns([1, 5])
    
    with col1:
        predict_button = st.button("Predict", type="primary")

    if predict_button:
        if input_text.strip() == "":
            st.warning("Please enter a message to analyze.")
        else:
            with st.spinner("Analyzing..."):
                # Preprocess
                transformed_text = transform_text(input_text)
                
                # Vectorize
                vector_input = tfidf.transform([transformed_text])
                
                # Predict
                model = models[model_name]
                prediction = model.predict(vector_input)[0]
                
                # Display result
                st.divider()
                if prediction == 1:
                    st.error("🚨 **SPAM DETECTED!**")
                    st.markdown("This message looks suspicious.")
                else:
                    st.success("✅ **NOT SPAM (HAM)**")
                    st.markdown("This message looks safe.")

                # Details
                with st.expander("See Analysis Details"):
                    st.text(f"Model Used: {model_name}")
                    st.text("Transformed Text (for debugging):")
                    st.code(transformed_text)

if __name__ == "__main__":
    main()
