from flask import Flask, render_template, request
import joblib
import numpy as np
import re
import os

app = Flask(__name__)

# Load the trained phishing detection model
model_path = "model.pkl"
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
else:
    print("Error: model.pkl not found. Train and save the model first.")
    model = None

def extract_features(url):
    """Extract numerical features from a URL for phishing detection."""
    if not url:
        raise ValueError("URL cannot be empty")

    url_length = len(url)
    num_dots = url.count(".")
    num_slashes = url.count("/")
    num_hyphens = url.count("-")
    num_question_marks = url.count("?")
    num_digits = sum(c.isdigit() for c in url)
    num_special_chars = len(re.findall(r"[^a-zA-Z0-9]", url))
    num_subdomains = url.count(".") - 1

    features = np.array([url_length, num_dots, num_slashes, num_hyphens, num_question_marks, num_digits, num_special_chars, num_subdomains])
    return features.reshape(1, -1)  # Reshape for scikit-learn model

@app.route("/")
def home():
    """Render the homepage with the URL input form."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle form submission, predict if URL is phishing or legitimate."""
    url = request.form.get("url", "").strip()
    print(f"Received URL: {url}")

    if not url:
        print("Error: Empty URL")
        return render_template("error.html", error="Please enter a valid URL.")

    if model is None:
        print("Error: Model not loaded")
        return render_template("error.html", error="Model is not loaded. Please train and save the model first.")

    try:
        features = extract_features(url)
        print(f"Extracted Features: {features}")
        prediction = model.predict(features)[0]
        print(f"Prediction: {prediction}")

        result = "Phishing" if prediction == 1 else "Legitimate"
        return render_template("result.html", url=url, result=result)
    
    except Exception as e:
        print(f"Exception: {str(e)}")
        return render_template("error.html", error=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
