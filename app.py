import os
import joblib
import numpy as np
import pandas as pd
import logging
from flask import Flask, request, jsonify, render_template

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Flask app
app = Flask(__name__)

# Load scaler and encoder 
scaler = joblib.load("files/scaler.pkl")
encoder = joblib.load("files/encoder.pkl")

# Load all models dynamically from "models" directory
MODEL_DIR = "models"
models = {}
for filename in os.listdir(MODEL_DIR):
    if filename.endswith(".pkl"):
        model_name = filename.split(".pkl")[0]  # Remove extension for key
        models[model_name] = joblib.load(os.path.join(MODEL_DIR, filename))

# Ensure at least one model is loaded
if not models:
    raise ValueError("No models found in the 'models' directory!")

logging.info(f"Loaded models: {list(models.keys())}")


# Function to process input data
def process_input(data):
    try:
        df = pd.DataFrame([data])

        # Extract numerical value from 'size' (e.g., "3 BHK" → 3)
        df["size"] = df["size"].astype(str).str.extract(r"(\d+)").astype(float)

        # Convert 'total_sqft' (handling ranges like "1200-1500")
        def convert_total_sqft(value):
            try:
                if "-" in value:
                    low, high = value.split("-")
                    return (float(low) + float(high)) / 2
                return float(value)
            except ValueError:
                return np.nan  # Handle invalid cases

        df["total_sqft"] = df["total_sqft"].astype(str).apply(convert_total_sqft)

        # One-hot encode 'area_type'
        encoded_features = encoder.transform(df[["area_type"]])
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(["area_type"]))

        # Drop original categorical column and merge encoded features
        df = df.drop(columns=["area_type"])
        df = pd.concat([df, encoded_df], axis=1)

        # Scale numerical features
        numerical_features = ["size", "bath", "balcony", "total_sqft"]
        df[numerical_features] = scaler.transform(df[numerical_features])

        return df
    except Exception as e:
        logging.error(f"Error processing input: {e}")
        return None


# API Endpoint to Predict Price
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        model_name = data.pop("model", list(models.keys())[0])  # Remove model key, default to first model
        
        if model_name not in models:
            return jsonify({"error": "Invalid model name"}), 400

        processed_data = process_input(data)
        if processed_data is None:
            return jsonify({"error": "Invalid input format"}), 400

        # Convert processed data to a NumPy array before passing to model
        model = models[model_name]
        prediction = model.predict(processed_data.values)[0]  # Ensure it's passed as an array

        return jsonify({"predicted_price": round(float(prediction), 2)})
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": "Internal server error"}), 500


# Home Page (HTML Form)
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Collect form data
            user_input = {
                "area_type": request.form["area_type"],
                "size": request.form["size"],
                "bath": float(request.form["bath"]),
                "balcony": float(request.form["balcony"]),
                "total_sqft": request.form["total_sqft"],
            }
            model_name = request.form["model"]

            # Process input and predict
            processed_data = process_input(user_input)
            if processed_data is None:
                return render_template("index.html", error="Invalid input")

            prediction = models[model_name].predict(processed_data)[0]
            return render_template("index.html", predicted_price=round(float(prediction), 2), models=models.keys())

        except Exception as e:
            logging.error(f"Form submission error: {e}")
            return render_template("index.html", error="Something went wrong!")

    return render_template("index.html", models=models.keys())


if __name__ == "__main__":
    app.run(debug=True)
