from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("fraud_detection_model.pkl")

# One-hot encoding mappings
time_of_day_mapping = {'morning': [0, 0, 0], 'afternoon': [1, 0, 0], 'evening': [0, 1, 0], 'night': [0, 0, 1]}
device_mapping = {'mobile': [0, 0], 'desktop': [1, 0], 'tablet': [0, 1]}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from the form
        amount = float(request.form.get("amount"))
        transaction_frequency = float(request.form.get("transaction_frequency"))
        time_of_day = request.form.get("time_of_day")
        device = request.form.get("device")

        # One-hot encode inputs
        time_of_day_features = time_of_day_mapping.get(time_of_day)
        device_features = device_mapping.get(device)

        if time_of_day_features is None or device_features is None:
            return render_template("result.html", error="Invalid inputs for time_of_day or device")

        # Prepare features for prediction
        features = np.array([[amount, transaction_frequency] + time_of_day_features + device_features])

        # Make prediction
        prediction = model.predict(features)[0]
        confidence = model.predict_proba(features)[0]

        # Format results
        result = "Fraudulent" if prediction == 1 else "Non-Fraudulent"
        confidence_scores = {
            "Fraudulent": round(confidence[1] * 100, 2),
            "Non-Fraudulent": round(confidence[0] * 100, 2)
        }

        return render_template("result.html", result=result, confidence=confidence_scores)
    except Exception as e:
        return render_template("result.html", error=str(e))
    
@app.route("/feature-importance", methods=["GET"])
def feature_importance():
    try:
        with open("feature_importance.json", "r") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

if __name__ == '__main__':
    app.run(debug=True)
