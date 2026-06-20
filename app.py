from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    senior = float(request.form["SeniorCitizen"])
    tenure = float(request.form["tenure"])
    monthly = float(request.form["MonthlyCharges"])
    total = float(request.form["TotalCharges"])

    # Create input array
    data = np.array([[senior, tenure, monthly, total]])

    # Add remaining columns as 0
    full_data = np.hstack((data, np.zeros((1, 15))))

    # Scale data
    scaled_data = scaler.transform(full_data)

    # Prediction
    prediction = model.predict(scaled_data)

    if prediction[0] == 1:
        result = "Customer Will Churn"
    else:
        result = "Customer Will Stay"

    return render_template("index.html", prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)
