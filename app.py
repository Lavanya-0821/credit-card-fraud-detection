# ============================================================
# CREDIT CARD FRAUD DETECTION
# FLASK BACKEND
# ============================================================

from flask import Flask, render_template, request, jsonify

import pandas as pd
import joblib


# ============================================================
# 1. CREATE FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# 2. LOAD TRAINED MODEL AND SCALER
# ============================================================

model = joblib.load("fraud_model.pkl")

scaler = joblib.load("scaler.pkl")

print("Model loaded successfully!")
print("Scaler loaded successfully!")


# ============================================================
# 3. LOAD DATASET
# ============================================================

df = pd.read_csv("creditcard.csv")

# All columns except Class are input features
feature_columns = [
    column for column in df.columns
    if column != "Class"
]


# ============================================================
# 4. HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        features=feature_columns
    )


# ============================================================
# 5. SAMPLE TRANSACTION
# ============================================================

@app.route("/sample", methods=["GET"])
def sample():

    # Select one transaction
    sample_row = df.sample(
        n=1,
        random_state=42
    )

    transaction = {}

    for column in feature_columns:

        transaction[column] = float(
            sample_row.iloc[0][column]
        )

    return jsonify(transaction)


# ============================================================
# 6. FRAUD PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get data from frontend
        data = request.get_json()

        values = []

        # Read all required features
        for column in feature_columns:

            if column not in data:

                return jsonify({
                    "success": False,
                    "error": f"Missing feature: {column}"
                })

            values.append(
                float(data[column])
            )


        # Convert transaction into DataFrame
        transaction = pd.DataFrame(
            [values],
            columns=feature_columns
        )


        # Scale the transaction
        transaction_scaled = scaler.transform(
            transaction
        )


        # Predict
        prediction = model.predict(
            transaction_scaled
        )[0]


        # Get fraud probability
        probability = model.predict_proba(
            transaction_scaled
        )[0][1]


        probability_percent = probability * 100


        # ====================================================
        # CHECK RESULT
        # ====================================================

        if prediction == 1:

            result = "FRAUD"

            message = (
                "This transaction appears to be fraudulent."
            )

        else:

            result = "NORMAL"

            message = (
                "This transaction appears to be normal."
            )


        # Send result to frontend
        return jsonify({

            "success": True,

            "prediction": int(prediction),

            "result": result,

            "message": message,

            "probability": round(
                probability_percent,
                2
            )

        })


    except Exception as e:

        return jsonify({

            "success": False,

            "error": str(e)

        })


# ============================================================
# 7. RUN FLASK SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )