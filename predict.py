import joblib
import pandas as pd

from config import *

# ======================================
# Load Model & Scaler
# ======================================

model = joblib.load(
    MODEL_PATH
)

scaler = joblib.load(
    SCALER_PATH
)

# ======================================
# Feature Names
# ======================================

FEATURES = [
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "D2",
    "PPE"
]

# ======================================
# Prediction Function
# ======================================

def predict_patient():

    print("=" * 60)
    print("Parkinson's Disease Prediction System")
    print("=" * 60)

    values = []

    for feature in FEATURES:

        value = float(
            input(f"{feature} : ")
        )

        values.append(value)

    sample = pd.DataFrame(
        [values],
        columns=FEATURES
    )

    sample = scaler.transform(
        sample
    )

    prediction = model.predict(
        sample
    )[0]

    probability = None

    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(
            sample
        )[0][1]

    print()

    print("=" * 60)

    if prediction == 1:

        print("Prediction : Parkinson's Disease Detected")

    else:

        print("Prediction : Healthy")

    if probability is not None:

        print(f"Probability : {probability:.2%}")

        if probability >= 0.80:

            print("Risk Level : High")

        elif probability >= 0.50:

            print("Risk Level : Moderate")

        else:

            print("Risk Level : Low")

    print("=" * 60)

# ======================================
# Main
# ======================================

if __name__ == "__main__":

    predict_patient()