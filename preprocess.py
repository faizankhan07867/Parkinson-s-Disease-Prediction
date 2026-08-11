import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler

df = pd.read_csv("dataset/parkinsons.csv")

X = df.drop("status", axis=1)

scaler = StandardScaler()

X = scaler.fit_transform(X)

joblib.dump(scaler, "model/scaler.pkl")

print("Preprocessing Completed Successfully")