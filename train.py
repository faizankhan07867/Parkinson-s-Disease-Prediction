import joblib
import numpy as np

from config import *

from utils.dataset import ParkinsonDataset
from utils.preprocessing import DataPreprocessor
from utils.logger import *
from utils.metrics import *
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier

# ======================================
# Load Dataset
# ======================================

dataset = ParkinsonDataset()

df = dataset.prepare()

processor = DataPreprocessor()

X_train, X_test, y_train, y_test, feature_names = processor.process(df)

log_info(

    "Dataset Preprocessing Completed"

)

# ======================================
# Models
# ======================================

models = {

    "Logistic Regression":

    LogisticRegression(

        max_iter=1000,

        random_state=RANDOM_STATE

    ),

    "Support Vector Machine":

    SVC(

        probability=True,

        random_state=RANDOM_STATE

    ),

    "Random Forest":

    RandomForestClassifier(

        n_estimators=200,

        random_state=RANDOM_STATE

    ),

    "XGBoost":

    XGBClassifier(

        random_state=RANDOM_STATE,

        eval_metric="logloss"

    )

}

scores = {}

best_model = None

best_score = -1

best_name = ""

# ======================================
# Train Models
# ======================================

for model_name, model in models.items():

    print("=" * 60)

    print(

        f"Training {model_name}"

    )

    print("=" * 60)

    model.fit(

        X_train,

        y_train

    )

    predictions = model.predict(

        X_test

    )

    probabilities = None

    if hasattr(

        model,

        "predict_proba"

    ):

        probabilities = model.predict_proba(

            X_test

        )[:,1]

    metrics = classification_metrics(

        y_test,

        predictions,

        probabilities

    )

    accuracy = metrics["Accuracy"]

    scores[model_name] = accuracy

    log_model(

        model_name,

        accuracy

    )
    
    cv_score = cross_val_score(

        model,

        X_train,

        y_train,

        cv=CV_FOLDS,

        scoring="accuracy"

    ).mean()

    print(

        f"Cross Validation Accuracy : {cv_score:.4f}"

    )

    if accuracy > best_score:

        best_score = accuracy

        best_model = model

        best_name = model_name

        print(

            "✅ Best Model Updated"

        )
        
# ======================================
# Save Best Model
# ======================================

joblib.dump(

    best_model,

    MODEL_PATH

)

log_best(

    best_name

)

print()

print("=" * 60)

print(

    "Best Model :",

    best_name

)

print(

    "Best Accuracy :",

    round(best_score,4)

)

print("=" * 60)

# ======================================
# Evaluation Graphs
# ======================================

print("\nGenerating Evaluation Graphs...")

predictions = best_model.predict(
    
    X_text
)  
  
report = classification_report(
    y_test,
    predictions
)   



save_confusion_matrix(

    y_test,

    predictions

)

save_roc_curve(

    best_model,

    X_test,

    y_test

)

save_feature_importance(

    best_model,

    feature_names

)

save_model_comparison(

    scores

)

print(

    "Graphs Saved Successfully"

)

# ======================================
# Save Classification Report
# ======================================

report_path = os.path.join(

    REPORT_DIR,

    "classification_report.txt"

)

with open(

    report_path,

    "w",

    encoding="utf-8"

) as file:

    file.write(

        report

    )

print()

print(

    "Classification Report Saved"

)

print(

    report_path

)

# ======================================
# Prediction Summary
# ======================================

print()

print("=" * 60)

print("Prediction Summary")

print("=" * 60)

print(

    "Training Samples :",

    len(y_train)

)

print(

    "Testing Samples :",

    len(y_test)

)

print(

    "Total Features :",

    len(feature_names)

)

print("=" * 60)

# ======================================
# Final Results
# ======================================

print()

print("=" * 60)

print("Parkinson's Disease Prediction Completed")

print("=" * 60)

print(

    f"Best Model : {best_name}"

)

print(

    f"Best Accuracy : {best_score:.4f}"

)

print("=" * 60)

# ======================================
# Saved Files
# ======================================

print()

print("=" * 60)

print("Saved Files")

print("=" * 60)

print(

    f"Best Model : {MODEL_PATH}"

)

print(

    f"Scaler : {SCALER_PATH}"

)

print(

    f"Graphs : {GRAPH_DIR}"

)

print(

    f"Report : {report_path}"

)

print("=" * 60)

log_info(

    "Parkinson's Disease Prediction Training Completed Successfully"

)