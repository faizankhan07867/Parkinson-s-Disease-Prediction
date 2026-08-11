import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

from config import GRAPH_DIR


# ======================================
# Classification Metrics
# ======================================

def classification_metrics(
    y_true,
    y_pred,
    y_prob=None
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred
    )

    recall = recall_score(
        y_true,
        y_pred
    )

    f1 = f1_score(
        y_true,
        y_pred
    )

    auc = None

    if y_prob is not None:

        auc = roc_auc_score(
            y_true,
            y_prob
        )

    print("=" * 60)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    if auc is not None:
        print(f"ROC AUC   : {auc:.4f}")

    print("=" * 60)

    return {

        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC": auc

    }
# ======================================
# Confusion Matrix
# ======================================

def save_confusion_matrix(
    y_true,
    y_pred
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot()

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            GRAPH_DIR,

            "confusion_matrix.png"

        )

    )

    plt.close()
    
# ======================================
# ROC Curve
# ======================================

def save_roc_curve(
    model,
    X_test,
    y_test
):

    if not hasattr(
        model,
        "predict_proba"
    ):
        return

    RocCurveDisplay.from_estimator(

        model,

        X_test,

        y_test

    )

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            GRAPH_DIR,

            "roc_curve.png"

        )

    )

    plt.close()
    
# ======================================
# Feature Importance
# ======================================

def save_feature_importance(
    model,
    feature_names
):

    if not hasattr(
        model,
        "feature_importances_"
    ):
        return

    importance = model.feature_importances_

    order = np.argsort(
        importance
    )

    plt.figure(figsize=(8,6))

    plt.barh(

        np.array(feature_names)[order],

        importance[order]

    )

    plt.xlabel("Importance")

    plt.title("Feature Importance")

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            GRAPH_DIR,

            "feature_importance.png"

        )

    )

    plt.close()
    
# ======================================
# Model Comparison
# ======================================

def save_model_comparison(
    scores
):

    plt.figure(figsize=(8,5))

    plt.bar(

        scores.keys(),

        scores.values()

    )

    plt.ylabel("Accuracy")

    plt.title("Model Comparison")

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            GRAPH_DIR,

            "model_comparison.png"

        )

    )

    plt.close()