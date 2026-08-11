import numpy as np

from sklearn.ensemble import RandomForestClassifier

from utils.metrics import *

X = np.random.rand(100,5)

y = np.random.randint(0,2,100)

model = RandomForestClassifier()

model.fit(X,y)

pred = model.predict(X)

prob = model.predict_proba(X)[:,1]

classification_metrics(
    y,
    pred,
    prob
)

save_confusion_matrix(
    y,
    pred
)

save_roc_curve(
    model,
    X,
    y
)

save_feature_importance(
    model,
    [f"Feature {i}" for i in range(5)]
)

save_model_comparison({

    "Random Forest":0.97,

    "SVM":0.95,

    "Logistic":0.93

})