import joblib
import sqlite3
import datetime

import pandas as pd
import streamlit as st

from config import *

# ======================================
# Streamlit Configuration
# ======================================

st.set_page_config(

    page_title="Parkinson's Disease Prediction",

    page_icon="🧠",

    layout="wide",

    initial_sidebar_state="expanded"

)

# ======================================
# Custom CSS
# ======================================

st.markdown("""

<style>

.main{

background:#F8F9FA;

}

h1{

color:#1565C0;

}

.stButton>button{

width:100%;

background:#1565C0;

color:white;

border-radius:10px;

font-size:16px;

}

</style>

""", unsafe_allow_html=True)

# ======================================
# Load Model
# ======================================

@st.cache_resource

def load_objects():

    model = joblib.load(

        MODEL_PATH

    )

    scaler = joblib.load(

        SCALER_PATH

    )

    return model, scaler

model, scaler = load_objects()


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
# SQLite Database
# ======================================

connection = sqlite3.connect(

    DATABASE_PATH,

    check_same_thread=False

)

cursor = connection.cursor()

cursor.execute("""

CREATE TABLE IF NOT EXISTS history(

id INTEGER PRIMARY KEY AUTOINCREMENT,

prediction TEXT,

probability REAL,

risk TEXT,

date TEXT

)

""")

connection.commit()

# ======================================
# Sidebar
# ======================================

st.sidebar.title(

    "🧠 Parkinson's Prediction"

)

page = st.sidebar.radio(

    "Navigation",

    [

        "Prediction",

        "Analytics",

        "Batch Prediction",

        "History",

        "About"

    ]

)

# ======================================
# Prediction Page
# ======================================

if page == "Prediction":

    st.title("🧠 Parkinson's Disease Prediction")

    st.write(
        "Enter the patient's voice feature values below."
    )

    values = []

    col1, col2 = st.columns(2)

    for i, feature in enumerate(FEATURES):

        if i % 2 == 0:

            with col1:

                value = st.number_input(

                    feature,

                    value=0.0,

                    format="%.6f",

                    key=feature

                )

        else:

            with col2:

                value = st.number_input(

                    feature,

                    value=0.0,

                    format="%.6f",

                    key=feature

                )

        values.append(value)

    predict_button = st.button(

        "🧠 Predict Disease"

    )
    
    if predict_button:

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

        probability = 0.0

        if hasattr(

            model,

            "predict_proba"

        ):

            probability = model.predict_proba(

                sample

            )[0][1]
        # ======================================
        # Risk Level
        # ======================================

        if probability >= 0.80:

            risk = "🔴 High Risk"

        elif probability >= 0.50:

            risk = "🟠 Moderate Risk"

        else:

            risk = "🟢 Low Risk"
            
        # ======================================
        # Prediction Result
        # ======================================

        if prediction == 1:

            result = "Parkinson's Disease Detected"

            st.error(result)

        else:

            result = "Healthy"

            st.success(result)

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(

                "Prediction",

                result

            )

        with col2:

            st.metric(

                "Probability",

                f"{probability:.2%}"

            )

        with col3:

            st.metric(

                "Risk Level",

                risk

            )
            
        # ======================================
        # Save Prediction
        # ======================================

        cursor.execute(

            """

            INSERT INTO history(

                prediction,

                probability,

                risk,

                date

            )

            VALUES(?,?,?,?)

            """,

            (

                result,

                float(probability),

                risk,

                datetime.datetime.now().strftime(

                    "%d-%m-%Y %H:%M"

                )

            )

        )

        connection.commit()

        st.success(

            "Prediction Saved Successfully"

        )
        # ======================================
        # Patient Summary
        # ======================================

        st.subheader(

            "📋 Patient Summary"

        )

        summary = pd.DataFrame({

            "Feature": FEATURES,

            "Value": values

        })

        st.dataframe(

            summary,

            use_container_width=True,

            hide_index=True

        )
        
        # ======================================
        # Probability Visualization
        # ======================================

        st.subheader(

            "📊 Prediction Probability"

        )

        probability_df = pd.DataFrame({

            "Category": [

                "Healthy",

                "Parkinson's"

            ],

            "Probability": [

                1 - probability,

                probability

            ]

        })

        st.bar_chart(

            probability_df.set_index(

                "Category"

            )

        )
        
        st.info(f"""

### Prediction Summary

**Diagnosis:** {result}

**Risk Level:** {risk}

**Probability:** {probability:.2%}

This prediction is generated using the trained Machine Learning model.
It is intended for educational purposes and should not replace a clinical diagnosis.

""")
        
# ======================================
# History Page
# ======================================

elif page == "History":

    st.title("📋 Prediction History")

    history = pd.read_sql_query(

        "SELECT * FROM history ORDER BY id DESC",

        connection

    )

    if history.empty:

        st.warning(

            "No Prediction History Available"

        )

    else:

        st.dataframe(

            history,

            use_container_width=True

        )
        st.divider()

        st.subheader(

            "📊 Prediction Statistics"

        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(

                "Total Predictions",

                len(history)

            )

        with col2:

            st.metric(

                "Healthy",

                (history["prediction"] == "Healthy").sum()

            )

        with col3:

            st.metric(

                "Detected",

                (

                    history["prediction"]

                    == "Parkinson's Disease Detected"

                ).sum()

            )

        with col4:

            st.metric(

                "Average Probability",

                f"{history['probability'].mean():.2%}"

            )
        st.divider()

        st.subheader(

            "📈 Disease Distribution"

        )

        disease_counts = history[

            "prediction"

        ].value_counts()

        st.bar_chart(

            disease_counts

        )
        
        st.divider()

        st.subheader(

            "🚨 Risk Level Distribution"

        )

        risk_counts = history[

            "risk"

        ].value_counts()

        st.bar_chart(

            risk_counts

        )
        
        st.divider()

        st.subheader(

            "📉 Prediction Probability Trend"

        )

        st.line_chart(

            history["probability"]

        )
        st.divider()

        st.subheader(

            "🔍 Search Predictions"

        )

        option = st.selectbox(

            "Filter by Prediction",

            history["prediction"]

            .unique()

        )

        filtered = history[

            history["prediction"]

            == option

        ]

        st.dataframe(

            filtered,

            use_container_width=True

        )
        
        st.divider()

        csv = history.to_csv(

            index=False

        ).encode(

            "utf-8"

        )

        st.download_button(

            "⬇ Download Prediction History",

            csv,

            file_name="prediction_history.csv",

            mime="text/csv"

        )
        
        if st.button(

            "🗑️ Clear History"

        ):

            cursor.execute(

                "DELETE FROM history"

            )

            connection.commit()

            st.success(

                "Prediction History Cleared"

            )

            st.rerun()
            
# ======================================
# Batch Prediction
# ======================================

elif page == "Batch Prediction":

    st.title("📂 Batch Parkinson's Prediction")

    uploaded_file = st.file_uploader(

        "Upload CSV File",

        type=["csv"]

    )

    if uploaded_file is not None:

        batch_df = pd.read_csv(

            uploaded_file

        )
        
        missing = [

            column

            for column in FEATURES

            if column not in batch_df.columns

        ]

        if missing:

            st.error(

                f"Missing Columns : {missing}"

            )

        else:

            sample = batch_df[FEATURES].copy()
            
            scaled = scaler.transform(

                sample

            )

            predictions = model.predict(

                scaled

            )

            batch_df["Prediction"] = predictions

            batch_df["Prediction"] = batch_df[

                "Prediction"

            ].map({

                0:"Healthy",

                1:"Parkinson's Disease"

            })
            
            if hasattr(

                model,

                "predict_proba"

            ):

                probabilities = model.predict_proba(

                    scaled

                )[:,1]

                batch_df["Probability"] = probabilities

                batch_df["Risk"] = batch_df[

                    "Probability"

                ].apply(

                    lambda x:

                    "High"

                    if x>=0.80

                    else

                    "Moderate"

                    if x>=0.50

                    else

                    "Low"

                )
            st.success(

                "Batch Prediction Completed"

            )

            col1,col2,col3 = st.columns(3)

            with col1:

                st.metric(

                    "Patients",

                    len(batch_df)

                )

            with col2:

                st.metric(

                    "Detected",

                    (

                        batch_df["Prediction"]

                        ==

                        "Parkinson's Disease"

                    ).sum()

                )

            with col3:

                st.metric(

                    "Healthy",

                    (

                        batch_df["Prediction"]

                        ==

                        "Healthy"

                    ).sum()

                )
                
            st.subheader(

                "📊 Disease Distribution"

            )

            st.bar_chart(

                batch_df["Prediction"]

                .value_counts()

            )
            
            if "Risk" in batch_df.columns:

                st.subheader(

                    "🚨 Risk Distribution"

                )

                st.bar_chart(

                    batch_df["Risk"]

                    .value_counts()

                )
            st.subheader(

                "📋 Prediction Results"

            )

            st.dataframe(

                batch_df,

                use_container_width=True

            )
            csv = batch_df.to_csv(

                index=False

            ).encode(

                "utf-8"

            )

            st.download_button(

                "⬇ Download Prediction Report",

                csv,

                file_name="parkinsons_predictions.csv",

                mime="text/csv"

            )
            
# ======================================
# Analytics Page
# ======================================

elif page == "Analytics":

    st.title("📊 Model Analytics")

    st.metric(

        "Features",

        len(FEATURES)

    )

    st.metric(

        "Prediction Model",

        "Best Trained Model"

    )

    st.subheader(

        "Feature List"

    )

    feature_df = pd.DataFrame({

        "Features":FEATURES

    })

    st.dataframe(

        feature_df,

        use_container_width=True

    )
    
    if hasattr(

        model,

        "feature_importances_"

    ):

        importance = pd.DataFrame({

            "Feature":FEATURES,

            "Importance":model.feature_importances_

        })

        importance = importance.sort_values(

            "Importance",

            ascending=False

        )

        st.subheader(

            "📈 Feature Importance"

        )

        st.bar_chart(

            importance.set_index(

                "Feature"

            )

        )
        
# ======================================
# About Page
# ======================================

elif page == "About":

    st.title("🧠 About Parkinson's Disease Prediction")

    st.markdown("""

## 🧠 Parkinson's Disease Prediction using Machine Learning

This application predicts the likelihood of Parkinson's Disease
using voice measurement features and multiple Machine Learning models.

### Features

✅ Parkinson's Disease Prediction

✅ Batch CSV Prediction

✅ Prediction History

✅ Risk Level Classification

✅ Feature Importance Visualization

✅ SQLite Database

✅ CSV Export

✅ Professional Streamlit Dashboard

""")
    
st.divider()

st.subheader("🤖 Model Information")

model_df = pd.DataFrame({

    "Parameter":[

        "Primary Model",

        "Alternative Models",

        "Framework",

        "Target",

        "Features"

    ],

    "Value":[

        "Best Selected Model",

        "Logistic Regression, SVM, Random Forest, XGBoost",

        "Scikit-Learn + XGBoost",

        "Parkinson's Disease",

        len(FEATURES)

    ]

})

st.table(model_df)

st.divider()

st.subheader("⚙️ Training Configuration")

config_df = pd.DataFrame({

    "Parameter":[

        "Test Size",

        "Cross Validation",

        "Random State",

        "Scaling"

    ],

    "Value":[

        TEST_SIZE,

        CV_FOLDS,

        RANDOM_STATE,

        "StandardScaler"

    ]

})

st.table(config_df)

st.divider()

st.subheader("📈 Evaluation Metrics")

metric_df = pd.DataFrame({

    "Metric":[

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score",

        "ROC-AUC"

    ],

    "Purpose":[

        "Overall Performance",

        "Positive Prediction Quality",

        "Disease Detection Rate",

        "Balanced Performance",

        "Model Discrimination"

    ]

})

st.table(metric_df)

st.divider()

st.subheader("🏥 Medical Disclaimer")

st.warning("""

This application is designed for educational and research purposes only.

Predictions generated by this model should NOT be used as a substitute
for professional medical diagnosis or treatment.

Always consult a qualified neurologist or healthcare professional
for clinical evaluation and diagnosis.

""")

st.divider()

st.subheader("👨‍💻 Developer")

st.info("""

Name : Faizan Khan

Project :
Parkinson's Disease Prediction

Technology Stack

• Python

• Pandas

• NumPy

• Scikit-Learn

• XGBoost

• Streamlit

• SQLite

• Matplotlib

Academic Project

B.Tech Information Technology

""")

# ======================================
# Footer
# ======================================

st.divider()

st.markdown("""

<div style="text-align:center;
padding:15px;
font-size:18px;">

🧠 Parkinson's Disease Prediction

Made with ❤️ using Python, Scikit-Learn & Streamlit

© 2026 Faizan Khan

</div>

""", unsafe_allow_html=True)

connection.close()