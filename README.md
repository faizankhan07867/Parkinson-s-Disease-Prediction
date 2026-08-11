# Parkinson-s-Disease-Prediction

# 🧠 Parkinson's Disease Prediction

## 📌 Overview

Parkinson's Disease Prediction is a Machine Learning project that predicts whether a person is likely to have Parkinson's disease using voice measurement features. The system uses a Support Vector Machine (SVM) model and provides instant predictions through a Flask web application.

---

## ✨ Features

- 🧠 Parkinson's Disease Prediction
- 🎤 Voice Feature Analysis
- 🤖 SVM Machine Learning Model
- 📊 Prediction with Confidence Score
- 🌐 Flask Web Application
- ⚡ Fast & Accurate Prediction

---

## 🛠 Technologies Used

- Python
- Flask
- Scikit-Learn
- Pandas
- NumPy
- Joblib
- HTML
- CSS

---

## 📂 Dataset

Dataset Name:

Parkinson's Disease Dataset

Features:

- MDVP:Fo(Hz)
- MDVP:Fhi(Hz)
- MDVP:Flo(Hz)
- MDVP:Jitter(%)
- MDVP:Shimmer
- Other Voice Features

Output Classes:

- Healthy
- Parkinson's Disease

---

## 📁 Project Structure

```
Parkinsons-Disease-Prediction/

│── app.py
│── train.py
│── predict.py
│── preprocess.py
│── requirements.txt
│── README.md

├── model/
│     parkinsons_model.pkl
│     scaler.pkl

├── dataset/
│     parkinsons.csv

├── templates/
│     index.html

├── static/
│     style.css

└── screenshots/
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Train Model

```bash
python train.py
```

---

## ▶️ Run Project

```bash
python app.py
```

Open Browser

```
http://127.0.0.1:5000
```

---

## 📊 Output

- Enter Voice Measurement Values
- Click **Predict**
- AI Predicts Disease Status
- Displays Prediction Result
- Shows Confidence Score

---

## 🎯 Future Improvements

- Voice Recording Support
- Mobile Application
- Real-Time Voice Analysis
- Healthcare Dashboard
- Cloud Deployment
- Doctor Recommendation System

---

## 👨‍💻 Author

**Faizan Khan**

B.Tech Information Technology

AI | Machine Learning | Data Science | Analytics

---

## 📜 License

MIT License
