# 🚀 AI-Based Credit Card Fraud Detection

An end-to-end Machine Learning project designed to detect fraudulent credit card transactions using advanced ML, ensemble learning, and boosting techniques on highly imbalanced financial datasets.

---

## 📌 Project Overview

Financial fraud detection is a critical problem in banking and fintech due to extreme class imbalance. Fraudulent transactions usually represent less than 1% of total transactions, making accurate detection challenging.

This project builds a robust fraud detection system that:

- Detects fraudulent transactions with high recall
- Minimizes false positives using precision-focused models
- Handles extreme class imbalance effectively
- Compares multiple ML and boosting algorithms
- Evaluates models using multiple performance metrics

---

## 🎯 Objectives

- Detect fraudulent credit card transactions accurately
- Compare multiple machine learning models
- Improve model performance using class balancing techniques
- Evaluate models using Precision, Recall, F1-Score, and AUC

---

## 🚀 Models Implemented

- Logistic Regression
- Random Forest Classifier
- XGBoost
- Deep Learning Models (ANN)

---

## 📊 Model Performance

| Model                | Precision | Recall | F1-Score | AUC  |
|----------------------|-----------|--------|----------|------|
| Logistic Regression  | 0.06      | 0.91   | 0.11     | 0.94 |
| Random Forest        | 0.96      | 0.74   | 0.83     | 0.87 |
| XGBoost              | 0.53      | 0.85   | 0.65     | 0.92 |

---

## 🧠 Key Techniques Used

- Handling Imbalanced Data
- Class Weight Adjustment
- Feature Scaling (StandardScaler)
- Ensemble Learning
- Boosting (XGBoost)
- Model Comparison Framework
- Confusion Matrix Visualization

---

## 🛠 Tech Stack

- Python
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## 📁 Project Structure

```
AI-Based-Fraud-Detection/
│
├── MachineLearningModels.py
├── DeepLearningModels.py
├── BalanceDataset.py
├── creditcard/
├── ieee-fraud-detection/
├── Project Reports/
└── README.md
```

---

## ▶️ How to Run

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run:
   ```
   python MachineLearningModels.py
   ```

---

## 📌 Results Summary

- Logistic Regression achieved high recall but low precision.
- Random Forest achieved the best overall balance.
- XGBoost provided strong performance with better recall than Random Forest.
- Class balancing significantly improved fraud detection capability.

---

## 👤 Author

**Japamani Mundas**  
Machine Learning & Backend Developer  
GitHub: https://github.com/jap6358
