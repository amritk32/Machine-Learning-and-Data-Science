# 🏎️ Pit Strategy Prediction using Machine Learning (F1 Dataset)

This project is a machine learning pipeline that predicts whether a driver will pit on the next lap using race telemetry data. The solution uses feature engineering, multiple ML models, and a soft-voting ensemble classifier.

---

## 🚀 Problem Statement

Given lap-by-lap racing data, predict:

> **PitNextLap (Binary Classification)**  
Whether a driver will pit in the next lap or not.

---

## 📊 Dataset

- Source: Kaggle Playground Series S6E5
- Rows: Lap-level telemetry data
- Target: `PitNextLap`

---

## 🧠 Approach

### 1. Data Preprocessing
- Removed irrelevant columns (`id`, `Driver`, `RaceProgress`) for training purpose
- Handled missing values
- Analyzed through Exploratory Data Analysis
- Stratified train-test split

### 2. Feature Engineering
Created domain-specific racing features:
- `Stint_Fatigue = Stint × LapNumber`
- `Pace_Efficiency = Stint / LapTime`
- `Tire_Danger_Index = Cumulative_Degradation × TyreLife`
- `Compound_Pit_Probability` (target encoding using training data only)
- Interaction features like `Z`, `Critical_Pit_Alarm`

---

### 3. Encoding
- OneHotEncoding applied to:
  - `Compound`
  - `Race`
- Ensured train-test consistency using fitted encoder

---

### 4. Models Used

Baseline models:
- Logistic Regression
- Ridge Classifier
- Random Forest
- KNN

Boosting models:
- XGBoost
- LightGBM

---

### 5. Final Model
A **Soft Voting Classifier**:

This ensemble improved stability and ROC-AUC performance.

---

## 📈 Results

| Model            | ROC-AUC |
|------------------|--------|
| XGBoost          | ~0.9483  |
| LightGBM         | ~0.9474|
| Random Forest    | ~0.9410 |
| Ensemble (Final) | ~0.94|

---

## 🏆 Kaggle Performance

- Score: ~0.947 (leaderboard)
- Rank: ~890 / 1500+

---

## 🧩 Key Learnings

- Importance of feature engineering > model tuning
- Target leakage awareness is critical
- Tree models outperform linear models on noisy racing data
- Proper train-test feature alignment is crucial
- Ensemble models improve stability

---

## ⚠️ Common Mistakes (Fixed in this project)

- Incorrect target selection (initial confusion between PitStop vs PitNextLap)
- Feature mismatch between train and test pipeline
- Encoding mismatch during inference

---

## 🛠 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- LightGBM
