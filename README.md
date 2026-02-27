# 📱 Google Play App Market Structure & Revenue Predictor

A data science project that analyzes the Google Play Store app market using **K-Means Clustering** and **Linear Regression** to predict expected installs and estimated revenue for paid apps.

---

## 📁 Project Structure

```
├── Google_Play_Market_Structure_Project.ipynb   # Full analysis & model training notebook
├── app.py                                        # Streamlit web app
├── revenue_model.pkl                             # Pre-trained Linear Regression model
├── requirements.txt                              # Python dependencies
└── README.md                                     # Project documentation
```

---

## 🎯 Project Objectives

- **Data Cleaning & Preprocessing** — Handle raw messy Google Play data (string-formatted installs, prices, missing values)
- **Exploratory Data Analysis (EDA)** — Understand distributions, trends, and correlations across 6 visual analyses
- **K-Means Clustering** — Segment the app market into 4 distinct clusters based on Rating, Price, and Installs
- **Demand Estimation (Regression)** — Train a Linear Regression model on paid apps only to predict installs from Rating and Price
- **Revenue Prediction** — Estimate expected revenue using `Revenue = Predicted Installs × Price`
- **Business Interpretation** — Link findings to pricing strategy, demand elasticity, and market segmentation

---

## ⚙️ Setup & Installation

**1. Clone or download the project folder**

**2. Install all dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the Streamlit app**
```bash
streamlit run app.py
```

The app will open in your browser automatically at `http://localhost:8501`

---

## 🤖 About the Model (`revenue_model.pkl`)

- The model is a **Linear Regression** trained exclusively on **paid apps** (Price > 0) from the Google Play Store dataset
- **Features used:** `Rating`, `Price`
- **Target:** `Log(Installs)` — predicted value is converted back to actual installs at runtime
- The `.pkl` file is pre-trained and ready to use — no need to re-run the notebook just to use the app

To retrain the model from scratch, run all cells in the notebook. The final training cell saves a fresh `revenue_model.pkl`:
```python
import joblib
joblib.dump(model, 'revenue_model.pkl')
```

---

## 🚀 How the App Works

1. Open the Streamlit app (`streamlit run app.py`)
2. Enter your app's **Rating** (1.0 – 5.0) using the slider
3. Enter your app's **Price** in USD
4. Click **Predict Revenue**
5. The app shows your **Expected Installs** and **Estimated Revenue**

---

## 📊 Notebook Walkthrough

| Step | Description |
|------|-------------|
| Step 1 | Load dataset from Kaggle via `kagglehub` |
| Step 2 | Data Cleaning — fix string-formatted installs/prices, remove duplicates, handle nulls |
| Step 3 | EDA — 6 visual analyses (ratings, installs, free vs paid, categories, scatter, heatmap) |
| Step 4 | K-Means Clustering — segment market into 4 clusters |
| Step 5 | Linear Regression — train on paid apps, evaluate with R² and MAE |
| Step 6 | Revenue Prediction — `predict_revenue(rating, price)` function |
| Step 7 | Business Interpretation — pricing strategy, demand elasticity, market segments |

---

## 📦 Requirements

All dependencies are listed in `requirements.txt`. Key libraries:

| Library | Purpose |
|---------|---------|
| `streamlit` | Web app interface |
| `scikit-learn` | K-Means Clustering & Linear Regression |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualizations |
| `joblib` | Save & load model (.pkl) |
| `kagglehub` | Download dataset from Kaggle |

---

## 💡 Business Interpretation

- **Rating as a demand signal** — Higher-rated apps attract more installs; the model's Rating coefficient quantifies this effect
- **Price elasticity** — Higher prices reduce install volume (negative Price coefficient), classic demand elasticity
- **Market Segmentation** — 4 clusters reveal distinct app market types (e.g. cheap/popular, niche/premium, low-rated/bargain)
- **Revenue optimization** — There is an optimal price point balancing install volume and per-unit revenue; use `predict_revenue()` to simulate scenarios

---

## 📂 Dataset

**Google Play Store Apps** — [Kaggle Dataset by Gauthamp10](https://www.kaggle.com/datasets/gauthamp10/google-playstore-apps)

Downloaded automatically via `kagglehub` when running the notebook.


## Screenshots
