# Google Play Revenue Predictor

Predicts expected installs and revenue for a paid app on Google Play
based on its rating and price, using a Linear Regression model trained
on real Google Play Store data.

## Project Structure
```
├── Google_Play_Market_Structure_Project.ipynb  # Full analysis & model training
├── app.py                                       # Standalone prediction script
├── revenue_model.pkl                            # Saved trained model
└── README.md
```

## Setup
```bash
pip install scikit-learn numpy joblib
```

## Usage
```bash
python app.py
```

Then enter your app's rating and price when prompted.

## Example Output
```
=====================================
       APP REVENUE PREDICTION        
=====================================
  App Rating         : 4.5
  App Price          : $2.99
-------------------------------------
  Expected Installs  : 1,243
  Estimated Revenue  : $3,716.57
=====================================
```

## How It Works

- Model is trained **only on paid apps** (Price > 0) from the Google Play dataset
- Features used: `Rating` and `Price`
- Target: `Log(Installs)` — converted back to actual installs at prediction time
- Revenue = Predicted Installs × Price

## Model Performance

| Metric | Value |
|--------|-------|
| R² Score | See notebook output |
| MAE | See notebook output |

## Dataset

[Google Play Store Apps — Kaggle](https://www.kaggle.com/datasets/gauthamp10/google-playstore-apps)
