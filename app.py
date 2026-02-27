import joblib
import numpy as np

# Load the trained model
model = joblib.load('revenue_model.pkl')

def predict_revenue(rating, price):
    if price <= 0:
        print("ERROR: Price must be greater than 0 for a paid app.")
        return
    if not (1.0 <= rating <= 5.0):
        print("ERROR: Rating must be between 1.0 and 5.0.")
        return

    log_installs = model.predict([[rating, price]])[0]
    installs = max(0, round(np.expm1(log_installs)))
    revenue = installs * price

    print("=====================================")
    print("       APP REVENUE PREDICTION        ")
    print("=====================================")
    print(f"  App Rating         : {rating}")
    print(f"  App Price          : ${price:.2f}")
    print("-------------------------------------")
    print(f"  Expected Installs  : {installs:,}")
    print(f"  Estimated Revenue  : ${revenue:,.2f}")
    print("=====================================")

if __name__ == "__main__":
    rating = float(input("Enter app rating (1.0 - 5.0): "))
    price  = float(input("Enter app price in USD (e.g. 2.99): "))
    predict_revenue(rating, price)
