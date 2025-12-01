import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
import pickle
import os

# ✅ Make sure model folder exists
os.makedirs("model", exist_ok=True)

# Common LabelEncoder
le = LabelEncoder()

# ---------------------------
# 🐶 1. Train Animal Model
# ---------------------------
try:
    df_animal = pd.read_csv("model/AnimalInformation.csv")
    df_animal.columns = df_animal.columns.str.strip()  # remove \n or spaces

    X = df_animal[["Animals", "Size of Animal", "Body Color"]]
    y = df_animal["Can we Pet them"]

    for col in X.columns:
        X[col] = le.fit_transform(X[col])
    y = le.fit_transform(y)

    animal_model = GaussianNB()
    animal_model.fit(X, y)

    with open("model/animal_model.pkl", "wb") as f:
        pickle.dump(animal_model, f)
    print("✅ animal_model.pkl saved successfully!")
except Exception as e:
    print("❌ Animal model error:", e)


# ---------------------------
# 📧 2. Train Email Spam Model
# ---------------------------
try:
    df_email = pd.read_csv("model/EmailSpamDetectionUpdated.csv")  # ✅ Fixed name (case-sensitive)
    df_email.columns = df_email.columns.str.strip()

    X = df_email[["Contains Offer", "Contains Link", "Contains Greeting", "Sender Known"]]
    y = df_email["Spam"]

    for col in X.columns:
        X[col] = le.fit_transform(X[col])
    y = le.fit_transform(y)

    email_model = GaussianNB()
    email_model.fit(X, y)

    with open("model/email_model.pkl", "wb") as f:
        pickle.dump(email_model, f)
    print("✅ email_model.pkl saved successfully!")
except Exception as e:
    print("❌ Email model error:", e)


# ---------------------------
# 💰 3. Train Loan Approval Model
# ---------------------------
try:
    df_loan = pd.read_csv("model/LoanApprovalupdated.csv")  # ✅ Fixed filename
    df_loan.columns = df_loan.columns.str.strip()

    # Print columns to check what’s inside
    print("Loan Columns:", df_loan.columns.tolist())

    # Adjust names if your CSV has newlines or different casing
    X = df_loan[["Age", "Income", "Credit", "Employment"]]
    y = df_loan["LoanApproved"]  # ✅ match exact column name

    for col in X.columns:
        X[col] = le.fit_transform(X[col])
    y = le.fit_transform(y)

    loan_model = GaussianNB()
    loan_model.fit(X, y)

    with open("model/loan_model.pkl", "wb") as f:
        pickle.dump(loan_model, f)
    print("✅ loan_model.pkl saved successfully!")
except Exception as e:
    print("❌ Loan model error:", e)


# ---------------------------
# 🌦️ 4. Train Weather & Road Condition Model
# ---------------------------
# 🌦️ 4. Train Weather Model (Fixed)
try:
    df_weather = pd.read_csv("model/Weatherandroadcondition.csv")
    print("Weather Columns:", df_weather.columns.tolist())

    # ✅ Use actual column names from CSV
    X = df_weather[["Weather Condition", "Road Condition", "Traffic Condition", "Engine Problem"]]
    y = df_weather["Accident"]

    for col in X.columns:
        X[col] = le.fit_transform(X[col])
    y = le.fit_transform(y)

    weather_model = GaussianNB()
    weather_model.fit(X, y)

    with open("model/weather_model.pkl", "wb") as f:
        pickle.dump(weather_model, f)
    print("✅ weather_model.pkl saved successfully!")
except Exception as e:
    print("❌ Weather model error:", e)

