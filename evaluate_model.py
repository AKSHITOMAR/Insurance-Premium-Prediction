import pickle
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from config.city_tier import tier_1_cities, tier_2_cities


# ============================================
# LOAD MODEL
# ============================================

with open("Model/model.pkl", "rb") as f:
    model = pickle.load(f)


# ============================================
# LOAD DATASET
# ============================================

df = pd.read_csv(r"C:\Users\acer\Downloads\insurance.csv")


# ============================================
# FEATURE ENGINEERING
# Same logic as UserInput
# ============================================

df["bmi"] = df["weight"] / (df["height"] ** 2)


def get_lifestyle_risk(row):
    if row["smoker"] and row["bmi"] > 30:
        return "high"
    elif row["smoker"] or row["bmi"] > 27:
        return "medium"
    else:
        return "low"


def get_age_group(age):
    if age < 25:
        return "young"
    elif age < 45:
        return "adult"
    elif age < 60:
        return "middle_aged"
    return "senior"


def get_city_tier(city):
    if city in tier_1_cities:
        return 1
    elif city in tier_2_cities:
        return 2
    else:
        return 3


df["lifestyle_risk"] = df.apply(get_lifestyle_risk, axis=1)
df["age_group"] = df["age"].apply(get_age_group)
df["city_tier"] = df["city"].apply(get_city_tier)


# ============================================
# INPUT FEATURES
# ============================================

X = df[
    [
        "bmi",
        "age_group",
        "lifestyle_risk",
        "city_tier",
        "income_lpa",
        "occupation"
    ]
]

y = df["insurance_premium_category"]


# ============================================
# PREDICTION
# ============================================

y_pred = model.predict(X)


# ============================================
# EVALUATION
# ============================================

accuracy = accuracy_score(y, y_pred)

precision = precision_score(
    y,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y,
    y_pred,
    average="weighted",
    zero_division=0
)


# ============================================
# RESULTS
# ============================================

print("\n============================================")
print("       INSURANCE PREMIUM MODEL EVALUATION")
print("============================================")

print(f"\nDataset Size : {len(df)}")

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")


# ============================================
# CONFUSION MATRIX
# ============================================

print("\n============================================")
print("CONFUSION MATRIX")
print("============================================")

labels = ["High", "Low", "Medium"]

cm = confusion_matrix(y, y_pred, labels=labels)

print("\n             Predicted")
print("           High  Low  Medium")

for label, row in zip(labels, cm):
    print(f"Actual {label:<7}", row)


# ============================================
# CLASSIFICATION REPORT
# ============================================

print("\n============================================")
print("CLASSIFICATION REPORT")
print("============================================")

print(
    classification_report(
        y,
        y_pred,
        labels=labels,
        zero_division=0
    )
)