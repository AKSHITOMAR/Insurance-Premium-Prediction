import pickle
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split
)

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
    return 3


df["lifestyle_risk"] = df.apply(get_lifestyle_risk, axis=1)
df["age_group"] = df["age"].apply(get_age_group)
df["city_tier"] = df["city"].apply(get_city_tier)


# ============================================
# FEATURES & TARGET
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
# 5-FOLD CROSS VALIDATION
# ============================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)


# ============================================
# TRAIN / TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================
# TRAIN MODEL
# ============================================

model.fit(X_train, y_train)


# ============================================
# PREDICTION
# ============================================

y_pred = model.predict(X_test)


# ============================================
# METRICS
# ============================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


# ============================================
# CONFUSION MATRIX
# ============================================

labels = ["High", "Low", "Medium"]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)


# ============================================
# CLASSIFICATION REPORT
# ============================================

report = classification_report(
    y_test,
    y_pred,
    labels=labels,
    output_dict=True,
    zero_division=0
)


# ============================================
# FINAL RESULTS
# ============================================

evaluation_results = {
    "cv_accuracy": cv_scores.mean(),
    "cv_std": cv_scores.std(),

    "test_accuracy": accuracy,
    "test_precision": precision,
    "test_recall": recall,
    "test_f1": f1,

    "confusion_matrix": cm,
    "classification_report": report,

    "labels": labels,

    "dataset_size": len(df),
    "training_size": len(X_train),
    "testing_size": len(X_test)
}