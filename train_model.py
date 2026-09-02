import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

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
# LOAD DATA
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
# PREPROCESSING
# ============================================

categorical_features = [
    "age_group",
    "lifestyle_risk",
    "occupation"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ============================================
# MODEL
# ============================================

classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ]
)

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

print("\n============================================")
print("5-FOLD CROSS VALIDATION")
print("============================================")

for i, score in enumerate(cv_scores, 1):
    print(f"Fold {i} Accuracy : {score:.4f}")

print(f"\nMean Accuracy : {cv_scores.mean():.4f}")
print(f"Std Deviation : {cv_scores.std():.4f}")


# ============================================
# TRAIN
# ============================================

model.fit(X_train, y_train)


# ============================================
# PREDICTION
# ============================================

y_pred = model.predict(X_test)


# ============================================
# EVALUATION
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
# RESULTS
# ============================================

print("\n============================================")
print("       MODEL EVALUATION")
print("============================================")

print(f"Total samples : {len(df)}")
print(f"Training data : {len(X_train)}")
print(f"Testing data  : {len(X_test)}")

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")


# ============================================
# CONFUSION MATRIX
# ============================================

labels = ["High", "Low", "Medium"]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print("\n============================================")
print("CONFUSION MATRIX")
print("============================================")

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
        y_test,
        y_pred,
        labels=labels,
        zero_division=0
    )
)


# ============================================
# SAVE MODEL
# ============================================

with open("Model/model_evaluated.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved as:")
print("Model/model_evaluated.pkl")