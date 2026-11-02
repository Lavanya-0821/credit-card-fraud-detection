# ============================================================
# CREDIT CARD FRAUD DETECTION
# TRAIN MODEL
# ============================================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("Loading dataset...")

df = pd.read_csv("creditcard.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nClass Distribution:")
print(df["Class"].value_counts())


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("Class", axis=1)

y = df["Class"]


# ============================================================
# 3. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data :", X_test.shape)


# ============================================================
# 4. FEATURE SCALING
# ============================================================

print("\nScaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ============================================================
# 5. HANDLE IMBALANCED DATA USING SMOTE
# ============================================================

print("\nApplying SMOTE...")

smote = SMOTE(
    random_state=42
)

X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("Before SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(y_train_balanced.value_counts())


# ============================================================
# 6. CREATE RANDOM FOREST MODEL
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 7. TRAIN MODEL
# ============================================================

model.fit(
    X_train_balanced,
    y_train_balanced
)

print("Random Forest training completed!")


# ============================================================
# 8. PREDICTION
# ============================================================

print("\nTesting model...")

y_pred = model.predict(
    X_test_scaled
)

y_prob = model.predict_proba(
    X_test_scaled
)[:, 1]


# ============================================================
# 9. EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

print("\n")
print("==============================================")
print("       RANDOM FOREST MODEL RESULTS")
print("==============================================")

print("Accuracy  :", round(accuracy, 4))
print("Precision :", round(precision, 4))
print("Recall    :", round(recall, 4))
print("F1-Score  :", round(f1, 4))
print("ROC-AUC   :", round(roc_auc, 4))

print("==============================================")


# ============================================================
# 11. SAVE TRAINED MODEL
# ============================================================

joblib.dump(
    model,
    "fraud_model.pkl"
)

print("\nModel saved successfully!")
print("File created: fraud_model.pkl")


# ============================================================
# 12. SAVE SCALER
# ============================================================

joblib.dump(
    scaler,
    "scaler.pkl"
)

print("Scaler saved successfully!")
print("File created: scaler.pkl")


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n==============================================")
print("TRAINING COMPLETED SUCCESSFULLY!")
print("==============================================")
print("You can now run app.py")