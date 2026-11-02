# ============================================================
# CREDIT CARD FRAUD DETECTION USING MACHINE LEARNING
# ============================================================

# Step 1: Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Classification Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc
)

# SMOTE
from imblearn.over_sampling import SMOTE


# ============================================================
# Step 2: Load Dataset
# ============================================================

df = pd.read_csv("creditcard.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nClass Distribution:")
print(df["Class"].value_counts())


# ============================================================
# Step 3: Separate Features and Target
# ============================================================

X = df.drop("Class", axis=1)
y = df["Class"]

print("\nFeatures Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)


# ============================================================
# Step 4: Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:")
print(X_train.shape)

print("\nTesting Data:")
print(X_test.shape)


# ============================================================
# Step 5: Feature Scaling
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature Scaling Completed!")


# ============================================================
# Step 6: Apply SMOTE
# ============================================================

smote = SMOTE(random_state=42)

X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("\nBefore SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(y_train_balanced.value_counts())


# ============================================================
# Step 7: Create 5 Classification Models
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=50,
            random_state=42,
            n_jobs=-1
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=5
        ),

    "SVM":
        LinearSVC(
            random_state=42,
            max_iter=5000
        )
}


# ============================================================
# Step 8: Train and Evaluate Models
# ============================================================

results = []

for name, model in models.items():

    print("\n====================================")
    print("Training:", name)
    print("====================================")

    # Train model
    model.fit(
        X_train_balanced,
        y_train_balanced
    )

    # Predict class
    y_pred = model.predict(
        X_test_scaled
    )

    # Get probability or decision score
    if hasattr(model, "predict_proba"):

        y_prob = model.predict_proba(
            X_test_scaled
        )[:, 1]

    else:

        y_prob = model.decision_function(
            X_test_scaled
        )

    # Calculate Accuracy
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    # Calculate Precision
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # Calculate Recall
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # Calculate F1-Score
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # Calculate ROC-AUC
    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    # Store results
    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ])

    # Display results
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1-Score :", f1)
    print("ROC-AUC  :", roc_auc)


# ============================================================
# Step 9: Model Comparison Table
# ============================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC"
    ]
)

print("\n====================================")
print("MODEL COMPARISON")
print("====================================")

print(results_df)


# ============================================================
# Step 10: Confusion Matrix
# ============================================================

for name, model in models.items():

    # Predict test data
    y_pred = model.predict(
        X_test_scaled
    )

    # Create confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred
    )

    # Plot
    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Fraud"],
        yticklabels=["Normal", "Fraud"]
    )

    plt.title(
        "Confusion Matrix - " + name
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()
    plt.show()


# ============================================================
# Step 11: Model Performance Comparison Graph
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "ROC-AUC"
]

for metric in metrics:

    plt.figure(figsize=(10, 5))

    plt.bar(
        results_df["Model"],
        results_df[metric]
    )

    plt.title(
        metric + " Comparison"
    )

    plt.xlabel(
        "Classification Model"
    )

    plt.ylabel(
        metric
    )

    plt.xticks(
        rotation=20
    )

    plt.ylim(
        0,
        1
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# Step 12: Find Best Model
# ============================================================

best_model = results_df.loc[
    results_df["F1-Score"].idxmax()
]

print("\n====================================")
print("BEST MODEL")
print("====================================")

print(
    "Model     :",
    best_model["Model"]
)

print(
    "Accuracy  :",
    best_model["Accuracy"]
)

print(
    "Precision :",
    best_model["Precision"]
)

print(
    "Recall    :",
    best_model["Recall"]
)

print(
    "F1-Score  :",
    best_model["F1-Score"]
)

print(
    "ROC-AUC   :",
    best_model["ROC-AUC"]
)


# ============================================================
# Step 13: ROC Curve
# ============================================================

plt.figure(figsize=(10, 7))

for name, model in models.items():

    # Get probability or decision score
    if hasattr(model, "predict_proba"):

        y_prob = model.predict_proba(
            X_test_scaled
        )[:, 1]

    else:

        y_prob = model.decision_function(
            X_test_scaled
        )

    # Calculate ROC values
    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_prob
    )

    # Calculate AUC
    roc_auc = auc(
        fpr,
        tpr
    )

    # Plot ROC curve
    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {roc_auc:.3f})"
    )


# Random classifier line
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve - Classification Models"
)

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


# ============================================================
# Step 14: Save Results
# ============================================================

results_df.to_csv(
    "model_comparison.csv",
    index=False
)

print(
    "\nModel comparison saved successfully!"
)


# ============================================================
# Step 15: Fraud Prediction Function
# ============================================================

def predict_fraud(model, transaction):

    # Convert transaction into 2D format
    transaction = [transaction]

    # Scale transaction
    transaction_scaled = scaler.transform(
        transaction
    )

    # Predict
    prediction = model.predict(
        transaction_scaled
    )[0]

    # Get probability or decision score
    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(
            transaction_scaled
        )[0][1]

        probability_percent = probability * 100

    else:

        decision_score = model.decision_function(
            transaction_scaled
        )[0]

        # Convert decision score to approximate percentage
        probability_percent = (
            1 / (1 + np.exp(-decision_score))
        ) * 100

    # Display result
    if prediction == 1:

        print("\n⚠️ FRAUD TRANSACTION")

    else:

        print("\n✅ NORMAL TRANSACTION")

    print(
        "Fraud Probability/Score:",
        round(probability_percent, 2),
        "%"
    )


# ============================================================
# Step 16: Select Best Model
# ============================================================

best_model_name = best_model["Model"]

best_model_object = models[
    best_model_name
]

print(
    "\nSelected Best Model:",
    best_model_name
)


# ============================================================
# Step 17: Take One Existing Transaction
# ============================================================

sample_transaction = X_test.iloc[0].values


# ============================================================
# Step 18: Predict Fraud
# ============================================================

predict_fraud(
    best_model_object,
    sample_transaction
)