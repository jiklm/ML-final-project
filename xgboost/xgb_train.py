from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
import os
import numpy as np
import pandas as pd
from tqdm import tqdm  # For progress bar

# Feature extraction


def extract_features_from_txt(file_path):
    # Load 6D sensor data (Ax, Ay, Az, Gx, Gy, Gz)
    data = np.loadtxt(file_path)
    features = []

    for i in range(data.shape[1]):
        x = data[:, i]
        features += [
            x.mean(),                 # Mean
            x.std(),                  # Standard deviation
            x.min(),                  # Minimum
            x.max(),                  # Maximum
            np.median(x),            # Median
            np.percentile(x, 25),    # 25th percentile
            np.percentile(x, 75),    # 75th percentile
            np.ptp(x),               # Peak-to-peak range
            np.sum(np.abs(np.diff(x)))  # Signal complexity (total variation)
        ]
    return np.array(features)


# Load training metadata
train_info = pd.read_csv("39_Training_Dataset/train_info.csv")
feature_list = []
label_list = []

# Extract features and labels for each training sample
for _, row in tqdm(train_info.iterrows(), total=len(train_info)):
    file_path = os.path.join(
        "39_Training_Dataset/train_data", f"{row['unique_id']}.txt")
    features = extract_features_from_txt(file_path)
    feature_list.append(features)

    # Label formatting:
    # gender and handed are binary
    # play years: 0/1/2
    # level: 2~5 mapped to 0~3
    label_list.append([
        1 if row["gender"] == 1 else 0,
        1 if row["hold racket handed"] == 1 else 0,
        row["play years"],
        row["level"] - 2
    ])

X = np.stack(feature_list)  # Shape = (num_samples, num_features)
y = np.stack(label_list)    # Shape = (num_samples, 4 tasks)

# Number of classes for each task
classes = {
    "gender": 2,
    "handed": 2,
    "play_years": 3,
    "level": 4
}

# Split the labels into individual tasks
y_gender, y_handed, y_play, y_level = y[:, 0], y[:, 1], y[:, 2], y[:, 3]

# Initialize a multi-output XGBoost model
model = MultiOutputClassifier(XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=3,  # To address imbalance
    use_label_encoder=False,
    eval_metric="logloss"
))

# Use 5-fold stratified cross-validation (stratify on handedness)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = {k: [] for k in classes}  # Store AUCs for each task

# Cross-validation loop
for train_idx, val_idx in skf.split(X, y_handed):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    # Train the model
    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_val)  # Get predicted probabilities

    # Compute AUC for each task
    for i, task in enumerate(classes.keys()):
        if classes[task] == 2:
            # Binary classification: direct AUC
            auc = roc_auc_score(y_val[:, i], y_pred_prob[i][:, 1])
        else:
            # Multi-class: use binarized labels
            y_true_bin = label_binarize(
                y_val[:, i], classes=range(classes[task]))
            y_score = y_pred_prob[i]
            auc = roc_auc_score(y_true_bin, y_score, average='micro')
        auc_scores[task].append(auc)

# Print average AUCs for each task
for task in auc_scores:
    print(f"{task} AUC: {np.mean(auc_scores[task]):.4f}")

# Load test metadata
test_info = pd.read_csv("39_Test_Dataset/test_info.csv")
test_feature_list = []

# Extract features for each test sample
for _, row in tqdm(test_info.iterrows(), total=len(test_info)):
    file_path = os.path.join(
        "39_Test_Dataset/test_data", f"{row['unique_id']}.txt")
    features = extract_features_from_txt(file_path)
    test_feature_list.append(features)

X_test = np.stack(test_feature_list)

# Retrain model on all training data
model.fit(X, y)
y_test_pred_prob = model.predict_proba(X_test)

# Create submission file
submission = pd.DataFrame()
submission["unique_id"] = test_info["unique_id"]

# Binary predictions: use probability of class 1
submission["gender"] = y_test_pred_prob[0][:, 1]
submission["hold racket handed"] = y_test_pred_prob[1][:, 1]

# Multi-class predictions: split into one column per class
play_years_probs = y_test_pred_prob[2]
for i in range(classes["play_years"]):
    submission[f"play years_{i}"] = play_years_probs[:, i]

level_probs = y_test_pred_prob[3]
for i in range(2, 6):  # Original labels were 2 to 5
    submission[f"level_{i}"] = level_probs[:, i - 2]

# Save submission
submission.to_csv("xgb_model.csv", index=False, float_format="%.4f")
print("saved submission.csv")
