from sklearn.model_selection import StratifiedKFold
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

# ---------- Configuration ----------
DATA_DIR = '39_Training_Dataset'
TRAIN_INFO_PATH = os.path.join(DATA_DIR, 'train_info.csv')
TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train_data')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
BATCH_SIZE = 32
LR = 5e-5

# ---------- Attention Module ----------


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):  # lstm_out: [batch, seq_len, hidden]
        weights = self.attn(lstm_out)  # [batch, seq_len, 1]
        weights = torch.softmax(weights, dim=1)
        context = (weights * lstm_out).sum(dim=1)  # [batch, hidden]
        return context

# ---------- Custom Dataset ----------


class RacketDataset(Dataset):
    def __init__(self, info_df, data_dir):
        self.info_df = info_df
        self.data_dir = data_dir

    def __len__(self):
        return len(self.info_df)

    def __getitem__(self, idx):
        row = self.info_df.iloc[idx]
        uid = row['unique_id']
        file_path = os.path.join(self.data_dir, f"{uid}.txt")
        data = []

        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    values = list(map(float, line.strip().split()))
                    if len(values) == 6:
                        data.append(values)

        data_tensor = torch.tensor(data, dtype=torch.float32)

        # Normalize sensor data
        mean = data_tensor.mean(dim=0, keepdim=True)
        std = data_tensor.std(dim=0, keepdim=True) + 1e-6
        norm_data = (data_tensor - mean) / std

        # Statistical features
        stat_features = torch.cat([
            data_tensor.mean(dim=0),
            data_tensor.std(dim=0),
            data_tensor.min(dim=0).values,
            data_tensor.max(dim=0).values,
        ])  # shape: [6*4 = 24]

        # Frequency-domain (FFT) features
        fft = torch.fft.rfft(data_tensor, dim=0)  # shape: [N/2+1, 6]
        fft_energy = torch.mean(torch.abs(fft), dim=0)  # shape: [6]

        meta_feat = torch.cat([stat_features, fft_energy], dim=0)  # [30]
        seq_len = norm_data.shape[0]
        meta_seq = meta_feat.unsqueeze(0).repeat(seq_len, 1)  # [seq_len, 30]
        full_data = torch.cat([norm_data, meta_seq], dim=1)  # [seq_len, 6+30]

        labels = {
            'gender': row['gender'] - 1,
            'handed': row['hold racket handed'] - 1,
            'play_years': row['play years'],
            'level': row['level'] - 2,
        }

        return full_data, labels

# ---------- Evaluation ----------


def evaluate(model, dataloader):
    model.eval()
    all_gender_logits = []
    all_gender_labels = []

    all_handed_logits = []
    all_handed_labels = []

    all_years_logits = []
    all_years_labels = []

    all_level_logits = []
    all_level_labels = []

    with torch.no_grad():
        for X, gender, handed, years, level in dataloader:
            X = X.to(DEVICE)
            gender = gender.to(DEVICE)
            handed = handed.to(DEVICE)
            years = years.to(DEVICE)
            level = level.to(DEVICE)

            outputs = model(X)

            all_gender_logits.append(F.softmax(outputs['gender'], dim=1).cpu())
            all_gender_labels.append(gender.cpu())

            all_handed_logits.append(F.softmax(outputs['handed'], dim=1).cpu())
            all_handed_labels.append(handed.cpu())

            all_years_logits.append(
                F.softmax(outputs['play_years'], dim=1).cpu())
            all_years_labels.append(years.cpu())

            all_level_logits.append(F.softmax(outputs['level'], dim=1).cpu())
            all_level_labels.append(level.cpu())

    # Concatenate predictions and labels
    y_gender_prob = torch.cat(all_gender_logits).numpy()
    y_gender_true = torch.cat(all_gender_labels).numpy()
    y_handed_prob = torch.cat(all_handed_logits).numpy()
    y_handed_true = torch.cat(all_handed_labels).numpy()
    y_years_prob = torch.cat(all_years_logits).numpy()
    y_years_true = torch.cat(all_years_labels).numpy()
    y_level_prob = torch.cat(all_level_logits).numpy()
    y_level_true = torch.cat(all_level_labels).numpy()

    # AUC for each task
    auc_gender = roc_auc_score(y_gender_true, y_gender_prob[:, 1])
    auc_handed = roc_auc_score(y_handed_true, y_handed_prob[:, 1])
    auc_years = roc_auc_score(
        y_years_true, y_years_prob, multi_class='ovr', average='micro')
    auc_level = roc_auc_score(
        y_level_true, y_level_prob, multi_class='ovr', average='micro')

    final_score = 0.25 * (auc_gender + auc_handed + auc_years + auc_level)

    print(f"Gender AUC: {auc_gender:.4f}")
    print(f"Handed AUC: {auc_handed:.4f}")
    print(f"Play Years AUC: {auc_years:.4f}")
    print(f"Level AUC: {auc_level:.4f}")
    print(f"Final Score (Average): {final_score:.4f}")
    return final_score

# ---------- Collate Function ----------


def collate_fn(batch):
    sequences = [item[0] for item in batch]
    padded = pad_sequence(sequences, batch_first=True)
    genders = torch.tensor([item[1]['gender']
                           for item in batch], dtype=torch.long)
    handeds = torch.tensor([item[1]['handed']
                           for item in batch], dtype=torch.long)
    years = torch.tensor([item[1]['play_years']
                         for item in batch], dtype=torch.long)
    levels = torch.tensor([item[1]['level']
                          for item in batch], dtype=torch.long)
    return padded, genders, handeds, years, levels

# ---------- LSTM Classifier ----------


class LSTMClassifier(nn.Module):
    def __init__(self, input_size=36, hidden_size=128, num_layers=2, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = Attention(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc_gender = nn.Linear(hidden_size * 2, 2)
        self.fc_handed = nn.Linear(hidden_size * 2, 2)
        self.fc_years = nn.Linear(hidden_size * 2, 3)
        self.fc_level = nn.Linear(hidden_size * 2, 4)
        self.norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        context = self.dropout(context)
        context = self.norm(context)
        return {
            'gender': self.fc_gender(context),
            'handed': self.fc_handed(context),
            'play_years': self.fc_years(context),
            'level': self.fc_level(context),
        }


# ---------- Training with Stratified K-Fold ----------
NUM_FOLDS = 5
SAVE_MODEL_PATH = 'best_model.pth'


def train_kfold():
    df = pd.read_csv(TRAIN_INFO_PATH)
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    best_score = -1
    fold_scores = []

    # Combine gender and handedness for stratification
    df['stratify_label'] = df['gender'].astype(
        str) + '_' + df['hold racket handed'].astype(str)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['stratify_label'])):
        print(f"\nFold {fold+1}/{NUM_FOLDS}")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_dataset = RacketDataset(train_df, TRAIN_DATA_DIR)
        val_dataset = RacketDataset(val_df, TRAIN_DATA_DIR)

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        model = LSTMClassifier().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2)
        criterion = nn.CrossEntropyLoss()

        early_stop_round = 5
        no_improve_count = 0
        for epoch in range(1, EPOCHS + 1):
            model.train()
            losses = []

            for X, gender, handed, years, level in tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch}"):
                X = X.to(DEVICE)
                gender = gender.to(DEVICE)
                handed = handed.to(DEVICE)
                years = years.to(DEVICE)
                level = level.to(DEVICE)

                output = model(X)
                loss = (
                    criterion(output['gender'], gender) +
                    criterion(output['handed'], handed) +
                    2.0 * criterion(output['play_years'], years) +
                    1.5 * criterion(output['level'], level)
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            print(
                f"Fold {fold+1} Epoch {epoch} Avg Loss: {np.mean(losses):.4f}")

        val_score = evaluate(model, val_loader)
        scheduler.step(val_score)
        fold_scores.append(val_score)

        # Save model for each fold
        MODEL_DIR = "models"
        os.makedirs(MODEL_DIR, exist_ok=True)
        fold_model_path = os.path.join(MODEL_DIR, f"fold{fold}.pth")
        torch.save(model.state_dict(), fold_model_path)
        print(f"Saved model for Fold {fold+1} to {fold_model_path}")

        # Save best model overall
        if val_score > best_score:
            best_score = val_score
            no_improve_count = 0
            best_model_path = os.path.join(MODEL_DIR, "best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(
                f"Saved best model from Fold {fold+1} (score={val_score:.4f}) to {best_model_path}")
        else:
            no_improve_count += 1
            if no_improve_count >= early_stop_round:
                print(f"Early stopped at epoch {epoch}")
                break

    print("\nAll Fold Scores:", fold_scores)
    print(f"Average AUC Across Folds: {np.mean(fold_scores):.4f}")


if __name__ == "__main__":
    train_kfold()
