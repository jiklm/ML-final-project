import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from tqdm import tqdm

# ---------- Configuration ----------
TEST_INFO_PATH = '39_Test_Dataset/test_info.csv'
TEST_DATA_DIR = '39_Test_Dataset/test_data'
MODEL_PATH = 'best_model.pth'
SUBMIT_CSV = 'submission.csv'
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "cpu")  # Use GPU if available


# ---------- Attention Mechanism ----------
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        # Compute attention scores [batch, seq_len, 1]
        weights = self.attn(lstm_out)
        weights = torch.softmax(weights, dim=1)  # Normalize weights
        # Weighted sum to get context vector [batch, hidden]
        context = (weights * lstm_out).sum(dim=1)
        return context


# ---------- Custom Dataset ----------
class RacketDataset(Dataset):
    def __init__(self, info_df, data_dir):
        self.info_df = info_df
        self.data_dir = data_dir

    def __len__(self):
        return len(self.info_df)

    def __getitem__(self, idx):
        uid = self.info_df.iloc[idx]['unique_id']
        file_path = os.path.join(self.data_dir, f"{uid}.txt")
        data = []

        # Load sensor time-series data
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    values = list(map(float, line.strip().split()))
                    if len(values) == 6:
                        data.append(values)

        data_tensor = torch.tensor(data, dtype=torch.float32)

        # Normalize the raw data
        mean = data_tensor.mean(dim=0, keepdim=True)
        std = data_tensor.std(dim=0, keepdim=True) + 1e-6
        norm_data = (data_tensor - mean) / std

        # Extract statistical features (mean, std, min, max) for each axis [6 axes * 4 stats = 24]
        stat_features = torch.cat([
            data_tensor.mean(dim=0),
            data_tensor.std(dim=0),
            data_tensor.min(dim=0).values,
            data_tensor.max(dim=0).values,
        ])  # Shape: [24]

        # Frequency domain features: FFT energy per axis [6]
        fft = torch.fft.rfft(data_tensor, dim=0)
        fft_energy = torch.mean(torch.abs(fft), dim=0)  # Shape: [6]

        # Combine all meta features
        meta_feat = torch.cat(
            [stat_features, fft_energy], dim=0)  # Shape: [30]
        seq_len = norm_data.shape[0]
        meta_seq = meta_feat.unsqueeze(0).repeat(
            seq_len, 1)  # Shape: [seq_len, 30]

        # Concatenate normalized data with meta features
        full_data = torch.cat([norm_data, meta_seq],
                              dim=1)  # Shape: [seq_len, 36]

        return full_data, uid


# ---------- Collate Function for Variable Length Sequences ----------
def collate_fn(batch):
    sequences = [item[0] for item in batch]
    uids = [item[1] for item in batch]
    # Pad sequences to same length
    padded = pad_sequence(sequences, batch_first=True)
    return padded, uids


# ---------- LSTM Classifier ----------
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=36, hidden_size=128, num_layers=2, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        # Attention on BiLSTM output
        self.attention = Attention(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)

        # Output heads for each label
        self.fc_gender = nn.Linear(hidden_size*2, 2)
        self.fc_handed = nn.Linear(hidden_size*2, 2)
        self.fc_years = nn.Linear(hidden_size*2, 3)
        self.fc_level = nn.Linear(hidden_size*2, 4)

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


# ---------- Inference and Submission Generation ----------
def predict_and_generate_submission():
    test_info = pd.read_csv(TEST_INFO_PATH)
    test_dataset = RacketDataset(test_info, TEST_DATA_DIR)
    test_loader = DataLoader(
        test_dataset, batch_size=32, collate_fn=collate_fn)

    NUM_FOLDS = 5
    models = []

    # Load each fold's trained model
    for fold in range(NUM_FOLDS):
        model = LSTMClassifier(input_size=36).to(DEVICE)
        model_path = os.path.join("models", f"fold{fold}.pth")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        models.append(model)

    # Optionally include the best overall model
    model = LSTMClassifier(input_size=36).to(DEVICE)
    model_path = os.path.join("models", f"best.pth")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    models.append(model)

    submission = []

    # Perform inference
    with torch.no_grad():
        for X, uids in tqdm(test_loader, desc="Predicting"):
            X = X.to(DEVICE)

            # Initialize predictions
            out_sum = {
                'gender': torch.zeros((X.size(0), 2)),
                'handed': torch.zeros((X.size(0), 2)),
                'play_years': torch.zeros((X.size(0), 3)),
                'level': torch.zeros((X.size(0), 4)),
            }

            # Aggregate predictions across models
            for model in models:
                out = model(X)
                for key in out:
                    out_sum[key] += torch.softmax(out[key], dim=1).cpu()

            # Average predictions
            for key in out_sum:
                out_sum[key] /= NUM_FOLDS

            # Prepare submission rows
            gender_probs = out_sum['gender'].numpy()
            handed_probs = out_sum['handed'].numpy()
            years_probs = out_sum['play_years'].numpy()
            level_probs = out_sum['level'].numpy()

            for i, uid in enumerate(uids):
                row = {
                    'unique_id': uid,
                    'gender': gender_probs[i][0],
                    'hold racket handed': handed_probs[i][0],
                    'play years_0': years_probs[i][0],
                    'play years_1': years_probs[i][1],
                    'play years_2': years_probs[i][2],
                    'level_2': level_probs[i][0],
                    'level_3': level_probs[i][1],
                    'level_4': level_probs[i][2],
                    'level_5': level_probs[i][3],
                }
                submission.append(row)

    # Save to CSV
    df = pd.DataFrame(submission)
    df.to_csv(SUBMIT_CSV, index=False, float_format="%.4f")
    print(f"Saved: {SUBMIT_CSV}")


# ---------- Main ----------
if __name__ == "__main__":
    predict_and_generate_submission()
