import os
import csv
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Audio parameters (defined before use)
SR = 22050         # sample rate (Hz)
N_FFT = 1024       # FFT window size
HOP_LEN = 512      # hop length (50% overlap)
N_MELS = 128       # number of mel frequency bins
SEGMENT_WIDTH = 128  # width of each segment in frames

# === Visualization Utilities ===

def plot_genre_distribution(labels, save_path='genre_distribution.png'):
    label_counts = pd.Series(labels).value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=label_counts.index, y=label_counts.values)
    plt.title('Genre Distribution')
    plt.ylabel('Count')
    plt.xlabel('Genre')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_sample_spectrograms(spectrograms, genres, save_path='sample_spectrograms.png'):
    plt.figure(figsize=(16, 10))
    for i, (spec, genre) in enumerate(zip(spectrograms[:10], genres[:10])):
        plt.subplot(2, 5, i + 1)
        plt.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
        plt.title(genre)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, labels, save_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_training_curves(train_losses_list, val_losses_list, train_accs_list, val_accs_list, save_path='training_curves_all.png'):
    plt.figure(figsize=(12, 5))
    # Loss curves
    plt.subplot(1, 2, 1)
    for i, (train_losses, val_losses) in enumerate(zip(train_losses_list, val_losses_list)):
        plt.plot(train_losses, label=f'Fold {i+1} Train')
        plt.plot(val_losses, label=f'Fold {i+1} Val', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend(fontsize='small', loc='upper right')
    # Accuracy curves
    plt.subplot(1, 2, 2)
    for i, (train_accs, val_accs) in enumerate(zip(train_accs_list, val_accs_list)):
        plt.plot(train_accs, label=f'Fold {i+1} Train')
        plt.plot(val_accs, label=f'Fold {i+1} Val', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend(fontsize='small', loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Data Processing: Load audio files, compute mel spectrograms, and slice into 128x128 segments
# Specify the path to GTZAN dataset (with genre subdirectories)
dataset_path = "datasets/Data/genres_original"
genres = sorted([d for d in os.listdir(dataset_path)
                if os.path.isdir(os.path.join(dataset_path, d))])

# === EDA Visualization ===
# Plot genre distribution
all_genres = []
for genre in genres:
    genre_path = os.path.join(dataset_path, genre)
    all_genres.extend([genre] * len(os.listdir(genre_path)))
plot_genre_distribution(all_genres)

# Plot 10 example mel spectrograms
example_specs = []
example_labels = []
for genre in genres:
    genre_dir = os.path.join(dataset_path, genre)
    file_list = [f for f in os.listdir(genre_dir) if f.endswith('.wav')][:1]
    for f in file_list:
        y, sr = librosa.load(os.path.join(genre_dir, f), sr=SR)
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LEN, n_mels=N_MELS)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        example_specs.append(mel_spec_db)
        example_labels.append(genre)
plot_sample_spectrograms(example_specs, example_labels)

all_data = []
all_labels = []
track_ids = []  # Stores track ID for each segment (for fold split)
for label_idx, genre in enumerate(genres):
    genre_dir = os.path.join(dataset_path, genre)
    for fname in sorted(os.listdir(genre_dir)):
        if not fname.lower().endswith('.wav'):
            continue
        # Retrieve track ID from filename (e.g., "blues.00000.wav" -> track_id 0)
        name_noext = os.path.splitext(fname)[0]    # "blues.00000"
        parts = name_noext.split(".")
        track_id = int(parts[1]) if len(parts) > 1 else -1

        # Load audio
        file_path = os.path.join(genre_dir, fname)
        y, sr = librosa.load(file_path, sr=SR, mono=True)
        # Compute mel spectrogram (power spectrogram)
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=SR,
            n_fft=N_FFT,
            hop_length=HOP_LEN,
            n_mels=N_MELS,
            power=2.0
        )
        # Convert to logarithmic scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        # Check that the spectrogram has at least 1280 frames (pad or trim to 1280)
        if mel_spec_db.shape[1] < SEGMENT_WIDTH * 10:
            pad_width = SEGMENT_WIDTH * 10 - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=mel_spec_db.min())
        elif mel_spec_db.shape[1] > SEGMENT_WIDTH * 10:
            mel_spec_db = mel_spec_db[:, :SEGMENT_WIDTH * 10]
        # Split into 10 segments of size 128x128
        for i in range(10):
            start = i * SEGMENT_WIDTH
            end = start + SEGMENT_WIDTH
            segment = mel_spec_db[:, start:end]
            all_data.append(segment.astype(np.float32))
            all_labels.append(label_idx)
            track_ids.append(track_id)

# Define PyTorch Dataset

class MelSpectrogramDataset(Dataset):
    def __init__(self, data_list, label_list):
        self.data_list = data_list
        self.label_list = label_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        X = torch.tensor(self.data_list[idx], dtype=torch.float32)
        X = X.unsqueeze(0)  # Add channel dimension -> (1, 128, 128)
        y = self.label_list[idx]
        return X, y

dataset = MelSpectrogramDataset(all_data, all_labels)

# Model Definition: CNN-TE architecture (Convolutional module + Transformer Encoder + classification head)

class CNNTEModel(nn.Module):
    def __init__(self, num_classes=10, seq_len=64, d_model=256, n_heads=16, dim_feedforward=512):
        super(CNNTEModel, self).__init__()
        self.conv_layers = nn.Sequential(
            # Conv layer 1, 32 channels
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Conv layer 2, 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Conv layer 3, 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Conv layer 4, 256 channels
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        pos = torch.arange(seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)  # shape (1, seq_len, d_model)
        self.register_buffer('pos_enc', pe)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        out = self.conv_layers(x)            # (batch, 256, 8, 8)
        batch, channels, height, width = out.size()
        out = out.view(batch, channels, height * width)  # (batch, 256, 64)
        out = out.permute(0, 2, 1)          # (batch, 64, 256)
        out = out + self.pos_enc            # (batch, 64, 256)
        out = self.transformer_encoder(out)  # (batch, 64, 256)
        out = out.mean(dim=1)              # (batch, 256)
        out = self.fc(out)                 # (batch, num_classes)
        return out

# Training and Evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(genres)  # should be 10 for GTZAN
kf_accum_preds = []
kf_accum_labels = []

print("Data loaded, starting training...", flush=True)

# Prepare to accumulate metrics across all folds
fold_train_losses_list = []
fold_val_losses_list = []
fold_train_accs_list = []
fold_val_accs_list = []

# 10-fold cross-validation
K = 10
max_epochs = 100
for fold in range(K):
    print(f"\n=== Fold {fold+1}/{K} ===", flush=True)
    test_track_ids = list(range(fold * 10, fold * 10 + 10))        # 10 track IDs for testing
    val_fold = (fold + 1) % K
    val_track_ids = list(range(val_fold * 10, val_fold * 10 + 10))   # 10 track IDs for validation
    test_indices = [idx for idx, tid in enumerate(track_ids) if tid in test_track_ids]
    val_indices = [idx for idx, tid in enumerate(track_ids) if tid in val_track_ids]
    train_indices = [idx for idx, tid in enumerate(track_ids) if (tid not in test_track_ids and tid not in val_track_ids)]
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_indices), batch_size=64, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_indices), batch_size=64, shuffle=False)
    test_loader = DataLoader(torch.utils.data.Subset(dataset, test_indices), batch_size=64, shuffle=False)

    model = CNNTEModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    best_val_loss = float("inf")
    best_model_wts = None
    patience = 10
    patience_counter = 0

    # Metrics lists for the current fold
    fold_train_losses = []
    fold_val_losses = []
    fold_train_accs = []
    fold_val_accs = []

    for epoch in range(1, max_epochs+1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_total += y_batch.size(0)
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                val_correct += (preds == y_val).sum().item()
                val_total += y_val.size(0)
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        # Save metrics for the current epoch in the current fold
        fold_train_losses.append(train_loss)
        fold_val_losses.append(val_loss)
        fold_train_accs.append(train_acc)
        fold_val_accs.append(val_acc)

        epoch_duration = time.time() - epoch_start
        remaining_epochs = max_epochs - epoch
        estimated_remaining_time = remaining_epochs * epoch_duration

        print(f"Fold {fold+1}/{K} | Epoch {epoch}/{max_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f} | Epoch Time: {epoch_duration:.2f}s | Est. remaining: {estimated_remaining_time/60:.2f} min", flush=True)

        # Additional comment: Early stopping is used to prevent overfitting.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    # Save metrics for the current fold for the final plot
    fold_train_losses_list.append(fold_train_losses)
    fold_val_losses_list.append(fold_val_losses)
    fold_train_accs_list.append(fold_train_accs)
    fold_val_accs_list.append(fold_val_accs)

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
        model.to(device)

    model.eval()
    fold_preds = []
    fold_true = []
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            outputs = model(X_test)
            _, predicted = torch.max(outputs, 1)
            fold_preds.extend(predicted.cpu().numpy().tolist())
            fold_true.extend(y_test.cpu().numpy().tolist())
    kf_accum_preds.extend(fold_preds)
    kf_accum_labels.extend(fold_true)

# Plot overall training curves for all folds
plot_training_curves(fold_train_losses_list, fold_val_losses_list, fold_train_accs_list, fold_val_accs_list, save_path="training_curves_all.png")

plot_confusion_matrix(kf_accum_labels, kf_accum_preds, genres)

conf_mat = confusion_matrix(kf_accum_labels, kf_accum_preds)
overall_acc = accuracy_score(kf_accum_labels, kf_accum_preds)
precision = precision_score(kf_accum_labels, kf_accum_preds, average='macro')
recall = recall_score(kf_accum_labels, kf_accum_preds, average='macro')
f1 = f1_score(kf_accum_labels, kf_accum_preds, average='macro')

cm_percent = conf_mat.astype('float') / conf_mat.sum(axis=1, keepdims=True) * 100

print("\nConfusion Matrix (in percentages):")
print(np.around(cm_percent, decimals=2))
print(f"\nOverall Accuracy: {overall_acc * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1-score: {f1 * 100:.2f}%")

with open('final_metrics.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Metric', 'Value'])
    writer.writerow(['Accuracy', overall_acc])
    writer.writerow(['Precision', precision])
    writer.writerow(['Recall', recall])
    writer.writerow(['F1 Score', f1])