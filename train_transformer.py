import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------- PATH SETUP --------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

SEQ_DIR = os.path.join(BASE_DIR, "outputs", "sequences")

# -------- LOAD DATA --------
X = np.load(os.path.join(SEQ_DIR, "X.npy"))   # (N, 16, 4)
y = np.load(os.path.join(SEQ_DIR, "y.npy"))   # (N,)

# -------- FEATURE NORMALIZATION (CRITICAL) --------
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

# -------- TRAIN / TEST SPLIT --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------- DATASET --------
class AccidentDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(
    AccidentDataset(X_train, y_train),
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    AccidentDataset(X_test, y_test),
    batch_size=32,
    shuffle=False
)

# -------- TRANSFORMER MODEL --------
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=4, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global Average Pooling
        return self.classifier(x).squeeze()

# -------- DEVICE --------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = TransformerClassifier().to(device)

# -------- CLASS WEIGHTED LOSS (VERY IMPORTANT) --------
pos_weight = torch.tensor(
    (len(y_train) - y_train.sum()) / y_train.sum(),
    device=device
)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -------- TRAINING --------
EPOCHS = 20

print("🚀 Training Transformer...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)

        optimizer.zero_grad()
        outputs = model(Xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f}")

# -------- EVALUATION --------
model.eval()
y_true, y_pred = [], []

THRESHOLD = 0.35   # important for accident detection

with torch.no_grad():
    for Xb, yb in test_loader:
        Xb = Xb.to(device)
        outputs = torch.sigmoid(model(Xb))
        preds = (outputs.cpu().numpy() > THRESHOLD).astype(int)

        y_pred.extend(preds)
        y_true.extend(yb.numpy())

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("\n📊 Evaluation Results")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-score  : {f1:.4f}")

torch.save(model.state_dict(), "transformer_model.pth")
print("Model saved successfully")
