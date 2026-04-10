import os
import numpy as np

# -------- PATH SETUP --------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

FEATURE_DIR = os.path.join(BASE_DIR, "outputs", "features")
SEQUENCE_DIR = os.path.join(BASE_DIR, "outputs", "sequences")

os.makedirs(SEQUENCE_DIR, exist_ok=True)

SEQ_LEN = 16      # number of frames per sequence
STRIDE = 8        # overlap

def load_sequences(class_name, label):
    class_dir = os.path.join(FEATURE_DIR, class_name)
    files = sorted(os.listdir(class_dir))
    X, y = [], []
    features = []
    for f in files:
        if f.endswith("_features.npy"):
            features.append(np.load(os.path.join(class_dir, f)))
    features = np.array(features)
    for i in range(0, len(features) - SEQ_LEN + 1, STRIDE):
        seq = features[i:i + SEQ_LEN]
        X.append(seq)
        y.append(label)
    return X, y

if __name__ == "__main__":
    print("🚀 Creating sequences...")

    X_acc, y_acc = load_sequences("accident", 1)
    X_norm, y_norm = load_sequences("normal", 0)

    X = np.array(X_acc + X_norm)
    y = np.array(y_acc + y_norm)

    np.save(os.path.join(SEQUENCE_DIR, "X.npy"), X)
    np.save(os.path.join(SEQUENCE_DIR, "y.npy"), y)

    print(f"✅ Sequences created")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
