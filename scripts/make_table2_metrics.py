import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

PROJECT = Path(r"C:\CT-Dimensionality-Study")
DATASET = Path(r"C:\final_team_dataset_v2_3class")

sys.path.insert(0, str(PROJECT / "scripts"))

from dataset_25d import Nodule25DDataset
from model_25d import ResNet25D


# SETTINGS

TRAIN_CSV = DATASET / "metadata" / "supervised_train.csv"

MODEL_A = PROJECT / "outputs" / "checkpoints" / "best_25d_model_A.pt"
MODEL_C = PROJECT / "outputs" / "checkpoints" / "best_25d_model_C.pt"

BATCH_SIZE = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# LOAD TRAINING DATA

dataset = Nodule25DDataset(TRAIN_CSV)

print("Training samples:", len(dataset))

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# EVALUATE ONE MODEL

def evaluate_model(checkpoint_path, model_name):

    print()
    print(" " * 55)
    print(model_name)
    print("-" * 55)

    model = ResNet25D(num_classes=3)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    # Some checkpoints contain the complete training state.
    # Others may contain only the model weights.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_labels.extend(
                labels.numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)

    train_macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    print(f"Training Macro-F1 = {train_macro_f1:.4f}")

    return train_macro_f1


# MODEL A — NO AUGMENTATION

train_f1_A = evaluate_model(
    MODEL_A,
    "MODEL A — NO AUGMENTATION"
)


# MODEL C — MILD AUGMENTATION

train_f1_C = evaluate_model(
    MODEL_C,
    "MODEL C — MILD AUGMENTATION"
)


# VALIDATION RESULTS

val_f1_A = 0.5883
val_f1_C = 0.5857


# TRAIN–VALIDATION GAP

gap_A = train_f1_A - val_f1_A
gap_C = train_f1_C - val_f1_C


# FINAL TABLE 2 NUMBERS

print()
print(" " * 55)
print("TABLE 2 METRICS")
print("-" * 55)

print()
print("                         No Aug (A)     Mild (C)")
print("------------------------------------------------")
print(f"Test Macro-F1            0.6075         0.6537")
print(f"Best Val Macro-F1        {val_f1_A:.4f}         {val_f1_C:.4f}")
print(f"Train Macro-F1           {train_f1_A:.4f}         {train_f1_C:.4f}")
print(f"Train-Val Gap            {gap_A:+.4f}        {gap_C:+.4f}")
print("Best Epoch               9              12")


# SUMMARY

print()
print(" " * 55)
print("SUMMARY")
print("-" * 55)

print(f"Model A training F1 = {train_f1_A:.4f}")
print(f"Model A validation F1 = {val_f1_A:.4f}")
print(f"Model A train-val gap = {gap_A:+.4f}")

print()

print(f"Model C training F1 = {train_f1_C:.4f}")
print(f"Model C validation F1 = {val_f1_C:.4f}")
print(f"Model C train-val gap = {gap_C:+.4f}")

print()
print("No strong-augmentation model was trained.")
print("Therefore, no strong column is included in Table 2.")