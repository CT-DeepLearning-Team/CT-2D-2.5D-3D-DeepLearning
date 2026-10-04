import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import f1_score

from dataset_25d import Nodule25DDataset
from moco_model import MoCo25D


# Settings

TRAIN_CSV = r"C:\final_team_dataset_v2_3class\metadata\supervised_train.csv"
VAL_CSV = r"C:\final_team_dataset_v2_3class\metadata\supervised_validation.csv"

MOCO_CHECKPOINT = r"C:\CT-Dimensionality-Study\outputs\checkpoints\moco_25d_pretrained.pt"

BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
PATIENCE = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_DIR = Path(
    r"C:\CT-Dimensionality-Study\outputs\checkpoints"
)

CURVE_DIR = Path(
    r"C:\CT-Dimensionality-Study\outputs\curves"
)

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CURVE_DIR.mkdir(parents=True, exist_ok=True)


# Dataset

train_dataset = Nodule25DDataset(TRAIN_CSV)
val_dataset = Nodule25DDataset(VAL_CSV)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# Load MoCo model

moco_checkpoint = torch.load(
    MOCO_CHECKPOINT,
    map_location=DEVICE
)

model = MoCo25D(
    feature_dim=128,
    queue_size=512,
    momentum=0.999,
    temperature=0.07
)

model.load_state_dict(
    moco_checkpoint["model_state_dict"]
)

# Use the pretrained query encoder
classifier = model.encoder_q

# Change 128-dimensional projection output
# into 3-class classification output
classifier.fc = nn.Linear(
    classifier.fc.in_features,
    3
)

classifier = classifier.to(DEVICE)


# Training setup

optimizer = torch.optim.AdamW(
    classifier.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

criterion = nn.CrossEntropyLoss()

best_val_f1 = -1
best_epoch = 0
patience_counter = 0

train_losses = []
val_losses = []
val_f1_history = []


# Validation function

def evaluate(model, loader):

    model.eval()

    running_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    avg_loss = running_loss / len(loader)

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro"
    )

    return avg_loss, macro_f1


# Training

print("2.5D Model B - MoCo Fine-tuning")
print("-" * 50)

print("Device:", DEVICE)
print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)
print()

for epoch in range(EPOCHS):

    classifier.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = classifier(images)

        loss = criterion(
            outputs,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    train_loss = (
        running_loss / len(train_loader)
    )

    val_loss, val_f1 = evaluate(
        classifier,
        val_loader
    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_f1_history.append(val_f1)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Macro-F1: {val_f1:.4f}"
    )

    # Save best model
    if val_f1 > best_val_f1:

        best_val_f1 = val_f1
        best_epoch = epoch + 1
        patience_counter = 0

        torch.save(
            {
                "model_state_dict": classifier.state_dict(),
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_val_f1
            },
            CHECKPOINT_DIR / "best_25d_model_B.pt"
        )

        print("  Best model saved.")

    else:

        patience_counter += 1

        if patience_counter >= PATIENCE:

            print("Early stopping.")

            break


# Save training history

import numpy as np

np.savez(
    CURVE_DIR / "model_B_history.npz",
    train_loss=np.array(train_losses),
    val_loss=np.array(val_losses),
    val_macro_f1=np.array(val_f1_history)
)


print()
print("Training finished.")
print("Best epoch:", best_epoch)
print("Best validation Macro-F1:", round(best_val_f1, 4))
print(
    "Checkpoint:",
    CHECKPOINT_DIR / "best_25d_model_B.pt"
)