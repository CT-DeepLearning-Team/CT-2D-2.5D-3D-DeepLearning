import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import f1_score
import numpy as np

from dataset_25d_binary import Nodule25DBinaryDataset
from model_25d import ResNet25D
from augment_25d import augment_25d


# Settings

TRAIN_CSV = r"C:\CT-Dimensionality-Study\team_package\binary_train.csv"
VAL_CSV = r"C:\CT-Dimensionality-Study\team_package\binary_validation.csv"

BATCH_SIZE = 8
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

EPOCHS = 20
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

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CURVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Datasets

train_dataset = Nodule25DBinaryDataset(
    TRAIN_CSV
)

val_dataset = Nodule25DBinaryDataset(
    VAL_CSV
)


# Data loaders

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


# Model

model = ResNet25D(
    num_classes=2
)

model = model.to(DEVICE)


# Loss and optimizer

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# Training history

train_losses = []
val_losses = []
val_macro_f1s = []


best_val_macro_f1 = 0.0
best_epoch = 0
patience_counter = 0


# Training

print("2.5D Binary Model - Mild Augmentation")
print("-" * 50)

print("Device:", DEVICE)
print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)
print()


for epoch in range(EPOCHS):

    # Training

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        # Same augmentation as Model C
        augmented_images = torch.stack(
            [
                augment_25d(image)
                for image in images
            ]
        )

        augmented_images = augmented_images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(
            augmented_images
        )

        loss = criterion(
            outputs,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()


    train_loss = (
        running_loss /
        len(train_loader)
    )


    # Validation

    model.eval()

    val_running_loss = 0.0

    all_labels = []
    all_predictions = []


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_running_loss += loss.item()

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


    val_loss = (
        val_running_loss /
        len(val_loader)
    )


    val_macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro"
    )


    # Save history

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_macro_f1s.append(val_macro_f1)


    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Macro-F1: {val_macro_f1:.4f}"
    )


    # Save best model

    if val_macro_f1 > best_val_macro_f1:

        best_val_macro_f1 = val_macro_f1
        best_epoch = epoch + 1

        patience_counter = 0

        checkpoint_path = (
            CHECKPOINT_DIR /
            "best_25d_binary_model.pt"
        )

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "best_epoch":
                    best_epoch,

                "best_val_macro_f1":
                    best_val_macro_f1
            },
            checkpoint_path
        )

        print("  Best model saved.")

    else:

        patience_counter += 1


    # Early stopping

    if patience_counter >= PATIENCE:

        print("Early stopping.")

        break


# Save training history

np.savez(
    CURVE_DIR / "binary_model_history.npz",
    train_loss=np.array(train_losses),
    val_loss=np.array(val_losses),
    val_macro_f1=np.array(val_macro_f1s)
)


print()
print("Training finished.")
print("Best epoch:", best_epoch)
print(
    "Best validation Macro-F1:",
    round(best_val_macro_f1, 4)
)
print(
    "Checkpoint:",
    CHECKPOINT_DIR /
    "best_25d_binary_model.pt"
)