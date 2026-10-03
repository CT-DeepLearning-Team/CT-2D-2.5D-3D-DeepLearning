import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from pathlib import Path
import numpy as np
import sys

sys.path.append(str(Path(__file__).parent))

from dataset_25d import Nodule25DDataset
from model_25d import ResNet25D


# Dataset paths
train_csv = r"C:\final_team_dataset_v2_3class\metadata\supervised_train.csv"
val_csv = r"C:\final_team_dataset_v2_3class\metadata\supervised_validation.csv"

checkpoint_dir = Path(r"C:\CT-Dimensionality-Study\outputs\checkpoints")
curve_dir = Path(r"C:\CT-Dimensionality-Study\outputs\curves")

checkpoint_dir.mkdir(parents=True, exist_ok=True)
curve_dir.mkdir(parents=True, exist_ok=True)


# Training settings
batch_size = 8
lr = 0.001
weight_decay = 0.0001
epochs = 20
patience = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# Load dataset
train_data = Nodule25DDataset(train_csv)
val_data = Nodule25DDataset(val_csv)

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)


# Create model
model = ResNet25D(num_classes=3).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay
)


best_f1 = 0
best_epoch = 0
wait = 0

train_losses = []
val_losses = []
val_f1s = []


print("Training samples:", len(train_data))
print("Validation samples:", len(val_data))
print()


# Training
for epoch in range(epochs):

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    train_loss = total_loss / len(train_data)


    # Validation
    model.eval()

    total_val_loss = 0
    true_labels = []
    predictions = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_val_loss += loss.item() * images.size(0)

            pred = torch.argmax(outputs, dim=1)

            true_labels.extend(labels.cpu().numpy())
            predictions.extend(pred.cpu().numpy())

    val_loss = total_val_loss / len(val_data)

    val_f1 = f1_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_f1s.append(val_f1)


    print(
        "Epoch", epoch + 1,
        "Train Loss:", round(train_loss, 4),
        "Val Loss:", round(val_loss, 4),
        "Val Macro-F1:", round(val_f1, 4)
    )


    # Save best model
    if val_f1 > best_f1:

        best_f1 = val_f1
        best_epoch = epoch + 1
        wait = 0

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": val_f1
            },
            checkpoint_dir / "best_25d_model_A.pt"
        )

        print("Best model saved.")

    else:

        wait += 1
        print("No improvement:", wait, "/", patience)


    # Early stopping
    if wait >= patience:

        print("Early stopping.")
        break


# Save training history
np.savez(
    curve_dir / "model_A_history.npz",
    train_loss=np.array(train_losses),
    val_loss=np.array(val_losses),
    val_macro_f1=np.array(val_f1s)
)


# Final result
print()
print("Model A finished.")
print("Best epoch:", best_epoch)
print("Best validation Macro-F1:", round(best_f1, 4))