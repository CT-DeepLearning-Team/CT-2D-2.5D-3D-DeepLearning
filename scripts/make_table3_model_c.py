import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score

from dataset_25d import Nodule25DDataset
from model_25d import ResNet25D


# Settings

TEST_CSV = r"C:\final_team_dataset_v2_3class\metadata\supervised_test.csv"
CHECKPOINT = r".\outputs\checkpoints\best_25d_model_C.pt"

BATCH_SIZE = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# Load test dataset

dataset = Nodule25DDataset(TEST_CSV)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Test samples:", len(dataset))


# Load Model C

model = ResNet25D(num_classes=3)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()


# Run inference

all_labels = []
all_predictions = []
all_probabilities = []


with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        predictions = torch.argmax(probabilities, dim=1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())


y_true = np.array(all_labels)
y_pred = np.array(all_predictions)
y_prob = np.array(all_probabilities)


# Overall test result

overall_accuracy = accuracy_score(y_true, y_pred)

overall_auc = roc_auc_score(
    y_true,
    y_prob,
    multi_class="ovr",
    average="macro"
)

print()
print(" " * 50)
print("OVERALL MODEL C TEST RESULT")
print("-" * 50)

print("N =", len(y_true))
print("Accuracy =", round(overall_accuracy, 4))
print("ROC-AUC =", round(overall_auc, 4))


# Load metadata for subgroup definitions

df = pd.read_csv(TEST_CSV)

df = df[
    df["supervised_eligible_v2"] == True
].reset_index(drop=True)


# Safety check
if len(df) != len(y_true):
    raise ValueError(
        f"Metadata/test mismatch: {len(df)} metadata rows vs "
        f"{len(y_true)} predictions"
    )


# Define "confident labels"

confident = (
    (df["midpoint_median_v2"] == False)
    & (df["strict_3_reader_eligible"] == True)
    & (df["disagreement_flag"] == False)
)

everything_else = ~confident


# Function for subgroup metrics

def evaluate_group(name, mask):

    indices = np.where(mask)[0]

    group_true = y_true[indices]
    group_pred = y_pred[indices]
    group_prob = y_prob[indices]

    accuracy = accuracy_score(
        group_true,
        group_pred
    )

    print()
    print(name)
    print("-" * 40)
    print("N =", len(indices))
    print("Class counts =", np.bincount(group_true, minlength=3))
    print("Accuracy =", round(accuracy, 4))

    # AUC requires all three classes to be present
    if len(np.unique(group_true)) == 3:

        auc = roc_auc_score(
            group_true,
            group_prob,
            multi_class="ovr",
            average="macro"
        )

        print("ROC-AUC =", round(auc, 4))

    else:
        print("ROC-AUC = undefined (not all 3 classes present)")


# Table 3 results

print()
print(" " * 50)
print("TABLE 3 SUBGROUP RESULTS")
print("-" * 50)

evaluate_group(
    "Confident labels",
    confident.to_numpy()
)

evaluate_group(
    "Everything else",
    everything_else.to_numpy()
)