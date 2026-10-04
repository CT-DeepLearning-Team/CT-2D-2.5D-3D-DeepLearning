import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path


DATASET = Path(r"C:\final_team_dataset_v2_3class")


class Nodule25DDataset(Dataset):

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

        # Only use samples approved for supervised training
        self.df = self.df[
            self.df["supervised_eligible_v2"] == True
        ].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        sample_path = DATASET / row["sample_path"]
        mask_path = DATASET / row["mask_path"]

        volume = np.load(sample_path)
        mask = np.load(mask_path)

        # Find nodule position
        z, y, x = np.where(mask > 0)

        # Safety check: mask should not be empty
        if len(z) == 0:
            raise ValueError(f"Empty mask for sample: {sample_path}")

        # Central Z slice
        center_z = int(np.median(z))

        # Safety check: make sure five slices are available
        if center_z - 2 < 0 or center_z + 2 >= volume.shape[0]:
            raise ValueError(
                f"Cannot extract 5 slices for sample: {sample_path}"
            )

        # Five slices: z-2, z-1, z, z+1, z+2
        slice_indices = [
            center_z - 2,
            center_z - 1,
            center_z,
            center_z + 1,
            center_z + 2
        ]

        slices = volume[slice_indices]

        # Clip CT intensity values
        slices = np.clip(slices, -1000, 400)

        # Scale values to [0, 1]
        slices = (slices + 1000) / 1400

        # Convert to PyTorch tensor
        image = torch.tensor(slices, dtype=torch.float32)

        # Label: 0 = benign, 1 = indeterminate, 2 = malignant
        label = int(row["class_index_v2"])

        return image, label