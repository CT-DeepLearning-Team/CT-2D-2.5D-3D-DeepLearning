import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path


DATASET = Path(r"C:\final_team_dataset_v2_3class")


class Nodule25DBinaryDataset(Dataset):

    def __init__(self, csv_file):

        self.df = pd.read_csv(csv_file)

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        sample_path = DATASET / row["sample_path"]
        mask_path = DATASET / row["mask_path"]

        volume = np.load(sample_path)
        mask = np.load(mask_path)

        # Find nodule voxels
        z, y, x = np.where(mask > 0)

        if len(z) == 0:
            raise ValueError(
                f"Empty mask for sample: {sample_path}"
            )

        # Center slice
        center_z = int(np.median(z))

        # Five neighboring slices
        if (
            center_z - 2 < 0
            or center_z + 2 >= volume.shape[0]
        ):
            raise ValueError(
                f"Cannot extract 5 slices for sample: {sample_path}"
            )

        slice_indices = [
            center_z - 2,
            center_z - 1,
            center_z,
            center_z + 1,
            center_z + 2
        ]

        slices = volume[slice_indices]

        # HU clipping
        slices = np.clip(
            slices,
            -1000,
            400
        )

        # Normalize to [0, 1]
        slices = (
            slices + 1000
        ) / 1400

        image = torch.tensor(
            slices,
            dtype=torch.float32
        )

        # Binary label
        label = int(row["binary_class_index"])

        return image, label