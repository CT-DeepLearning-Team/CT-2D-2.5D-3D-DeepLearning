# External data

Raw and processed datasets are intentionally not committed here. Place them on
local storage and pass their paths through a local configuration file.

The final V2 dataset is expected to expose this structure:

```text
final_team_dataset_v2_3class/
├── samples/*.npy
├── masks/*.npy
├── metadata/
│   ├── all_nodules_v2.csv
│   ├── qc_exclusions.csv
│   ├── supervised_train.csv
│   ├── supervised_validation.csv
│   ├── supervised_test.csv
│   └── ssl_pretrain_train.csv
├── docs/
├── statistics_v2.json
└── manifest_v2.json
```

The CSV paths are relative to the external dataset root. No raw DICOM, XML,
NIfTI, mask, sample, or other dataset artifact belongs in `data/`.
