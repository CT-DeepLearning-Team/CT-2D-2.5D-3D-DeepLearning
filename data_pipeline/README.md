# Cleaning pipeline source

This directory contains the reproducible Python source and a sanitized example
configuration for the LIDC-IDRI cleaning and final V2 metadata workflow.

The raw DICOM/XML input, processed volumes, manifests, masks, `.npy` arrays,
logs, and reports are intentionally kept outside this repository. Copy
`config/default.example.json` to a local configuration file and replace the
placeholder paths with local paths before running the scripts.

From this directory, the main stages are:

```text
python -m scripts.run_phase1 --config config/default.example.json
python -m scripts.run_phase2 --config config/default.example.json
python -m scripts.run_phase3 --config config/default.example.json
python -m scripts.build_v2_dataset --source-root <phase3-output> --output-root <v2-output>
python -m scripts.validate_v2_migration --source <source-dataset> --destination <v2-output>
```

The V2 builder copies already-created arrays and derives metadata; it does not
reconstruct, resample, or edit the source arrays.
