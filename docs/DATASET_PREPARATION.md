# Dataset preparation

This document describes the external LIDC-IDRI dataset used by the team and
why the cleaning pipeline is required before training. The existing
[`docs/dataset.md`](dataset.md) remains the team's original dataset overview.

LIDC-IDRI contains thoracic CT studies, pulmonary nodule XML annotations, and
reader malignancy-risk assessments. It is not a ready-to-train table: one
patient can have several studies and nodules, annotations reference DICOM
instances by UID, and multiple readers may describe the same nodule. The
cleaning workflow resolves those relationships, creates aligned crops and
masks, and freezes leakage-safe patient splits.

The final V2 dataset contains 7,385 consensus nodules from 990 patients,
2,672 rated and 4,713 unrated. The supervised cohort has 2,654 rated nodules:
873 benign, 1,226 indeterminate, and 555 malignant. The 5,133-row SSL pool is
restricted to technically valid training-patient nodules; labels are ignored
during SSL.

See [LABEL_POLICY.md](LABEL_POLICY.md), [DATA_SPLITS.md](DATA_SPLITS.md), and
[DATA_VALIDATION.md](DATA_VALIDATION.md) for the operational contracts.
