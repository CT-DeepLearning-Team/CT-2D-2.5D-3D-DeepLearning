# Label policy

The primary task is three-class pulmonary nodule malignancy-risk
classification. It uses the median of the available reader malignancy scores.
Individual LIDC-IDRI reader scores are integers from 1 to 5; half values such
as 2.5 and 3.5 arise when scores are aggregated across readers.

| Median malignancy | Operational class | Index |
|---|---|---:|
| 1.0–2.0 | benign | 0 |
| 2.5–3.0 | indeterminate | 1 |
| 3.5–5.0 | malignant | 2 |

The labels express radiologist malignancy risk and uncertainty. They are not
pathology-confirmed ground-truth cancer diagnoses.

## Binary sensitivity experiment

The supplementary binary experiment keeps benign nodules (1.0–2.0) as label
0 and malignant nodules (3.5–5.0) as label 1. Indeterminate nodules (2.5–3.0)
are excluded. It uses the existing fixed patient assignments and does not
create a new split. The three-class experiment remains the primary task.
