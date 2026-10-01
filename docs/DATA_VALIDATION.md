# Data validation and quality control

Quality control is applied before the supervised CSVs are frozen. It protects
against wrong-series joins, malformed crops, missing paired masks, and patient
leakage.

## Cleaning checks

- DICOM headers are inventoried without loading pixel arrays for indexing.
- CT modality and CT SOP classes are checked before a series is selected.
- XML study/series/SOP references are matched to the DICOM inventory, with
  context errors and unmatched references reported.
- Slice geometry is checked and slices are ordered physically using image
  orientation and position metadata.
- Stored pixels are converted to Hounsfield units using DICOM rescale slope
  and intercept.
- Reader polygons are rasterized into reader-level masks. Missing slice
  references and out-of-bounds polygons are counted rather than hidden.
- Reader annotations are clustered within a CT series while preventing
  same-session duplicate inflation.
- Suspicious clusters, incomplete crop coverage, and missing sample/mask files
  are excluded from the supervised cohort.
- Consensus nodules are resampled to 1 mm isotropic spacing and cropped to
  `(104,72,80)` in `(Z,Y,X)` order.
- Fixed train/validation/test assignments are checked for patient leakage.
- The final V2 migration checks sample/mask IDs, array sizes, and hashes.

## Verified final V2 checks

The retained V2 validation report records:

- 7,385 samples and 7,385 masks;
- 27 technical exclusions: 23 suspicious clusters and 4 incomplete crops;
- zero missing sample files, missing mask files, array-size mismatches, array
  ID mismatches, and patient leakage;
- 2,654 supervised nodules and 5,133 SSL-training nodules.

The original Phase 1 report records 244,527 readable DICOM records, 1,318
readable XML files, 1,035 CT annotation XML files, 283 non-CT/CXR XML files,
1,018 CT series, 77,262 SOP references, 77,100 matches, 162 unmatched SOP
references, and zero failed patients. The 290 excluded-series count in that
report is a series-level count and should not be read as the XML-file count.

The Phase 2 source code exposes an `unmatched_roi_references` statistic. The
cleaning repository retained worker logs but not the final Phase 2 statistics
artifact; consequently the final aggregate value of 97 is recorded as a
pipeline-run statistic supplied for this project, not independently recovered
from a retained Phase 2 JSON report.

## Validation status

The source repository contains both an earlier superseded Phase 3 run (7,387
physical nodules) and the final run used for V2 (7,385 physical nodules). The
final run is the one documented here and agrees with `manifest_v2.json`,
`statistics_v2.json`, and the final metadata.
