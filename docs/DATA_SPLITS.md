# Data splits

Splitting is performed at patient level because one patient can contribute
multiple nodules. Putting nodules from the same patient in different splits
would let evaluation examples share patient-specific acquisition and anatomy
with training examples, inflating performance.

| Split | Patients | Nodules | Benign | Indeterminate | Malignant |
|---|---:|---:|---:|---:|---:|
| train | 602 | 1,876 | 622 | 876 | 378 |
| validation | 134 | 382 | 134 | 163 | 85 |
| test | 134 | 396 | 117 | 187 | 92 |

The splits are fixed in the supplied metadata and must not be recreated by
individual model implementations. Patient overlap is zero. The SSL pool has
5,133 nodules from training patients only (692 patients); validation and test
patients never enter MoCo pretraining.
