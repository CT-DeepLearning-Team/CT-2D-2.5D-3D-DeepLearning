# 2D Analysis and Conclusion — Bahodir

Written narrative to accompany the generated numbers in `REPORT.md`. Covers the
points requested in `README_TEAM_EXPERIMENTS.txt`.

## Headline numbers (test, mean ± std over seeds 0/1/2)

| model | macro F1 | accuracy | balanced acc | ROC-AUC |
|---|---|---|---|---|
| 3-class supervised (A) | 0.5908 ± 0.0241 | 0.5850 ± 0.0320 | 0.5920 ± 0.0158 | 0.7511 ± 0.0067 |
| 3-class MoCo v2 (B) | 0.5803 ± 0.0261 | 0.5732 ± 0.0292 | 0.5774 ± 0.0264 | 0.7515 ± 0.0174 |
| **binary supervised** | **0.8323 ± 0.0218** | 0.8357 ± 0.0215 | 0.8304 ± 0.0213 | 0.8790 ± 0.0171 |

Majority-class baselines: 3-class macro F1 0.2138 / accuracy 0.4722;
binary macro F1 0.3372 / accuracy 0.5598.

## Was supervised or MoCo better? Did SSL help?

**Supervised (Model A) was marginally better, and SSL did not help for 2D.**

MoCo pretraining changed test macro-F1 by **−0.0105** (0.5803 vs 0.5908). That
difference is smaller than the seed-to-seed spread (±0.024 and ±0.026) and far
inside the bootstrap 95% CIs, which span roughly ±0.05 on a 396-nodule test set.
The honest statement is therefore **no measurable effect**, not "SSL hurt".
Validation agrees: 0.5767 vs 0.5823.

Two observations worth reporting rather than hiding:

- ROC-AUC was **identical** (0.7515 vs 0.7511). The ranking quality of the
  model did not change at all; only the argmax decisions moved slightly.
- The pretext task itself trained well — InfoNCE loss 7.482 → 4.138 and
  instance-discrimination accuracy 0.863 over 200 epochs. So the encoder did
  learn to discriminate instances; that skill simply did not transfer into
  malignancy-risk separation.

The most likely reason is a mismatch between the pretext task and the target.
MoCo learns to tell one nodule crop from another, and the cues that make a crop
identifiable — surrounding vessels, pleura, overall lung texture, crop framing —
are largely not the cues that indicate malignancy risk. On a single 72×80 slice
there is also very little for the augmentations to work with compared with a
full volume. I would expect 3D to have more to gain here, which makes Zaineb's
Model B the more informative one.

## Which class was easiest and hardest?

**Malignant was easiest, benign was hardest** — which is the opposite of what
the class counts would suggest, since benign has 622 training examples and
malignant only 378.

3-class Model A per-class test F1:

| class | F1 | recall |
|---|---|---|
| benign | 0.5158 | 0.5499 |
| indeterminate | 0.5877 | 0.5740 |
| malignant | **0.6689** | 0.6522 |

Malignant nodules are visually more distinctive (larger, spiculated, irregular),
so the model finds them despite having the fewest examples. Benign nodules are
mostly small and smooth and are constantly mistaken for indeterminate.

## What does the confusion matrix show?

Pooled over 3 seeds, 3-class Model A (rows = true):

```
                   benign  indeterminate  malignant
true benign           193            140         18
true indeterminate    175            322         64
true malignant         31             65        180
```

This is the single most informative result in my part. The errors are not spread
evenly:

- **benign ↔ indeterminate: 315 errors** (140 + 175)
- indeterminate ↔ malignant: 129 errors (64 + 65)
- **benign ↔ malignant: only 49 errors** (18 + 31)

So the model separates the two clinically meaningful extremes well and fails
almost entirely on the boundary between benign and indeterminate. That is
expected once you recall how *indeterminate* is defined: it is the median of
disagreeing radiologist scores (2.5–3.0), i.e. the class exists precisely where
human readers could not agree. We are asking the model to reproduce a boundary
that has no consistent visual definition.

The binary confusion matrix confirms the extremes are genuinely separable:

```
               benign  malignant
true benign       307         44
true malignant     59        217
```

## Was overfitting observed?

**Yes, severely, and it is the main limitation of the 2D result.**

At the selected epoch the train−validation macro-F1 gap was **+0.386** for
Model A and **+0.420** for Model B; by the last epoch run it reached +0.456.
Before regularisation it was worse still: an early unregularised run reached
train macro-F1 0.93 by epoch 3 while validation sat near 0.50.

This is a capacity/data mismatch, not a bug: ResNet-18 has 11.17M parameters and
the 3-class training split has 1876 nodules. Early stopping selected epochs
11–19 out of 80, so the useful training window is about 15 epochs.

What I used against it (all inside the agreed protocol): dropout 0.5, weight
decay 0.05, strong training-only augmentation, label smoothing 0.10,
inverse-frequency class weights, and early stopping on validation macro-F1. A
validation-only sweep of 8 configurations showed these matter a lot — validation
macro-F1 moved from 0.5478 (light regularisation, lr 3e-4) to 0.6093 (lr 1e-3
with strong augmentation, weight decay 0.05, dropout 0.5).

### Which regulariser actually mattered

Ablations at the selected learning rate (1e-3), validation only, seed 0. Each
row changes exactly one thing relative to the final configuration:

| configuration | val macro F1 | Δ vs final |
|---|---|---|
| **final** (lr 1e-3, wd 0.05, dropout 0.5, strong aug, ls 0.10) | **0.6093** | — |
| mild augmentation instead of strong | 0.6003 | −0.0090 |
| weight decay 0.01 instead of 0.05 | 0.6055 | −0.0038 |
| **no dropout** instead of 0.5 | 0.5850 | −0.0243 |
| lr 3e-4 (with wd 0.05, dropout 0.5, mild aug) | 0.5654 | −0.0439 |
| lr 1e-4, wd 0.1, dropout 0.5, strong aug | 0.5191 | −0.0902 |

Two conclusions. **Learning rate dominated** — moving from 3e-4 to 1e-3 was worth
more than every regulariser combined, and dropping to 1e-4 was catastrophic.
Among the regularisers, **dropout contributed most** (−0.024 without it),
followed by strong augmentation (−0.009), with weight decay the least sensitive
(−0.004). Over-regularising hurt: weight decay 0.1 with lr 1e-4 gave the worst
result in the whole sweep.

A caveat: these deltas are single-seed and measured on 382 validation nodules,
so only the dropout and learning-rate effects are clearly larger than run-to-run
noise. The augmentation and weight-decay differences are suggestive, not proven.

Notably the **binary model overfits much less** (gap +0.192 vs +0.386), despite
having *fewer* training samples (1000 vs 1876). Removing the ambiguous class
removes label noise, and label noise is what a high-capacity network memorises
first.

## Were validation and test close?

**Yes — this is the reassuring part.** Test macro-F1 minus validation macro-F1:

- Model A: **+0.0085**
- Model B: **+0.0037**
- Binary: **+0.0303**

All three are tiny and positive, so the checkpoints selected on validation
generalised to test without optimistic bias. Combined with the untouched test
protocol (test was evaluated only after the configuration was frozen, and never
used for tuning or selection), I am confident the reported numbers are honest.

## What changed between the 3-class and binary experiments?

Removing *indeterminate* — same images, same patient splits, same architecture,
same hyperparameters, same seeds:

| metric | 3-class | binary | Δ |
|---|---|---|---|
| macro F1 | 0.5908 | 0.8323 | +0.2415 |
| accuracy | 0.5850 | 0.8357 | +0.2507 |
| balanced accuracy | 0.5920 | 0.8304 | +0.2384 |
| ROC-AUC | 0.7511 (OvR macro) | 0.8790 (binary) | +0.1279 |
| F1 benign | 0.5158 | 0.8563 | +0.3404 |
| F1 malignant | 0.6689 | 0.8083 | +0.1394 |

**Important caveat for the final table:** macro-F1 is not directly comparable
across 2 and 3 classes, because the chance level differs (majority baseline
0.3372 for binary vs 0.2138 for 3-class). Quoting "+24 points" without that
caveat overstates the effect. The fair comparisons are **per-class F1 for benign
and malignant** and **ROC-AUC**, and both still improve substantially: benign F1
+0.34, malignant F1 +0.14, AUC +0.13. So the conclusion survives the caveat.

**Conclusion of the sensitivity analysis:** the *indeterminate* category is the
dominant source of difficulty in our task, not the imaging, not the
representation, and not the architecture. A 2D single-slice ResNet-18 can
separate benign from malignant at ~0.83 macro-F1 and 0.88 AUC. The three-class
scores near 0.59 mostly measure how hard it is to reproduce radiologist
disagreement.

## Main limitations

1. **Small labelled cohort.** 1876 training / 382 validation / 396 test nodules
   against an 11.17M-parameter encoder. Severe overfitting follows directly.
2. **Wide confidence intervals.** Bootstrap 95% CIs on test macro-F1 span about
   ±0.05 (e.g. Model A seed 0: [0.543, 0.642]). Differences below ~0.05 between
   any two of our models should not be interpreted as real. This applies to the
   2D-vs-2.5D-vs-3D comparison too.
3. **Labels are radiologist opinion, not histopathology.** Malignancy *risk*,
   not confirmed cancer, and the *indeterminate* class is defined by reader
   disagreement.
4. **Midpoint medians are arbitrary at the boundary.** 449 nodules have medians
   of exactly 2.5 or 3.5 (`midpoint_median_v2`). Whether 2.5 is indeterminate and
   3.5 is malignant is a project convention, not a clinical fact. A re-run
   excluding those would be a useful extra sensitivity check.
5. **Single slice discards context by construction.** That is the point of my
   arm of the study, but it means my model cannot see nodule shape across slices.
6. **z-axis interpolation.** Volumes were resampled from 2.5 mm to 1 mm in z, so
   inter-slice context is partly interpolated. This affects 2.5D and 3D more than
   my 2D result, but it belongs in the paper's limitations.
7. **One SSL configuration only.** MoCo v2 was run with a single set of
   hyperparameters (K 4096, T 0.2, m 0.999, 200 epochs). A negative transfer
   result should not be read as "SSL cannot help 2D", only that this
   configuration did not.
8. **Hyperparameters tuned on 382 validation nodules**, which is itself a small
   sample, so some of the sweep's apparent gains may not generalise.

## Verification performed on the shared data

- 7385 samples and 7385 masks, IDs matching exactly, constant `(104, 72, 80)`
  shape, no truncated files.
- Zero patient overlap and zero nodule overlap between train/validation/test,
  re-asserted programmatically on **every** training run (the code raises, not
  warns).
- SSL pool: 5133 nodules from 692 training-split patients, containing **zero**
  validation or test patients.
- Binary CSVs: counts match the specification exactly (1000/219/209 with
  622+378, 134+85, 117+92) and every binary split is a strict **subset** of the
  corresponding original split, proving patients were filtered rather than
  moved or resplit.
- Package `ssl_pretrain_train.csv` is byte-identical (md5) to the copy in the V2
  dataset.

**One warning for the team:** the binary CSVs contain two label columns. The
correct one is **`binary_class_index`** (0 = benign, 1 = malignant). The legacy
`binary_label` column inherited from the V2 metadata is **empty for 73% of rows**
(731 of 1000 in train); reading it silently corrupts the labels. Our loader now
raises on empty label values.

**One decision that needs ratifying:** the HU window was never specified in the
protocol. Raw values in the stored arrays range from −3024 to +3080, so training
requires clipping. I used **clip [−1000, 400] → [0,1]**, then standardised with
the 3-class train-split mean/std (0.3651 / 0.3145), documented in
`docs/preprocessing_contract.md`. If Fatima or Zaineb used a different window,
our three results are not measuring spatial context alone and this must be
reconciled before the final comparison table is built.
