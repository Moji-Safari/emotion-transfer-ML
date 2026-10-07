# Lab Notebook — Emotion Transfer Project

Personal log of the WESAD stress-recognition work.
Entries are informal. Measured numbers live in `results/metrics/`.
Raw output lives in `results/logs/`.

---

## 2026-10-07 — Model comparison

Ran `scripts/compare_models.py`. Five models compared under LOSO on
15 WESAD subjects, 889 windows, 9 handcrafted EDA features.

Results (per-subject F1):
- svm_rbf        0.7190 ± 0.2258   ← best
- random_forest  0.6349 ± 0.2357
- svm_linear     0.5866 ± 0.2821
- logistic       0.5804 ± 0.2803
- majority       0.0000            ← floor (always predicts baseline)

Takeaways:
1. Nonlinear > linear by ~0.10 F1. The boundary between baseline and
   stress in this feature space is not a straight line.
2. Per-subject variance is huge. Some subjects near F1 = 1.0 (S4, S16),
   others near 0.0 (S14 for three models). The average hides this.
3. Majority baseline gets 64% accuracy. All models beat it, but only
   by ~8-14 points. Real ML gain, not a landslide.

Things I don't understand yet:
- Why does S14 fail so badly for logistic, svm_linear, and RF but
  svm_rbf gets 0.091? Is S14 just a hard subject, or is there
  something odd about S14's signal? Should check.
- Why does RF collapse on S5 (F1 = 0.338) while svm_linear gets
  0.864 and svm_rbf gets 0.667? That's backwards. RF should be at
  least as good as linear on nonlinear data. Suspicious. Check later.

---

## 2026-10-07 — Feature ablation

Ran `scripts/feature_ablation.py`. Baseline = all 9 features with
RBF SVM. Then dropped one feature at a time, re-ran LOSO.

Baseline: F1(subj) = 0.7190.

Biggest losses (dropping HURT):
  drop min      → +0.0070
  drop max      → +0.0067
  drop mean     → +0.0054
  drop median   → +0.0054

Biggest gains (dropping HELPED):
  drop std      → -0.0263   ← weird, the largest effect
  drop range    → -0.0138
  drop slope    → -0.0060

Everything else within ±0.003.

Takeaways:
1. ALL ΔF1 values are tiny (≤ 0.033 in magnitude). Per-subject std is
   ±0.23, so this is basically noise-level.
2. No single feature is critical. Dropping the "best" one costs 0.007.
   That means the 9 features are basically redundant with each other.
3. My guess: the 9 features collapse into ~3 concepts:
     - central level: mean, median, min, max, range
     - dispersion:    std, mean_absolute_change, std_change
     - trend:         slope
4. Dropping std gave F1 0.7453, a +0.026 gain. Is that real or noise?
   Need a paired test across folds to find out. If real, remove std.
   If noise, keep it.

What I should test next:
- Does a 3-feature model (mean + std + slope, or similar) match the
  9-feature baseline? If yes, the other 6 features are dead weight.
- What's actually wrong with S14? Investigate.
- Should I trust the std result? Paired test.

Honest self-critique:
- I predicted mean would be the most important feature. Wrong.
- I predicted slope would be useless. Half right — it's noise.
- The correlation structure of statistical EDA features is something
  I underweighted. Lesson: don't assume "average value" dominates a
  nonlinear model the way it dominates a linear one.

---

## Open questions / TODO

- [ ] Reduced-feature experiment (3 features vs 9)
- [ ] Paired test on the std improvement
- [ ] Investigate S14 and S5 anomalies
- [ ] Document the model comparison run (logs + metrics files)
- [ ] Decide: keep 9 features or reduce?
- [ ] After EDA is settled: add wrist TEMP features
- [ ] Long-term: phasic EDA decomposition (cvxEDA), nSCR features
- [ ] Long-term: the emotion-transfer half of the project

-----------
## 2026-10-08 — Reduced-feature experiment

Ran `scripts/reduced_features.py`.

Headline: dropping `std` from the 9-feature set improves F1 by +0.026
with p = 0.0125 (Wilcoxon). 9/15 subjects improved, 1 got worse,
5 unchanged. This is a REAL effect, not noise.

Also: 3 features (mean, std, mean_absolute_change) match the 9-feature
baseline within 0.005 F1. The other 6 features are dead weight.

Interesting: `mean` alone gets F1 = 0.577 with std ±0.37. Enormous
variance. So central EDA level alone is weak — dispersion matters.

And: `slope` (temporal trend) is worse than `mean_absolute_change`
(rate of change). This means the model cares about local dynamics,
not global direction. Makes sense physiologically — SCR events
are sharp, not gradual.

Decision: canonical feature set = {mean, std, mean_absolute_change}.

Remaining questions:
- Why does std hurt the RBF SVM? Is it a scaling issue? RBF SVM
  uses distance in feature space; std has a different scale and
  distribution from mean. Might be hurting the kernel's geometry.
  Interesting but not blocking.
- Should I check whether `mean, MAC` (2 features, no std) works
  even better? Quick test. Not urgent.

Next: add wrist TEMP.

--------------
