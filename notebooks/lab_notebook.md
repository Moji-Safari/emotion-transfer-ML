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
## 2026-10-11 — EDA vs TEMP vs EDA+TEMP

Ran `scripts.compare_eda_temp`.

Headline: EDA+TEMP gets F1(subj) = 0.7545 vs EDA-only 0.7142, a
+0.040 improvement. BUT the paired test gives p = 0.198 — not
significant.

Surprising: TEMP-only (0.7155) ≈ EDA-only (0.7142). Literature says
EDA should be much stronger. Two possibilities:
  - My 3-feature EDA set is weaker than the full 9-feature set
  - TEMP is picking up a confound (posture, room temp drift)

Question: should I trust TEMP?

Quick answer: not fully, but it's not hurting either. 9/15 subjects
improved, 5 worse. That's slightly in favor of TEMP.

Decision to make: keep 6-feature EDA+TEMP, or reduce TEMP to
1 feature (temp_mean)? If temp_mean alone gives most of the gain,
the wearable model is 4 features — much better.

TODO:
- [ ] TEMP feature ablation (or at least test temp_mean alone)
- [ ] Check whether TEMP gain is spread across subjects or driven
      by 2-3 outliers
- [ ] If keeping TEMP, decide: 6 features or fewer?
- [ ] Then move to wrist BVP

Also: note that the +0.040 is not significant with 15 subjects.
This is a real limitation. If I had 30+ subjects, might cross the
significance threshold. Or might not — the effect could be small.


----------
## 2026-10-12 — TEMP feature ablation

Ran `scripts.temp_ablation`.

Headline: EDA (3 features) + temp_mean (1 feature) = 4 features,
gets F1 = 0.7842. That's +0.070 over EDA-only with p = 0.0076.
This is our first statistically significant improvement.

And it beats the 6-feature EDA+TEMP model (0.7545) by +0.030.
Fewer features, better result. Classic.

Why does only temp_mean help?
- Wrist TEMP responds to stress via vasoconstriction -> mean
  skin temperature shifts over tens of seconds.
- Within a 30-second window, TEMP variability and slope are
  dominated by sensor noise and thermal drift.
- So only the level carries stress info within our window size.

Odd: S13 goes from 0.609 (EDA-only) -> 0.313 (EDA+all TEMP)
-> 0.778 (EDA+temp_mean). Same EDA features. Only TEMP set
differs. The 6-feature model is unstable on this subject.

Decision: canonical feature set v3 = {eda_mean, eda_std,
eda_mean_absolute_change, temp_mean}. 4 features. Best F1 so far,
wearable-friendly.

TODO:
- [ ] Investigate S13 anomaly (why did adding temp_std/temp_slope
      break it?)
- [ ] Check TEMP confound (is temp_mean really capturing stress,
      or is it capturing room/posture?)
- [ ] Move to next modality: wrist BVP for heart-rate features
- [ ] Consider: does temp_mean help EDA-only more than EDA-only
      helps temp_mean? (asymmetry test)
- [ ] Save feature v3 as the canonical baseline for future
      experiments

------------------
## 2026-10-14 — EDA+TEMP vs EDA+TEMP+BVP

Headline: EDA+TEMP+BVP gets F1(subj) = 0.8247 vs EDA+TEMP at
0.7842. That's +0.040. But p = 0.69 — not significant.

Then I looked at the per-subject table.

S14: 0.087 → 0.865 (+0.778). Massive.
S10: 0.955 → 0.639 (−0.316). Broken.
S2:  0.710 → 0.444 (−0.265). Broken.
S15: 0.977 → 0.718 (−0.259). Broken.

S14's +0.778 is more than the total sum of deltas (+0.607). The
average improvement is basically the S14 rescue with everything
else slightly negative.

This is a textbook outlier-driven result. The mean went up. The
std went down. But the mechanism is not "BVP helps everyone" —
it's "BVP rescues one catastrophic subject while damaging three
good ones."

Lesson (again): look at per-subject tables, not just means.

Question: why does BVP rescue S14? Is it a real physiological
difference (S14's EDA is unreliable, BVP saves the day), or is it
chance?

Decision: run a BVP feature ablation. Maybe only one of the three
BVP features (hr_mean?) is doing useful work, like what happened
with TEMP. If a smaller BVP set gets the S14 gain without breaking
S10/S2/S15, it's real. Otherwise drop BVP.

Also: standard deviation dropped from 0.22 to 0.14. That's not
because the model got more consistent — it's because S14 is no
longer at F1 = 0.09, so the bottom tail got trimmed. The metric
is real but the interpretation is wrong.

TODO:
- [ ] BVP feature ablation
- [ ] If BVP fails, note it as a negative result and move on
- [ ] Investigate S14 after BVP decision
------------------------- 

## 2026-10-15 — BVP feature ablation

Ran `scripts.bvp_ablation`.

Answer: one BVP feature (bvp_hr_mean) matches the full 3-feature BVP
set. But the same pattern repeats: S14 gains +0.87, S2/S10/S15 lose
0.29–0.38. Paired test still p = 0.69.

This is not a feature problem. This is a subject problem.

Look at the same four subjects across the BVP experiments:
  S14: +0.87 (was F1 = 0.09 with EDA+TEMP, becomes 0.95 with BVP)
  S2:  −0.38
  S10: −0.34
  S15: −0.29

The same four subjects swing back and forth. That's systematic.

Hypothesis: something is different about these subjects' signals.
Possible causes:
  - Sensor placement issues during recording
  - Atypical physiology
  - Motion artifacts in one modality but not another
  - Some subjects have corrupted EDA/TEMP, others have corrupted BVP

If S14's EDA is corrupted, BVP saving it is legitimate. If BVP is
also corrupted, we're exploiting an artifact.

Can't tell without investigating. And the investigation would take
1–2 hours with an uncertain payoff.

Decision: reject BVP. Stick with 4 features (EDA+TEMP).
Report the BVP negative result honestly in the writeup.

Lesson: the +0.04 mean improvement from BVP was not a finding.
It was the S14 outlier wearing a mean as a disguise.

TODO:
- [ ] Note S14 anomaly in writeup as a limitation
- [ ] Move to next phase of the project
-----------------
## 2026-10-15 — Subject diagnosis

Ran `scripts.diagnose_subjects` on S2, S4, S10, S14, S15, S16.

The key column is Δ = stress mean − baseline mean per signal:

  Subject  EDA Δ     TEMP Δ   BVP Δ
  S2       +0.561    −2.341   −0.076
  S4       +2.188    −0.534   −0.188
  S10      +2.073    −0.266   −0.019
  S14      +0.051    +0.717   −0.007   ← !!!
  S15      +1.028    +0.135   −0.001
  S16      +1.699    +0.734   +0.018

S14's EDA delta is +0.051. Effectively zero. The stress signal
is not present in S14's EDA.

Everything clicks:
- S14 fails with EDA+TEMP because there is no EDA stress signal.
- S14 "succeeds" with EDA+TEMP+BVP because the model is
  overfitting to noise in S14's BVP features.
- The +0.87 "improvement" is not learning signal; it's fitting noise.

Also: S2 and S15 have very noisy BVP (std 75.87 and 67.47 vs 29.80
for S16). Their BVP has motion artifacts or bad sensor contact.
This explains why they get worse when BVP is added.

S10 remains unexplained: strong EDA, normal BVP stats, but breaks
with BVP. Might need plot inspection. Not blocking.

Decision: BVP is out. Final model = EDA+TEMP, 4 features.

Lesson: An improvement in the aggregate metric is not a finding
unless it survives per-subject analysis. S14 was a warning sign,
not a win. This is the second time this has happened (the first
was when std looked important but turned out to be noise after
paired testing).

TODO:
- [ ] Note in the writeup: S14's EDA signal carries no stress
      information; the reported F1 for that subject is a
      limitation of the dataset, not the model.
- [ ] Move to Phase 8 (Flask API).
------------
Boucsein (2012) notes that a subset of individuals exhibit low electrodermal reactivity. More recent work (Thomas & Rabinak, 2025) finds that approximately 16% of healthy adults meet the criteria for SCR non-responders, supporting the classification of S14 as a physiological non-responder rather than a model failure.
-------------
## 2026-10-10 — Calibration-aware inference

Ran `scripts.compare_calibration`.

WOW. This worked.

  No calibration: F1 = 0.7842 ± 0.2193
  Calibrated:     F1 = 0.8617 ± 0.2434

+0.078 improvement, p = 0.041 (significant). 12/15 improved.
Median per-subject F1: 0.778 -> 0.952. The typical subject is
now almost perfectly classified.

Biggest gains:
  S2:  +0.243
  S9:  +0.226
  S13: +0.222
  S3:  +0.194

Two subjects got worse:
  S14: -0.087 (0.087 -> 0.000). Makes sense — no baseline
       reactivity to calibrate. Calibration can't help when
       there's nothing to transform.
  S16: -0.250 (1.000 -> 0.750). Interesting. S16 was perfect
       before. Why does calibration break it?

Theory for S16: the formula divides by |B_s|. For TEMP, |B_s|
is ~30. For EDA, |B_s| is ~0.3. After calibration, TEMP features
get squeezed into a tiny range and EDA features get amplified.
The SVM's distance metric is dominated by EDA. If S16 was
relying on TEMP to distinguish its windows, that signal is now
suppressed.

Possible fix: normalize by baseline std instead of baseline mean.
  X* = (X - B_s) / sigma_s
This would give each feature unit variance after calibration.

Lesson: this is the THIRD experiment where looking at per-
subject results (not just aggregates) was essential. Two of
three improvements in this project have been broad, one was
outlier-driven. Always check.

Decision: keep calibration. Report S14 and S16 as limitations.

TODO:
- [ ] Consider trying std-based normalization to fix S16
- [ ] Update writeup with calibration as the headline result
- [ ] Next: Flask API or Deep Learning phase
--------------