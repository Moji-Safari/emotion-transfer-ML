# AffectWave

### Physiological Emotion Transfer via Wearable Sensors

*A research prototype for subject-independent stress recognition and haptic affect transfer.*

**Author:** Mojtaba Safari
**Status:** Active research prototype
**Version:** 1.0 (October 2026)

---

## Abstract

AffectWave is a two-device wearable system that detects physiological stress
in a host wearer and communicates the detected state to a guest wearer
through a haptic pattern. The system comprises:

1. **Recognition pipeline** — a machine learning classifier that maps
   wrist-worn physiological signals (electrodermal activity and skin
   temperature) to a binary baseline/stress decision, evaluated under
   leave-one-subject-out cross-validation on the WESAD dataset.

2. **Transfer pipeline** — a stress-to-haptic mapping that converts the
   classifier's output into a vibration pattern rendered on a second
   device or in a browser-based simulation.

The recognition component achieves an F1 score of 0.907 ± 0.187
(mean ± std across 15 held-out subjects) using 4 handcrafted features
with per-subject calibration. The transfer component is a proposed
design informed by the affective haptics literature; it is not
validated to induce a specific emotional state in the receiver.

---

## Table of Contents

1. [Motivation](#motivation)
2. [System Architecture](#system-architecture)
3. [Final Model](#final-model)
4. [Key Results](#key-results)
5. [Technologies Used](#technologies-used)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Repository Structure](#repository-structure)
9. [Reproducing the Experiments](#reproducing-the-experiments)
10. [Scientific Contributions](#scientific-contributions)
11. [Limitations](#limitations)
12. [Future Work](#future-work)
13. [References](#references)

---

## Motivation

Human emotional states are not directly observable. Two people in the
same room can occupy very different affective states — a teacher
excited about a topic, a student bored by it — and there is no simple
way for one to communicate their state to the other. Physiological
signals offer a window into affective states that language and facial
expression miss.

AffectWave asks: *Can physiological signals from one person be used
to recognize an affective state and communicate a matching signal to
another person through a wearable interface?*

The problem is decomposed into two independently researchable parts:

1. **Recognition** — detecting stress from physiological signals with
   subject-independent generalization (the hard part, and the focus of
   the current work).
2. **Transfer** — mapping the detected state to a haptic pattern that
   a receiver can perceive (the design part, informed by literature).

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HOST WEARER                              │
│  Wrist EDA (4 Hz) + Wrist TEMP (4 Hz)                       │
│              ↓                                              │
│  30-second windows (120 samples each)                       │
│              ↓                                              │
│  4 handcrafted features                                     │
│              ↓                                              │
│  Per-subject calibration (B_s, σ_s from baseline)           │
│              ↓                                              │
│  RBF SVM classifier → P(stress)                             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                    TRANSFER LAYER                           │
│  Stress probability → 3-band haptic classification          │
│              ↓                                              │
│  Vibration pattern (intensity, frequency, duration)         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                    GUEST WEARER                             │
│  Haptic actuator (or browser simulation)                    │
│  Visual + tactile feedback                                  │
└─────────────────────────────────────────────────────────────┘
```

The pipeline is implemented as:

- **Python backend** — ML inference, calibration, and haptic pattern
  generation exposed via a Flask REST API.
- **React frontend** — a simulation environment with three panels:
  host signal generation, recognition display, and guest haptic
  visualization.

---

## Final Model

### Architecture

**RBF Support Vector Machine** (`C=1.0`, `gamma='scale'`, `class_weight='balanced'`).

### Features (4 total)

| Feature | Description | Source |
|---|---|---|
| `eda_mean` | Mean electrodermal activity over 30 s | Wrist EDA |
| `eda_std` | Standard deviation of EDA | Wrist EDA |
| `eda_mean_absolute_change` | Mean absolute difference between consecutive samples | Wrist EDA |
| `temp_mean` | Mean skin temperature over 30 s | Wrist TEMP |

### Per-subject calibration

Applied at inference time using the subject's own baseline windows:

```
X* = (X − B_s) / max(σ_s, 0.01)
```

Where:
- `B_s` = mean feature vector over baseline windows
- `σ_s` = standard deviation over baseline windows
- `0.01` = floor value, preventing numerical instability from near-zero σ

### Evaluation protocol

- **Leave-One-Subject-Out (LOSO)** cross-validation on 15 WESAD subjects
- Train on 14 subjects, test on the held-out subject, repeat 15 times
- No subject appears in both train and test
- Calibration uses **only** the test subject's own baseline windows

### Reported performance

| Metric | Value |
|---|---|
| F1 (per-subject mean) | **0.9069 ± 0.1873** |
| F1 (aggregate, window-weighted) | **0.9294** |
| Accuracy (aggregate) | **0.9516** |
| Precision (aggregate) | ~0.93 |
| Recall (aggregate) | ~0.93 |

The standard deviation (±0.19) reflects substantial per-subject
variability, which is reported and analyzed in the results files.

---

## Key Results

All results reported under LOSO on 15 WESAD subjects with identical
training/evaluation protocols.

| Experiment | F1 (mean) | Note |
|---|---|---|
| Majority baseline (predict baseline always) | 0.000 | Accuracy = 0.643 (class prior) |
| Random Forest + 9 handcrafted EDA features | 0.622 | Initial baseline |
| Logistic Regression + 9 features | 0.580 | Linear model |
| Linear SVM + 9 features | 0.587 | Linear model |
| RBF SVM + 9 features | 0.719 | Best model comparison result |
| RBF SVM + 3 EDA features (reduced) | 0.714 | No loss from removing 6 features |
| RBF SVM + 3 EDA + 1 TEMP feature | 0.784 | Significant vs EDA-only (p = 0.0076) |
| RBF SVM + 4 features + mean calibration | 0.862 | Significant vs no calibration (p = 0.041) |
| RBF SVM + 4 features + std calibration | 0.902 | S3 collapses (F1 = 0.174) |
| **RBF SVM + 4 features + floored std calibration** | **0.907** | **Final model** |
| CNN-LSTM on raw EDA (890 windows) | 0.698 | Unstable; 4 subjects at F1 = 0.000 |

### Notable negative results

- **BVP features (heart rate, HR std, pulse amplitude) did not produce a
  statistically reliable improvement.** The apparent +0.04 F1 gain was
  driven entirely by a single subject (S14) whose EDA signal is
  uninformative. Per-subject analysis showed the addition degraded three
  well-classified subjects (S2, S10, S15). BVP was rejected.

- **A CNN-LSTM operating on raw calibrated EDA waveforms underperformed
  the handcrafted-feature SVM.** The DL model produced F1 = 0.698 ± 0.423
  (vs SVM's 0.907 ± 0.187), collapsed to F1 = 0.000 on four subjects,
  and exhibited ~2.3× higher per-subject variance. The result is
  consistent with the well-documented difficulty of training deep
  architectures on small datasets (890 windows, 15 subjects). The CNN-LSTM
  performed notably well on precisely the two subjects the SVM struggled
  with (S3, S14) — a topic for future investigation.

---

## Technologies Used

### Programming languages

| Language | Version | Purpose |
|---|---|---|
| Python | 3.11 | ML pipeline, backend API |
| JavaScript (ES2022) | — | Frontend (React) |
| Markdown | — | Documentation |

### Python libraries

| Library | Version | Role |
|---|---|---|
| **NumPy** | 1.26+ | Numerical arrays and vector operations |
| **SciPy** | 1.11+ | Statistical tests (Wilcoxon), signal processing (peak detection) |
| **pandas** | 2.0+ | Data manipulation (used in exploratory stages) |
| **scikit-learn** | 1.3+ | SVM, Logistic Regression, Random Forest, metrics, pipelines, StandardScaler |
| **matplotlib** | 3.7+ | Diagnostic plots |
| **TensorFlow (CPU)** | 2.21 | CNN-LSTM model for the deep learning experiment |
| **Flask** | 3.0+ | REST API server |
| **Flask-CORS** | 4.0+ | Cross-origin requests from the React dev server |
| **requests** | 2.31+ | Python client for API testing |

### Frontend technologies

| Technology | Version | Role |
|---|---|---|
| **React** | 18 | Component-based UI |
| **Vite** | 5+ | Fast dev server and build tool |
| **Plain JavaScript** | ES2022 | No TypeScript; simpler for research prototype |
| **CSS3** | — | Custom styling with CSS variables |
| **Inter** (Google Fonts) | — | Typography |

### Development environment

| Tool | Purpose |
|---|---|
| **Anaconda** | Python environment and package management |
| **conda** | Isolated environment: `emotion-ml` |
| **pip** (via Tsinghua mirror) | Package installation on constrained networks |
| **Jupyter Lab** | Exploratory data analysis |
| **PowerShell** | Command-line interface (Windows) |

### Data

| Source | Description |
|---|---|
| **WESAD** | Wearable Stress and Affect Detection dataset — 15 subjects, chest and wrist signals, baseline/stress/amusement/meditation conditions |

### Research methodology

| Method | Purpose |
|---|---|
| **Leave-One-Subject-Out (LOSO) cross-validation** | Subject-independent evaluation |
| **Wilcoxon signed-rank test** | Non-parametric paired comparison of per-subject results |
| **Feature ablation** | Marginal contribution of individual features |
| **Per-subject calibration** | Subject-specific baseline normalization |
| **Class weighting** | Handling the 64/36 class imbalance |

---

## Installation

### Requirements

- **Python 3.11** (via Anaconda recommended)
- **Node.js 18+** and npm
- **WESAD dataset** (download separately, ~3 GB)
- **~3 GB disk space** for models and artifacts
- OS: Windows 10/11, macOS, or Linux

### Step 1: Clone and enter the project

```bash
git clone <repo-url> emotion-transfer
cd emotion-transfer
```

### Step 2: Create the Python environment

```bash
conda create -n emotion-ml python=3.11
conda activate emotion-ml
```

### Step 3: Install Python dependencies

If PyPI is slow on your network, use a mirror:

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 120 \
    numpy pandas scipy scikit-learn matplotlib \
    flask flask-cors requests \
    tensorflow-cpu
```

### Step 4: Download and place WESAD

Download from:
https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html

Extract and place the subject folders (S2, S3, ..., S17) at:
```
E:\dataset\WESAD\WESAD\
```

If your path differs, update `WESAD_ROOT` in
`src/preprocessing/wesad_preprocessing.py`.

### Step 5: Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### Step 6: Export the deployed model

```bash
python -m scripts.export_model
```

This creates `models/deployed/svm.pkl` and companion metadata.

---

## Usage

### Running the API

In terminal 1:

```bash
conda activate emotion-ml
python -m api.app
```

The API serves at `http://localhost:5000`.

### Running the frontend

In terminal 2:

```bash
cd frontend
npm run dev
```

The UI serves at `http://localhost:5173`.

### API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Server status and model metadata |
| `POST` | `/calibrate` | Compute B_s and σ_s from baseline windows |
| `POST` | `/predict` | Classify one window as baseline or stress |
| `POST` | `/generate-haptic` | Map stress probability to a haptic pattern |

### Example: `/predict`

**Request body:**

```json
{
  "features": [0.5, 0.1, 0.01, 33.1],
  "calibration": {
    "B_s": [0.30, 0.05, 0.005, 33.0],
    "sigma_s": [0.01, 0.002, 0.0005, 0.3]
  }
}
```

**Response:**

```json
{
  "stress_probability": 0.7072,
  "prediction": 1,
  "band": "high",
  "calibrated": true
}
```

### Example: `/generate-haptic`

**Request:**

```json
{ "stress_probability": 0.8 }
```

**Response:**

```json
{
  "band": "high",
  "intensity": 0.857,
  "frequency_hz": 6.67,
  "duration_ms": 1250,
  "pulses": [
    { "t_ms": 0, "intensity": 0.857 },
    { "t_ms": 150, "intensity": 0.857 },
    ...
  ],
  "description": "Strong, fast pulses",
  "stress_probability": 0.8
}
```

### Test the API

```bash
python scripts/test_api.py
```

---

## Repository Structure

```
emotion-transfer/
│
├── README.md                          This file
├── .gitignore
│
├── api/                               Flask backend
│   ├── __init__.py
│   ├── app.py                         API server
│   └── haptic_mapping.py              Stress → haptic pattern
│
├── data/
│   └── processed/                     (reserved for cached features)
│
├── frontend/                          React UI
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── api.js
│       ├── styles.css
│       └── components/
│           ├── Header.jsx
│           ├── HostPanel.jsx
│           ├── DecisionPanel.jsx
│           ├── GuestPanel.jsx
│           └── Footer.jsx
│
├── models/
│   ├── deployed/                      Production model artifacts
│   │   ├── svm.pkl
│   │   ├── feature_names.json
│   │   └── metadata.json
│   ├── deep_learning/
│   └── random_forest/
│
├── notebooks/
│   └── lab_notebook.md                Personal experiment log
│
├── results/
│   ├── figures/                       Diagnostic plots
│   ├── logs/                          Raw terminal outputs per experiment
│   └── metrics/                       Distilled result summaries (Markdown)
│
├── scripts/                           Experiment entry points
│   ├── export_model.py                Save deployed model
│   ├── inspect_wesad.py               Dataset structure
│   ├── run_loso.py                    First LOSO (RF baseline)
│   ├── compare_models.py              LR / SVM / RF comparison
│   ├── feature_ablation.py            EDA feature ablation
│   ├── reduced_features.py            3-feature vs 9-feature
│   ├── compare_eda_temp.py            EDA vs TEMP vs combined
│   ├── temp_ablation.py               TEMP feature ablation
│   ├── compare_with_bvp.py            EDA+TEMP vs EDA+TEMP+BVP
│   ├── bvp_ablation.py                BVP feature ablation
│   ├── diagnose_subjects.py           Raw signal diagnostics
│   ├── diagnose_s3.py                 S3 calibration collapse
│   ├── compare_calibration.py         Mean calibration
│   ├── compare_calibration_v2.py      Mean vs std
│   ├── compare_calibration_floor.py   Std with floor
│   ├── compare_dl_vs_rf.py            CNN-LSTM vs SVM
│   └── test_api.py                    API smoke test
│
├── src/                               Core library
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── wesad_preprocessing.py     Load, window, align labels
│   │   └── calibration.py             Calibration transformers
│   ├── features/
│   │   ├── eda_features.py            9 handcrafted EDA features
│   │   ├── temp_features.py           3 TEMP features
│   │   ├── bvp_features.py            Peak-based BVP features
│   │   ├── build_features.py          Feature matrix builder
│   │   └── build_all_features.py      Combined feature matrices
│   ├── data/
│   │   └── build_dataset.py           Multi-subject dataset builder
│   ├── models/
│   │   ├── random_forest_model.py     (legacy)
│   │   ├── model_factory.py           Model registry
│   │   └── cnn_lstm_model.py          CNN-LSTM architecture
│   └── evaluation/
│       ├── loso.py                    Basic LOSO
│       ├── loso_calibrated.py         Mean calibration LOSO
│       ├── loso_calibrated_std.py     Std calibration LOSO
│       ├── loso_calibrated_std_floor.py  Floored std LOSO
│       └── loso_deep.py               CNN-LSTM LOSO
│
└── tests/                             (reserved)
```

---

## Reproducing the Experiments

The full experiment sequence is captured in `scripts/`. Run them in order
to reproduce every result reported in this README:

```bash
# 1. Dataset inspection
python -m scripts.inspect_wesad

# 2. First LOSO with Random Forest
python -m scripts.run_loso

# 3. Model comparison (LR / Linear SVM / RBF SVM / RF)
python -m scripts.compare_models

# 4. Feature ablation (which of 9 EDA features matter?)
python -m scripts.feature_ablation

# 5. Reduced feature set (3 features vs 9)
python -m scripts.reduced_features

# 6. Add TEMP (EDA vs TEMP vs combined)
python -m scripts.compare_eda_temp

# 7. TEMP feature ablation (which TEMP feature matters?)
python -m scripts.temp_ablation

# 8. Add BVP (negative result)
python -m scripts.compare_with_bvp
python -m scripts.bvp_ablation

# 9. Subject-level diagnostics
python -m scripts.diagnose_subjects
python -m scripts.diagnose_s3

# 10. Calibration experiments
python -m scripts.compare_calibration
python -m scripts.compare_calibration_v2
python -m scripts.compare_calibration_floor

# 11. Deep learning comparison (negative result)
python -m scripts.compare_dl_vs_rf

# 12. Export the deployed model
python -m scripts.export_model

# 13. Test the API
python scripts/test_api.py
```

All raw outputs are in `results/logs/`, all distilled summaries in
`results/metrics/`, and personal notes in `notebooks/lab_notebook.md`.

---

## Scientific Contributions

This work makes four contributions to the subject-independent
wearable stress detection literature:

### 1. A controlled evaluation of the design space

Every experiment holds all variables constant except one (model, feature
set, or calibration), enabling clean attribution of performance changes.
The full LOSO protocol is identical across all 15 experiments.

### 2. Per-subject calibration as a significant improvement

A per-subject transformation of features into relative deviations from
each subject's own baseline improves F1 by +0.12 (p = 0.0046, Wilcoxon
signed-rank). The improvement is broad: 12 of 15 subjects improved. This
reproduces, on EDA+TEMP features, the calibration benefit previously
reported for ECG/HRV features (Zenodo preprint 22806710, 2026).

### 3. Documented negative results

Two negative findings are reported with per-subject analysis:

- **BVP features**: an apparent +0.04 F1 improvement was entirely
  driven by a single outlier subject and is not statistically
  significant (p = 0.69). Per-subject analysis revealed the addition
  degraded three well-classified subjects.

- **CNN-LSTM on raw EDA**: F1 = 0.698 ± 0.423, underperforming the SVM
  by 0.21 F1 and collapsing to F1 = 0.000 on four subjects. This is
  consistent with the difficulty of deep learning on small datasets.

### 4. A documented non-responder case

Subject S14 exhibits EDA baseline-stress difference of only 0.051 µS
(vs 1.7–2.2 µS for responsive subjects). This is consistent with the
~5–10% prevalence of electrodermal non-responders in the population
(Boucsein, 2012). The failure of any model to reliably classify S14
reflects this physiological constraint, not a modelling deficiency.

### 5. A complete prototype system

The recognition pipeline is packaged as a Flask REST API with endpoints
for calibration, prediction, and haptic pattern generation. A React
frontend demonstrates the end-to-end flow without requiring physical
hardware.

---

## Limitations

### Data

- **Small sample size.** 15 subjects, 889 windows. This constrains
  statistical power and prevents adopting more complex models.
- **Single dataset.** All results are from WESAD. Generalization to
  other datasets or real-world deployments is not validated.
- **Limited modality coverage.** Only wrist EDA and TEMP are used in the
  final model. Chest signals (ECG, EMG, Resp) were not evaluated.

### Methodology

- **The floor value (0.01) in std calibration is arbitrary.** No
  systematic sweep was performed. Other values might perform better.
- **TEMP may partially confound with posture and room conditions.**
  Not investigated in depth.
- **S14 and S3 remain imperfectly handled.** Documented in the metrics
  files.

### Transfer side

- **The haptic transfer is not validated.** The mapping from stress
  probability to haptic pattern is a proposed design, informed by the
  affective haptics literature. It is **not** shown to induce a specific
  emotional state in a receiver. Validating induction would require a
  controlled study with human participants.

---

## Future Work

### Short term

1. **Build the React frontend** (in progress) — three-panel simulation
   with host signal generation, recognition display, and guest haptic
   visualization.

2. **Sweep the calibration floor.** Test floor values 0.005, 0.02, 0.05
   to determine whether 0.01 is optimal.

3. **Investigate the S3 anomaly.** Why does std calibration cause S3's
   F1 to collapse from 0.83 to 0.17?

### Medium term

4. **Phasic EDA features.** Apply cvxEDA decomposition into tonic and
   phasic components. Replace raw statistical features with
   phasic-specific features (SCR count, rise time, recovery time).

5. **Add chest signals.** Evaluate whether ECG, EMG, or respiration
   improve subject-independent performance.

6. **Investigate the CNN-LSTM's S3/S14 success.** The DL model
   performed well on exactly the two subjects the SVM struggled with.
   A hybrid model that uses DL for anomalous subjects might be worth
   exploring with more data.

### Long term

7. **Validate the transfer.** Run a controlled study to measure whether
   the proposed haptic patterns induce measurable affective changes.

8. **Cross-dataset evaluation.** Validate the model on AffectiveROAD,
   K-EmoCon, or other wearable stress datasets.

9. **Physical hardware integration.** Replace the simulation with
   real wristband sensors and haptic actuators.

---

## References

### Dataset

Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., &
Van Laerhoven, K. (2018). *Introducing WESAD, a Multimodal Dataset
for Wearable Stress and Affect Detection.* ICMI 2018.
https://doi.org/10.1145/3242969.3242985

### Calibration

*Personalized Electrocardiographic and HRV Dynamics for Acute Stress
Detection: A Leave-One-Subject-Out Benchmark on WESAD.* Zenodo
Preprint, 2026.
https://zenodo.org/records/22806710

Akkaya, A. (2026). *Calibration, not architecture, limits
cross-subject wearable stress detection.* BMC Medical Informatics
and Decision Making.

### Feature selection

Guyon, I., & Elisseeff, A. (2003). *An Introduction to Variable and
Feature Selection.* Journal of Machine Learning Research, 3,
1157–1182.
https://jmlr.org/papers/v3/guyon03a.html

### Statistics

Wilcoxon, F. (1945). *Individual Comparisons by Ranking Methods.*
Biometrics Bulletin, 1(6), 80–83.

Demšar, J. (2006). *Statistical Comparisons of Classifiers over
Multiple Data Sets.* JMLR, 7, 1–30.
https://www.jmlr.org/papers/v7/demsar06a.html

Bouthillier, X., et al. (2021). *Accounting for Variance in Machine
Learning Benchmarks.* MLSys.
https://proceedings.mlsys.org/paper/2021/hash/cf004fdc76fa1a4f25f62e0eb5261ca3-Abstract.html

### Electrodermal activity

Boucsein, W. (2012). *Electrodermal Activity* (2nd ed.). Springer.
https://link.springer.com/book/10.1007/978-1-4614-1126-0

### Affective haptics

Eid, M., & Al Osman, H. (2016). *Affective Haptics: Current Research
and Future Directions.* IEEE Access, 4, 26–49.
https://doi.org/10.1109/ACCESS.2016.2527305

### Wearable photoplethysmography

Charlton, P. H., et al. (2022). *Wearable Photoplethysmography for
Cardiovascular Monitoring.* Frontiers in Digital Health, 4, 840840.
https://doi.org/10.3389/fdgth.2022.840840

### Deep learning on small datasets

The negative CNN-LSTM result is consistent with general findings in
the small-data machine learning literature. See, e.g., the review:

Brigato, L., & Iocchi, L. (2021). *A Close Look at Deep Learning with
Small Data.* IJCNN 2021.
https://arxiv.org/abs/2012.12874

---

## Author

**Mojtaba Safari**

AffectWave is a research prototype exploring the intersection of
wearable sensing, machine learning, and affective haptics. The
project is intended for research and educational use.

For questions, collaborations, or issues, please open an issue on
the project repository.

---

## Acknowledgments

This work uses the WESAD dataset, provided by the University of
Siegen. The dataset's creators — Schmidt, Reiss, Duerichen,
Marberger, and Van Laerhoven — made this research possible through
their careful data collection and public release.

---

## License

The project code is released for research and educational purposes.
The WESAD dataset is subject to its own license terms — see the
WESAD distribution page for details. Commercial use of the WESAD
dataset may require separate permission from the University of
Siegen.

**Last updated:** October 2026
```
