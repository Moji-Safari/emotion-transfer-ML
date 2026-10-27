# AffectWave

**Physiological Emotion Transfer via Wearable Sensors**

> A research prototype that detects affective states from wearable physiological signals and maps the detected state to a vibrotactile pattern.

**A project by Mojtaba Safari**

---

## Overview

AffectWave is a two-part wearable system:

1. **Host Device** — a wearable that records physiological signals and uses a machine learning classifier to detect the wearer's affective state (baseline, stress, amusement, or meditation).
2. **Guest Device** — a haptic wristband that receives the detected state and renders a vibrotactile pattern.

The system runs as a Flask REST API with a React frontend. The frontend includes a mock subject panel that generates physiologically plausible body conditions for testing.

**Important:** The recognition component is validated on the WESAD dataset. The transfer component is a proposed design informed by the affective haptics literature; it is **not** validated to induce a specific emotional state in the receiver.

---

## Current status

| Component | Status |
|---|---|
| WESAD data pipeline | Complete |
| Wrist binary pipeline (stress) | Complete |
| Chest four-class pipeline | Complete |
| Per-subject calibration | Complete |
| Feature ablation | Complete |
| Model comparison | Complete |
| Deep learning evaluation | Complete (negative result) |
| Flask REST API | Complete |
| React frontend (dashboard) | Complete |
| Haptic mapping | Complete (design prototype) |
| Physical hardware | Future work |

---

## Final models

### Wrist binary pipeline (stress detection)

| Property | Value |
|---|---|
| **Signals** | Wrist EDA (4 Hz), wrist TEMP (4 Hz) |
| **Window** | 30 s non-overlapping, 120 samples |
| **Features (4)** | `eda_mean`, `eda_std`, `eda_mean_absolute_change`, `temp_mean` |
| **Calibration** | Floored standard deviation: \(X^* = (X - B_s) / \max(\sigma_s, 0.01)\) |
| **Model** | RBF SVM (`C=1.0`, `gamma='scale'`, `class_weight='balanced'`) |
| **Evaluation** | LOSO, 15 subjects, 889 windows |

**Results:**
- F1 (per-subject mean): **0.907 ± 0.187**
- F1 (aggregate): **0.929**
- Accuracy (aggregate): **0.952**

### Chest four-class pipeline (affective state recognition)

| Property | Value |
|---|---|
| **Signals** | Chest ECG (700 Hz), chest RESP (700 Hz) |
| **Window** | 60 s non-overlapping, 42,000 samples |
| **Features (9)** | HR mean, RMSSD, SDNN, LF/HF, total power, respiratory rate, spectral energy, autocorrelation lag, RRV |
| **Calibration** | Floored standard deviation (same formula) |
| **Model** | RBF SVM (one-vs-rest) |
| **Evaluation** | LOSO, 15 subjects, 696 windows |

**Results (calibrated):**
- Macro F1: **0.777**
- Baseline F1: **0.861**
- Stress F1: **0.823**
- Amusement F1: **0.560**
- Meditation F1: **0.866**

---

## Haptic transfer

The detected state maps to a vibrotactile pattern with a specific pulse rate:

| State | Rate | Rationale |
|---|---:|---|
| Baseline | 0.3 Hz | Low-arousal resting pattern |
| Meditation | 0.5 Hz | Slow rhythmic pattern aligned with resting breathing |
| Amusement | 2.0 Hz | Medium rate for positive, high-arousal affect |
| Stress | 6.0 Hz | Fast, strong vibration associated with high arousal |

These rates are prototype choices informed by the affective haptics literature. They are **not** validated to induce the corresponding states.

---

## Key scientific findings

1. **Per-subject calibration is the strongest improvement.** It improves the wrist pipeline by +0.12 F1 (\(p = 0.0046\)) and the chest pipeline by +0.12 Macro F1. The improvement generalizes across modalities and task types.
2. **Nonlinear models beat linear models.** RBF SVM (0.719) outperforms Random Forest (0.635), Linear SVM (0.587), and Logistic Regression (0.580). The boundary between states is not linearly separable.
3. **Feature redundancy.** The nine EDA features can be reduced to three with only a 0.005 F1 loss. The nine features represent approximately three underlying concepts: central level, dispersion, and temporal trend.
4. **Amusement is the hardest class (F1 = 0.56).** This is a valence limitation: peripheral signals capture arousal more directly than valence, so amusement (positive, high arousal) and stress (negative, high arousal) can be difficult to separate without a valence-sensitive signal such as EEG.
5. **Two documented negative results.**
   - BVP features appeared to improve F1 by +0.04, but the gain was driven by one subject and was not significant (\(p = 0.69\)).
   - A CNN-LSTM on raw EDA underperformed the SVM (F1 = 0.70 vs. 0.91), consistent with the difficulty of deep learning on small datasets.
6. **S14 is an electrodermal non-responder.** Its EDA delta between baseline and stress is +0.051 µS, versus +1.7 to +2.2 µS for responsive subjects. No model fully recovers S14; this appears to be a physiological constraint rather than simply a modelling deficiency.

---

## Repository structure

```text
emotion-transfer/
├── README.md
├── api/                          # Flask backend
│   ├── app.py                    # API server
│   └── haptic_mapping.py          # State-to-haptic mapping
├── frontend/                     # React UI
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
│           ├── HostNecklace.jsx
│           ├── GuestWristband.jsx
│           ├── HumanMock.jsx
│           ├── PredictionForm.jsx
│           └── Footer.jsx
├── models/
│   └── deployed/
│       ├── svm.pkl                       # Wrist binary model
│       ├── feature_names.json
│       ├── metadata.json
│       ├── chest_svm.pkl                 # Chest four-class model
│       ├── chest_feature_names.json
│       └── chest_metadata.json
├── results/
│   ├── figures/                  # Diagnostic figures
│   ├── logs/                     # Raw experiment outputs
│   └── metrics/                  # Distilled results (Markdown)
├── scripts/                      # Experiment entry points
│   ├── export_model.py
│   ├── export_chest_model.py
│   ├── diagnose_chest_model.py
│   ├── compare_models.py
│   ├── feature_ablation.py
│   ├── reduced_features.py
│   ├── compare_eda_temp.py
│   ├── temp_ablation.py
│   ├── compare_with_bvp.py
│   ├── bvp_ablation.py
│   ├── diagnose_subjects.py
│   ├── diagnose_s3.py
│   ├── compare_calibration.py
│   ├── compare_calibration_v2.py
│   ├── compare_calibration_floor.py
│   ├── chest_30s.py
│   ├── compare_dl_vs_rf.py
│   └── test_api.py
├── src/                          # Core library
│   ├── preprocessing/
│   │   ├── wesad_preprocessing.py
│   │   └── calibration.py
│   ├── features/
│   │   ├── eda_features.py
│   │   ├── temp_features.py
│   │   ├── bvp_features.py
│   │   ├── ecg_features.py
│   │   ├── resp_features.py
│   │   ├── build_features.py
│   │   ├── build_all_features.py
│   │   └── build_chest_features.py
│   ├── data/
│   │   └── build_dataset.py
│   ├── models/
│   │   ├── random_forest_model.py
│   │   ├── model_factory.py
│   │   └── cnn_lstm_model.py
│   └── evaluation/
│       ├── loso.py
│       ├── loso_calibrated.py
│       ├── loso_calibrated_std.py
│       ├── loso_calibrated_std_floor.py
│       └── loso_deep.py
└── notebooks/
    └── lab_notebook.md
```

---

## Installation

### Requirements

- Python 3.11 (Anaconda recommended)
- Node.js 18+
- WESAD dataset (approximately 3 GB)
- Approximately 3 GB of disk space, in addition to any space needed for the environment and extracted files

### Step 1: Clone the repository

Replace `<repo-url>` with the actual repository URL.

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

Ensure `requirements.txt` exists at the project root, then run:

```bash
pip install -r requirements.txt
```

If your network requires an alternative package index, configure pip accordingly rather than hard-coding a mirror into the project instructions.

### Step 4: Download and place WESAD

Download the dataset from the [official WESAD distribution page](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html).

Extract the dataset and place the subject folders (`S2`, `S3`, ..., `S17`) in the location expected by the preprocessing script. The original development path was:

```text
E:\dataset\WESAD\WESAD\
```

If your path differs, update `WESAD_ROOT` in `src/preprocessing/wesad_preprocessing.py`.

### Step 5: Export the deployed models

Run these commands if the required processed data and experiment dependencies are available:

```bash
python -m scripts.export_model
python -m scripts.export_chest_model
```

The exported files should be placed in `models/deployed/`.

### Step 6: Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

---

## Usage

### Run the API

In Terminal 1:

```bash
conda activate emotion-ml
python -m api.app
```

The API should be available at `http://localhost:5000`.

### Run the frontend

In Terminal 2:

```bash
cd frontend
npm run dev
```

The Vite development server should be available at `http://localhost:5173`.

### Use the dashboard

The dashboard has three main panels:

1. **Host · Necklace** — a minimal pendant illustration with animated yellow waves indicating active transmission. Wave speed reflects the detected state.
2. **Guest · Wristband** — a minimal band with four navy-blue circles that pulse in size. Pulse rate reflects the haptic pattern for the detected state.
3. **Mock Subject** — a blueprint-style human figure with a **Generate mock** button that fills the form with randomized, physiologically plausible body conditions.

Below the panels is a body-condition form with seven inputs:

- Heart rate (bpm)
- HRV RMSSD (ms)
- HRV SDNN (ms)
- Respiratory rate (breaths/min)
- Respiratory energy
- Respiratory variability
- Skin temperature (°C)

Click **Predict state** to send the current condition to the API. The API calibrates the input against five baseline readings initialized on page load. These readings can be regenerated using the **Regenerate baseline** button. The response includes a predicted state and a probability distribution across the four classes.

### API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Server status and model metadata |
| `POST` | `/calibrate` | Compute baseline \(B_s\) and \(\sigma_s\) from baseline feature windows |
| `POST` | `/predict` | Classify one feature window using the wrist binary model |
| `POST` | `/predict-from-conditions` | Classify body conditions using the wrist binary model |
| `POST` | `/predict-affective-state` | Classify body conditions using the chest four-class model |
| `POST` | `/generate-haptic` | Map stress probability to a vibrotactile pattern |

For exact request and response schemas, consult the implementation in `api/app.py` and the API tests in `scripts/test_api.py`.

---

## Reproducing the experiments

The following commands reflect the experiment entry points listed for this project. Run them from the repository root in the activated `emotion-ml` environment. Some commands may require intermediate datasets or outputs generated by earlier steps.

```bash
# 1. Inspect the dataset and run baseline evaluation
python -m scripts.inspect_wesad
python -m scripts.run_loso

# 2. Model comparison
python -m scripts.compare_models

# 3. Feature ablation
python -m scripts.feature_ablation
python -m scripts.reduced_features

# 4. Modality evaluation
python -m scripts.compare_eda_temp
python -m scripts.temp_ablation
python -m scripts.compare_with_bvp
python -m scripts.bvp_ablation

# 5. Subject diagnosis
python -m scripts.diagnose_subjects
python -m scripts.diagnose_s3

# 6. Calibration
python -m scripts.compare_calibration
python -m scripts.compare_calibration_v2
python -m scripts.compare_calibration_floor

# 7. Window length
python -m scripts.chest_30s

# 8. Deep learning comparison
python -m scripts.compare_dl_vs_rf

# 9. Export deployed models
python -m scripts.export_model
python -m scripts.export_chest_model
```

All raw outputs are intended to be stored in `results/logs/`, distilled summaries in `results/metrics/`, and research notes in `notebooks/lab_notebook.md`.

**Reproducibility note:** Confirm that each script listed above exists in the repository. The experiment list includes `scripts.inspect_wesad` and `scripts.run_loso`, which are not shown in the repository tree supplied here. Add those files to the tree or update the commands if their actual names differ.

---

## Technologies

### Backend (Python)

| Library | Purpose |
|---|---|
| NumPy, SciPy | Numerical computation and signal processing |
| pandas | Data manipulation |
| scikit-learn | SVM, Random Forest, Logistic Regression, metrics, and pipelines |
| NeuroKit2 | ECG and respiration feature extraction |
| TensorFlow (CPU) | CNN-LSTM experiment |
| Flask, Flask-CORS | REST API |
| requests | API testing |

### Frontend (Node.js)

| Library | Purpose |
|---|---|
| React 19 | UI components |
| Vite | Development server and build tooling |
| Plain JavaScript | Frontend implementation; no TypeScript |
| Inter | Typography |

### Design system

- Background: `#FAFAF8` (warm off-white)
- Accent: `#00704A` (green)
- Haptic circles: `#1F3A5F` (navy)
- Transfer waves: `#E8B84B` (yellow)
- Text: `#1A1A1A` (near-black), `#6B6B6B` (muted)

---

## Limitations

### Data limitations

- The evaluation uses 15 WESAD subjects. This is a small sample, and per-subject variance is high.
- Only one dataset is used; generalization to other datasets has not been validated.
- The deployed models use wrist EDA + TEMP and chest ECG + RESP. Other signals, including chest EMG and accelerometry, remain unexplored in these deployed pipelines.

### Methodological limitations

- Calibration requires a known baseline period for each subject. In deployment, this is approximated using user-provided baseline readings.
- The sigma floors in the API are pragmatic safeguards against numerical instability; they are not necessarily identical to the statistics used in the training runs.
- S3 (high baseline variance) and S14 (electrodermal non-responder) remain imperfectly handled.
- Amusement F1 is 0.56. Peripheral physiological signals may capture arousal more directly than valence, which can limit separation between amusement and stress.
- The body-condition-to-feature conversion used by the deployment-facing model is heuristic. It should not be interpreted as a substitute for direct sensor measurements, and the UI should clearly label simulated or inferred inputs.

### Transfer limitations

- The haptic mapping is a proposed design informed by the affective haptics literature. It has **not** been validated to induce a specific emotional state in a receiver.
- The dashboard is a software prototype. Physical sensors, wireless hardware, and haptic actuators have not yet been integrated.

---

## Future work

1. **Validate the haptic transfer** in a controlled study that measures whether the proposed patterns produce measurable affective changes in receivers.
2. **Add phasic EDA features** through cvxEDA decomposition, including skin conductance response count, rise time, and recovery time.
3. **Add chest ECG-derived HRV features to the wrist pipeline** to test whether combined features improve either task.
4. **Explore an EEG headset** for valence-sensitive classification. Peripheral signals alone may not sufficiently distinguish affective states with similar arousal but different valence.
5. **Investigate class weighting and other imbalance strategies** for amusement and meditation, then validate any improvements using subject-independent evaluation.
6. **Integrate physical hardware** to replace the browser simulation with real wristband sensors and haptic actuators.

---

## References

### Dataset

Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K. (2018). *Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection.* Proceedings of the 2018 ACM International Conference on Multimodal Interaction (ICMI).

https://doi.org/10.1145/3242969.3242985

### Calibration

*Personalized Electrocardiographic and HRV Dynamics for Acute Stress Detection: A Leave-One-Subject-Out Benchmark on WESAD.* Zenodo preprint, 2026.

https://zenodo.org/records/22806710

Akkaya, A. (2026). *Calibration, not architecture, limits cross-subject wearable stress detection.* BMC Medical Informatics and Decision Making.

### Feature selection

Guyon, I., & Elisseeff, A. (2003). *An Introduction to Variable and Feature Selection.* Journal of Machine Learning Research, 3, 1157–1182.

https://jmlr.org/papers/v3/guyon03a.html

### Statistics

Wilcoxon, F. (1945). *Individual Comparisons by Ranking Methods.* Biometrics Bulletin, 1(6), 80–83.

Demšar, J. (2006). *Statistical Comparisons of Classifiers over Multiple Data Sets.* Journal of Machine Learning Research, 7, 1–30.

https://www.jmlr.org/papers/v7/demsar06a.html

### Electrodermal activity

Boucsein, W. (2012). *Electrodermal Activity* (2nd ed.). Springer.

https://link.springer.com/book/10.1007/978-1-4614-1126-0

### Affective haptics

Eid, M., & Al Osman, H. (2016). *Affective Haptics: Current Research and Future Directions.* IEEE Access, 4, 26–49.

https://doi.org/10.1109/ACCESS.2016.2527305

### Respiration and affect

*State-specific respiratory signatures for affective and stress recognition.* ScienceDirect, 2026. **Verify the full bibliographic details and publication link before citing this reference in a paper.**

### Photoplethysmography

Charlton, P. H., et al. (2022). *Wearable Photoplethysmography for Cardiovascular Monitoring.* Frontiers in Digital Health, 4, 840840.

https://doi.org/10.3389/fdgth.2022.840840

---

## Author

**Mojtaba Safari**

AffectWave is a research prototype exploring the intersection of wearable sensing, machine learning, and affective haptics. The project is intended for research and educational use.

---

## License

The project code is released for research and educational purposes. The WESAD dataset is subject to its own license terms; consult the [WESAD distribution page](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html) for details.

**Last updated:** October 2026
