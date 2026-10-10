"""
Test the Flask API endpoints.

Run:
    python scripts/test_api.py

Requires the Flask server to be running at http://localhost:5000
and the `requests` library to be installed.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import requests


BASE = "http://localhost:5000"


def main():
    print("=" * 70)
    print("API TEST")
    print("=" * 70)

    # 1. Health
    print("\n1. GET /health")
    r = requests.get(f"{BASE}/health")
    print(f"   status: {r.status_code}")
    print(f"   body:   {json.dumps(r.json(), indent=2)}")

    # 2. Calibrate
    print("\n2. POST /calibrate")
    baseline = [
        [0.30, 0.050, 0.0050, 33.00],
        [0.31, 0.052, 0.0051, 33.10],
        [0.29, 0.048, 0.0049, 32.90],
        [0.30, 0.051, 0.0050, 33.00],
        [0.30, 0.050, 0.0050, 33.00],
    ]
    r = requests.post(f"{BASE}/calibrate",
                      json={"baseline_features": baseline})
    print(f"   status: {r.status_code}")
    print(f"   body:   {json.dumps(r.json(), indent=2)}")
    cal = r.json()

    # 3. Predict (with calibration)
    print("\n3. POST /predict (calibrated)")
    r = requests.post(f"{BASE}/predict", json={
        "features": [0.5, 0.1, 0.01, 33.1],
        "calibration": {
            "B_s": cal["B_s"],
            "sigma_s": cal["sigma_s"],
        },
    })
    print(f"   status: {r.status_code}")
    print(f"   body:   {json.dumps(r.json(), indent=2)}")

    # 4. Predict (a "baseline-like" input to see the low band)
    print("\n4. POST /predict (baseline-like features)")
    r = requests.post(f"{BASE}/predict", json={
        "features": [0.30, 0.050, 0.0050, 33.00],
        "calibration": {
            "B_s": cal["B_s"],
            "sigma_s": cal["sigma_s"],
        },
    })
    print(f"   status: {r.status_code}")
    print(f"   body:   {json.dumps(r.json(), indent=2)}")

    # 5. Generate haptic
    print("\n5. POST /generate-haptic (p=0.8)")
    r = requests.post(f"{BASE}/generate-haptic",
                      json={"stress_probability": 0.8})
    print(f"   status: {r.status_code}")
    print(f"   body:   {json.dumps(r.json(), indent=2)}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()