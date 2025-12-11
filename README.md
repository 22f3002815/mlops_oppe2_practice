# 🚀 Fraud Detection MLOps Pipeline – Full Assignment Implementation

**Author:** Yuvraj
**Environment:** Google Cloud Platform (GCP) – Jupyter Notebook VM
**Model:** XGBoost (production), fallback pipelines for SHAP/Fairness/Drift
**Dataset:** Credit Card Fraud Detection (transactions.csv)

---

# 📌 Overview

This project demonstrates a **complete MLOps workflow** for a real-time fraud detection system deployed on Google Cloud. It implements modern practices spanning the entire ML lifecycle:

✔ Model Training
✔ CI/CD with GitHub Actions
✔ Docker-based Model Serving API
✔ Deployment on GKE + Autoscaling
✔ Load Testing & Observability
✔ Data Poisoning Attack Simulation
✔ Explainability (SHAP)
✔ Fairness Analysis
✔ Concept Drift Detection

Each requirement from the assignment has been **fully satisfied** and documented below.

---

# 🧭 Project Structure

```
├── scripts/
│   ├── add_location.py
│   ├── train_explain_fair_monitor.py
│   ├── poison_data.py
│   ├── train_poisoned_models.py
├── app/
│   ├── main.py (FastAPI app)
│   ├── model/model.pkl
│   ├── requirements.txt
├── Dockerfile
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── hpa.yaml
├── locustfile.py
├── .github/
│   └── workflows/
│       └── cd.yml
├── data/
│   ├── v0/
│   ├── v1/
├── dvc.yaml
├── README.md
```

---

# 🧩 Task 1 — CI/CD & API Containerization

### ✅ 1. FastAPI Prediction Service

* Implemented `/predict` endpoint that accepts JSON payload with transaction features.
* Returns prediction + probability.
* Model loaded from serialized XGBoost file.

**Key File:** `api/main.py`

---

### ✅ 2. Dockerization

* Dockerfile builds lightweight Python image
* Installs FastAPI + Uvicorn + model dependencies
* Exposes port `8080`

**Key File:** `Dockerfile`

---

### ✅ 3. GitHub Actions CI/CD → Google Artifact Registry

Workflow performs:

1. Trigger on push to `main`
2. Build Docker image
3. Authenticate to Google Cloud via Workload Identity Federation
4. Push to Artifact Registry
5. Post CML report back to GitHub PR

**Key File:** `.github/workflows/cd.yml`

---

# 🧩 Task 2 — Deployment, Orchestration & Scaling

### ✅ 1. Deploy FastAPI Service on GKE

Deployment object pulls image from Artifact Registry + runs Uvicorn.

**Key File:**

* `kubernetes/deployment.yaml`

---

### ✅ 2. Expose with LoadBalancer

Creates external endpoint accessible via GCP Load Balancer.

**Key File:**

* `kubernetes/service.yaml`

---

### ✅ 3. Autoscaling with HPA

Horizontal Pod Autoscaler automatically scales pods based on CPU usage.

**Key File:**

* `kubernetes/hpa.yaml`

---

### ✅ 4. Load Testing with Locust

Simulates concurrent `/predict` requests.

**Key File:**

* `locust/locustfile.py`

Observations logged:

* Increased RPS → HPA scales from 1 pod → 3 pods
* Latency monitored via Locust UI in real-time

---

### ✅ 5. OpenTelemetry Instrumentation

Added:

* Custom tracing span `"model_predict_time"`
* Tracks latency of `model.predict()`

This allows GKE observability dashboards to show model-level latency.

---

# 🧩 Task 3 — MLSecurityOps: Data Poisoning Attack Simulation

### ✅ 1. Generate Poisoned Datasets

Created 3 poisoned versions of v0 dataset:

| File                      | % Flipped                  |
| ------------------------- | -------------------------- |
| `poisoned_2_percent.csv`  | 2% of class 0 flipped to 1 |
| `poisoned_8_percent.csv`  | 8% flipped                 |
| `poisoned_20_percent.csv` | 20% flipped                |

**Script:** `scripts/poison_data.py`

---

### ✅ 2. Versioning with DVC + GCS Remote

* Added poisoned datasets
* Tracked using DVC
* Stored remotely on GCS bucket

---

### ✅ 3. MLflow Experiment Tracking

For each poisoning level, trained separate models.

Logged to MLflow:

* `poisoning_level = 2 | 8 | 20`
* F1-scores
* Model artifacts

**Script:** `scripts/train_poisoned_models.py`

Observations:

* Higher poisoning → F1 drops significantly
* Demonstrates vulnerability of fraud model to label corruption

---

# 🧩 Task 4 — Explainability, Fairness & Monitoring

## 4.1 Introduce Sensitive Attribute

### ✅ Added synthetic `"location"` column

Randomly assigned:

* `Location_A`
* `Location_B`

**Script:** `scripts/add_location.py`

---

## 4.2 Explain Predictions (SHAP)

### ✅ Model retrained with `location` included

### ✅ SHAP beeswarm summary plot generated

File saved as:

```
artifacts/shap_summary.png
```

Logged to MLflow → `explainability/`

---

## 4.3 Fairness Audit

Using Fairlearn:

* Computed `demographic_parity_difference`
* Logged to MLflow

**Result:** Very low bias (≈ 0.0011)

---

## 4.4 Concept Drift Detection

### Steps performed:

1. Model trained on v0 dataset
2. Predictions generated on v1 dataset
3. Logged F1, precision, recall
4. Drift comparison plot generated:

```
artifacts/drift_comparison.png
```

Logged under: `monitoring/` in MLflow

### Observations:

* F1_v0_val = 0.6667
* F1_v1_full = 0.6667
  → Very low drift between years

---

# 📊 MLflow Artifacts Summary

| Artifact                     | Purpose                  |
| ---------------------------- | ------------------------ |
| `shap_summary.png`           | Global explainability    |
| `drift_comparison.png`       | Drift monitoring         |
| `fairness.txt`               | Bias metrics summary     |
| `model/`                     | Serialized model         |
| `model_raw/xgb_model.joblib` | Raw model for deployment |

---

# 🛠 Technologies Used

### **MLOps**

* DVC (Data versioning)
* MLflow (Experiment tracking)
* GitHub Actions (CI/CD)
* CML (PR reporting)
* Docker + Artifact Registry
* GKE + HPA autoscaling

### **Modeling**

* XGBoost
* SHAP
* Fairlearn

### **Monitoring**

* OpenTelemetry
* Locust Load Testing

---

# 🧾 Final Notes

This project demonstrates a complete real-world MLOps production pipeline applied to a fraud detection use-case.
All tasks (1–4) have been **implemented, tested, logged, and deployed** on Google Cloud.