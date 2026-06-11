# 🎯 OLIST Dynamic Pricing System - Complete Testing Guide

## 📋 Overview

This document explains how to test all components of the Olist Dynamic Pricing System:

1. **✅ Pipeline données** - Data preparation ETL
2. **✅ Modèle DQN** - Deep Q-Network pricing model
3. **✅ API service** - FastAPI REST service
4. **✅ Application Streamlit** - Interactive web interface

---

## 🚀 Quick Start

### 1️⃣ Run All Tests at Once

```bash
python test_all_components.py
```

This runs comprehensive checks on all 4 components and outputs a detailed report.

**Expected Output:**
```
✓ Pipeline données - All files present and accessible
✓ Modèle DQN - Checkpoint and scaler loaded
✓ API service - Running and responding to requests
✓ Application Streamlit - Ready to deploy

✓ ALL TESTS PASSED - SYSTEM READY FOR DEPLOYMENT
```

### 2️⃣ Interactive Testing (Component by Component)

```bash
python test_interactive.py
```

Menu-driven interface to test each component individually with:
- Component descriptions
- Available sample commands
- Interactive execution
- Detailed output inspection

---

## 🧪 Component Testing Details

### TEST 1: Data Pipeline

**Location:** `data_pipeline.py`

**What it does:**
- Loads 9 raw CSV files (90MB total)
- Consolidates data with multiple joins
- Cleans missing values, duplicates
- Extracts pricing features (seasonality, elasticity, basket metrics)
- Outputs 2 processed datasets

**Files Generated:**
- `olist_full_cleaned.csv` (52 MB) - Complete consolidated dataset
- `olist_pricing_features.csv` (39 MB) - Features for DQN model

**Test it:**
```bash
# Run full pipeline
python data_pipeline.py --raw-dir . --output-dir . --cleaned-filename olist_full_cleaned.csv --features-filename olist_pricing_features.csv

# Inspect output
python -c "import pandas as pd; df = pd.read_csv('olist_full_cleaned.csv'); print(f'Rows: {len(df)}, Columns: {len(df.columns)}')"
```

---

### TEST 2: DQN Model & Artifacts

**Location:** `artifacts/`

**Files:**
- `dqn_pricing_checkpoint.pt` (1.1 MB) - Trained PyTorch model
- `dqn_scaler.pkl` (0.8 KB) - Feature normalization scaler

**What the model does:**
- Takes 8 market condition inputs
- Recommends 1 of 5 pricing actions (-30%, -15%, 0%, +15%, +30%)
- Scores all actions based on Q-values
- Outputs expected demand, profit, satisfaction metrics

**Test it:**
```bash
# Load checkpoint
python -c "import torch; ckpt = torch.load('artifacts/dqn_pricing_checkpoint.pt', map_location='cpu'); print('Keys:', list(ckpt.keys()))"

# Load scaler
python -c "import pickle; scaler = pickle.load(open('artifacts/dqn_scaler.pkl', 'rb')); print('Scaler loaded:', type(scaler))"
```

---

### TEST 3: API Service

**Location:** `api_service.py`

**Server:** `http://localhost:8000`

**Endpoints:**

#### GET /health
```bash
curl http://localhost:8000/health
# Response: {"status": "ok"}
```

#### GET /categories
```bash
curl http://localhost:8000/categories
# Response: {"categories": ["agro_industria_e_comercio", "alimentos", ...]}
```

#### POST /predict
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "category": "informatica",
    "current_price": 500.0,
    "month": 6,
    "review_score": 4.5,
    "marketing_boost": 0.0,
    "demand_shock": 0.0,
    "stock_pressure": 0.0,
    "cost_shock": 0.0
  }'
```

**Response:**
```json
{
  "suggested_price": 500.0,
  "recommended_action": "MAINTAIN",
  "reward_proxy": 97.2454,
  "scores": [
    {
      "delta_pct": 0.0,
      "expected_orders": 1.4,
      "gross_profit": 597.66,
      "churn_risk": 0.015,
      "satisfaction": 4.45
    },
    ...
  ]
}
```

**Test it:**
```bash
# Start API (if not already running)
python run_api.py

# In another terminal:
python test_api_full.py
```

---

### TEST 4: Streamlit Application

**Location:** `simulateur_web.py`

**What it does:**
- Interactive UI for pricing simulation
- Category selection with 73 options
- Real-time DQN predictions
- Performance visualization

**Test it:**
```bash
# Start Streamlit app
streamlit run simulateur_web.py --server.port 8501

# Open browser to: http://localhost:8501
```

**Features:**
- 🎯 Select product category
- 💰 Set current price, review score, costs
- 📊 See DQN recommendation
- 📈 View action scores comparison
- 🔍 Analyze demand elasticity

---

## 📊 Component Status Check

### Quick Health Check

```bash
# Check all files exist
ls -lh data_pipeline.py api_service.py simulateur_web.py artifacts/

# Check API is running
curl -s http://localhost:8000/health | python -m json.tool

# Verify dependencies
python -c "import pandas, torch, fastapi, streamlit; print('✓ All imports OK')"
```

---

## 🔧 Troubleshooting

### API not responding?
```bash
# Check if running
netstat -an | grep 8000

# Restart it
python run_api.py
```

### Import errors?
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### Data pipeline fails?
```bash
# Verify CSV files exist
ls olist_*.csv

# Check permissions
ls -l *.csv
```

---

## 📈 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│        Olist Dynamic Pricing System                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  INPUT: 9 Raw CSVs (90 MB)                             │
│    ↓                                                    │
│  [1] Data Pipeline (data_pipeline.py)                  │
│    ↓                                                    │
│  INTERMEDIATE: 2 Processed CSVs (91 MB)                │
│    ↓                                                    │
│  [2] DQN Model (artifacts/checkpoint + scaler)         │
│    ↓ (loaded at startup)                               │
│  [3] API Service (api_service.py @ :8000)              │
│    ↓                                                    │
│  [4] Streamlit App (simulateur_web.py @ :8501)         │
│    ↓                                                    │
│  OUTPUT: Pricing Recommendations (Real-time)           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## ✅ Validation Checklist

- [ ] All 9 input CSV files present (90 MB)
- [ ] `olist_full_cleaned.csv` exists (52 MB)
- [ ] `olist_pricing_features.csv` exists (39 MB)
- [ ] `artifacts/dqn_pricing_checkpoint.pt` exists (1.1 MB)
- [ ] `artifacts/dqn_scaler.pkl` exists (0.8 KB)
- [ ] API responds to `/health` endpoint
- [ ] API returns 73 categories
- [ ] API `/predict` endpoint returns valid JSON with 5 actions
- [ ] Streamlit app syntax is valid
- [ ] All Python dependencies installed

---

## 🐳 Docker Deployment

Coming soon:
- `Dockerfile` - Containerize API service
- `docker-compose.yml` - Multi-container setup (API + Streamlit)

Run with: `docker-compose up`

---

## 📞 Support

For detailed information about each component:
- Data Pipeline: See `data_pipeline.py` comments
- API Service: See `api_service.py` comments
- Streamlit App: See `simulateur_web.py` comments
- Notebook Reference: See `chargement_datasets.ipynb`

---

*Last updated: 2026-06-10*
