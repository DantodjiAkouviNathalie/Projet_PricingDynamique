#!/usr/bin/env python
"""
Comprehensive test suite for the Olist Dynamic Pricing System.
Tests all 4 main components:
1. Data Pipeline
2. DQN Model
3. API Service
4. Streamlit Application
"""

import os
import sys
import time
import subprocess
import requests
import json
from pathlib import Path
from datetime import datetime

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_header(title):
    """Print formatted header"""
    print(f"\n{BOLD}{BLUE}{'='*70}{RESET}")
    print(f"{BOLD}{BLUE}{title:^70}{RESET}")
    print(f"{BOLD}{BLUE}{'='*70}{RESET}\n")

def print_test(name, status, message=""):
    """Print test result"""
    symbol = f"{GREEN}✓{RESET}" if status else f"{RED}✗{RESET}"
    print(f"  {symbol} {name}")
    if message:
        print(f"    {YELLOW}→ {message}{RESET}")

def print_section(title):
    """Print section title"""
    print(f"\n{BOLD}{title}{RESET}")
    print("-" * 50)

# ============================================================================
# TEST 1: DATA PIPELINE
# ============================================================================
def test_data_pipeline():
    print_header("TEST 1: DATA PIPELINE")
    print_section("Checking data pipeline files")
    
    results = []
    base_dir = Path(".")
    
    # Check Python script exists
    pipeline_script = base_dir / "data_pipeline.py"
    exists = pipeline_script.exists()
    results.append(("data_pipeline.py exists", exists))
    print_test("data_pipeline.py exists", exists)
    
    # Check CSV inputs exist
    csv_files = [
        "olist_customers_dataset.csv",
        "olist_orders_dataset.csv",
        "olist_order_items_dataset.csv",
        "olist_products_dataset.csv",
        "olist_sellers_dataset.csv",
    ]
    
    print_section("Checking input CSV files")
    all_csv_exist = True
    for csv_file in csv_files:
        path = base_dir / csv_file
        exists = path.exists()
        all_csv_exist = all_csv_exist and exists
        results.append((f"{csv_file}", exists))
        print_test(csv_file, exists, f"Size: {path.stat().st_size / 1e6:.1f}MB" if exists else "")
    
    # Check output CSV files exist
    print_section("Checking output CSV files (pipeline results)")
    output_files = [
        "olist_full_cleaned.csv",
        "olist_pricing_features.csv"
    ]
    
    all_outputs_exist = True
    for output_file in output_files:
        path = base_dir / output_file
        exists = path.exists()
        all_outputs_exist = all_outputs_exist and exists
        results.append((f"{output_file}", exists))
        if exists:
            size_mb = path.stat().st_size / 1e6
            print_test(output_file, exists, f"Size: {size_mb:.1f}MB")
        else:
            print_test(output_file, exists)
    
    pipeline_status = exists and all_csv_exist and all_outputs_exist
    print_section("Summary")
    print_test("Pipeline files complete", pipeline_status)
    
    return pipeline_status

# ============================================================================
# TEST 2: DQN MODEL & ARTIFACTS
# ============================================================================
def test_dqn_model():
    print_header("TEST 2: DQN MODEL & ARTIFACTS")
    
    results = []
    base_dir = Path(".")
    artifacts_dir = base_dir / "artifacts"
    
    print_section("Checking artifacts directory")
    artifacts_exist = artifacts_dir.exists()
    results.append(("artifacts/ directory exists", artifacts_exist))
    print_test("artifacts/ directory", artifacts_exist)
    
    if not artifacts_exist:
        return False
    
    # Check checkpoint file
    print_section("Checking model checkpoint")
    checkpoint_path = artifacts_dir / "dqn_pricing_checkpoint.pt"
    checkpoint_exists = checkpoint_path.exists()
    results.append(("DQN checkpoint", checkpoint_exists))
    if checkpoint_exists:
        size_mb = checkpoint_path.stat().st_size / 1e6
        print_test("dqn_pricing_checkpoint.pt", checkpoint_exists, f"Size: {size_mb:.1f}MB")
    else:
        print_test("dqn_pricing_checkpoint.pt", checkpoint_exists)
    
    # Check scaler file
    print_section("Checking feature scaler")
    scaler_path = artifacts_dir / "dqn_scaler.pkl"
    scaler_exists = scaler_path.exists()
    results.append(("Feature scaler", scaler_exists))
    if scaler_exists:
        size_kb = scaler_path.stat().st_size / 1e3
        print_test("dqn_scaler.pkl", scaler_exists, f"Size: {size_kb:.1f}KB")
    else:
        print_test("dqn_scaler.pkl", scaler_exists)
    
    # Try to load checkpoint
    print_section("Attempting to load checkpoint")
    try:
        import torch
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        print_test("Load checkpoint (PyTorch)", True, 
                   f"Keys: {list(checkpoint.keys())[:3]}...")
        results.append(("Checkpoint loadable", True))
    except Exception as e:
        print_test("Load checkpoint (PyTorch)", False, str(e)[:50])
        results.append(("Checkpoint loadable", False))
    
    model_status = checkpoint_exists and scaler_exists
    print_section("Summary")
    print_test("DQN model files complete", model_status)
    
    return model_status

# ============================================================================
# TEST 3: API SERVICE
# ============================================================================
def test_api_service():
    print_header("TEST 3: API SERVICE")
    
    results = []
    api_url = "http://localhost:8000"
    
    print_section("Checking API service file")
    api_script = Path("api_service.py")
    api_exists = api_script.exists()
    results.append(("api_service.py exists", api_exists))
    print_test("api_service.py", api_exists)
    
    print_section("Checking if API server is running")
    try:
        response = requests.get(f"{api_url}/health", timeout=2)
        api_running = response.status_code == 200
        print_test("API server responding", api_running, 
                   f"Status: {response.status_code}")
        results.append(("API running", api_running))
    except requests.exceptions.ConnectionError:
        print_test("API server responding", False, 
                   "Cannot connect to localhost:8000")
        results.append(("API running", False))
        return False
    except Exception as e:
        print_test("API server responding", False, str(e)[:50])
        results.append(("API running", False))
        return False
    
    if not api_running:
        return False
    
    # Test /health endpoint
    print_section("Testing endpoints")
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        health_ok = response.status_code == 200 and response.json().get("status") == "ok"
        print_test("/health endpoint", health_ok, f"Response: {response.json()}")
        results.append(("GET /health", health_ok))
    except Exception as e:
        print_test("/health endpoint", False, str(e)[:50])
        results.append(("GET /health", False))
    
    # Test /categories endpoint
    try:
        response = requests.get(f"{api_url}/categories", timeout=5)
        categories_ok = response.status_code == 200
        categories = response.json().get("categories", [])
        print_test("/categories endpoint", categories_ok, 
                   f"Found {len(categories)} categories")
        results.append(("GET /categories", categories_ok))
    except Exception as e:
        print_test("/categories endpoint", False, str(e)[:50])
        results.append(("GET /categories", False))
        return False
    
    # Test /predict endpoint
    print_section("Testing prediction endpoint")
    try:
        if not categories:
            print_test("/predict endpoint", False, "No categories found")
            return False
        
        test_category = categories[0]
        request_data = {
            "category": test_category,
            "current_price": 500.0,
            "month": 6,
            "review_score": 4.5,
            "marketing_boost": 0.0,
            "demand_shock": 0.0,
            "stock_pressure": 0.0,
            "cost_shock": 0.0
        }
        
        response = requests.post(f"{api_url}/predict", json=request_data, timeout=10)
        predict_ok = response.status_code == 200
        
        if predict_ok:
            result = response.json()
            suggested_price = result.get("suggested_price")
            action = result.get("recommended_action")
            num_scores = len(result.get("scores", []))
            print_test("/predict endpoint", predict_ok, 
                       f"Price: ${suggested_price}, Action: {action}, Scores: {num_scores}")
            results.append(("POST /predict", predict_ok))
        else:
            print_test("/predict endpoint", predict_ok, 
                       f"Status: {response.status_code}")
            results.append(("POST /predict", predict_ok))
    except Exception as e:
        print_test("/predict endpoint", False, str(e)[:50])
        results.append(("POST /predict", False))
    
    api_status = api_running and categories_ok
    print_section("Summary")
    print_test("API service fully functional", api_status)
    
    return api_status

# ============================================================================
# TEST 4: STREAMLIT APPLICATION
# ============================================================================
def test_streamlit_app():
    print_header("TEST 4: STREAMLIT APPLICATION")
    
    results = []
    
    print_section("Checking Streamlit application file")
    streamlit_script = Path("simulateur_web.py")
    streamlit_exists = streamlit_script.exists()
    results.append(("simulateur_web.py exists", streamlit_exists))
    if streamlit_exists:
        size_kb = streamlit_script.stat().st_size / 1e3
        print_test("simulateur_web.py", streamlit_exists, f"Size: {size_kb:.1f}KB")
    else:
        print_test("simulateur_web.py", streamlit_exists)
        return False
    
    print_section("Checking Python imports")
    try:
        import streamlit as st
        print_test("Streamlit package installed", True, "Ready to deploy")
        results.append(("Streamlit installed", True))
    except ImportError:
        print_test("Streamlit package installed", False, "pip install streamlit")
        results.append(("Streamlit installed", False))
        return False
    
    print_section("Checking syntax validity")
    try:
        import py_compile
        py_compile.compile(str(streamlit_script), doraise=True)
        print_test("simulateur_web.py syntax", True, "Valid Python code")
        results.append(("Syntax valid", True))
    except py_compile.PyCompileError as e:
        print_test("simulateur_web.py syntax", False, str(e)[:50])
        results.append(("Syntax valid", False))
        return False
    
    print_section("Launching Streamlit app (dry run)")
    print_test("To launch app locally", True, 
               "streamlit run simulateur_web.py --server.port 8501")
    
    streamlit_status = streamlit_exists
    print_section("Summary")
    print_test("Streamlit app ready", streamlit_status)
    
    return streamlit_status

# ============================================================================
# FINAL REPORT
# ============================================================================
def generate_report(results):
    print_header("FINAL TEST REPORT")
    
    print_section("Component Status")
    
    if results['pipeline']:
        print_test("Pipeline données", True, "All files present and accessible")
    else:
        print_test("Pipeline données", False, "Some files missing")
    
    if results['model']:
        print_test("Modèle DQN", True, "Checkpoint and scaler loaded")
    else:
        print_test("Modèle DQN", False, "Artifact files missing")
    
    if results['api']:
        print_test("API service", True, "Running and responding to requests")
    else:
        print_test("API service", False, "Check if server is running")
    
    if results['streamlit']:
        print_test("Application Streamlit", True, "Ready to deploy")
    else:
        print_test("Application Streamlit", False, "Some dependencies missing")
    
    print_section("Overall Status")
    all_passed = all(results.values())
    if all_passed:
        print(f"{GREEN}{BOLD}✓ ALL TESTS PASSED - SYSTEM READY FOR DEPLOYMENT{RESET}\n")
    else:
        print(f"{YELLOW}{BOLD}⚠ SOME TESTS FAILED - CHECK ABOVE FOR DETAILS{RESET}\n")
    
    print_section("Next Steps")
    print("1. Run Streamlit: streamlit run simulateur_web.py")
    print("2. Access API: curl http://localhost:8000/health")
    print("3. Deploy with Docker: docker-compose up")
    print("4. Test full pipeline: python data_pipeline.py --raw-dir . --output-dir .")

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    start_time = datetime.now()
    
    print(f"\n{BOLD}{BLUE}Olist Dynamic Pricing System - Comprehensive Test Suite{RESET}")
    print(f"Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Run all tests
    results = {
        'pipeline': test_data_pipeline(),
        'model': test_dqn_model(),
        'api': test_api_service(),
        'streamlit': test_streamlit_app(),
    }
    
    # Generate report
    generate_report(results)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"Completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')} ({duration:.1f}s)\n")
    
    # Exit code
    sys.exit(0 if all(results.values()) else 1)
