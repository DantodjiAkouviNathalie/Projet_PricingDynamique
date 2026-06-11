#!/usr/bin/env python
"""
Interactive testing guide for all components.
Demonstrates how to use each part of the system.
"""

import subprocess
import sys
from pathlib import Path

def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def run_command(cmd, description):
    """Display and optionally run a command"""
    print(f"📌 {description}")
    print(f"   Command: {cmd}\n")
    response = input("   Run this command? (y/n): ").strip().lower()
    if response == 'y':
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            print(f"\n{result.stdout}")
            if result.stderr:
                print(f"Errors: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("⚠ Command timed out (>30s)")
        except Exception as e:
            print(f"❌ Error: {e}")
    print("\n" + "-"*70)

def main():
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "OLIST DYNAMIC PRICING SYSTEM" + " "*25 + "║")
    print("║" + " "*18 + "Component Testing Guide" + " "*27 + "║")
    print("╚" + "="*68 + "╝")
    
    while True:
        print_section("MAIN MENU")
        print("1. Test Data Pipeline")
        print("2. Test DQN Model")
        print("3. Test API Service")
        print("4. Test Streamlit Application")
        print("5. Run Full System Test")
        print("6. Exit\n")
        
        choice = input("Select option (1-6): ").strip()
        
        if choice == '1':
            test_pipeline()
        elif choice == '2':
            test_model()
        elif choice == '3':
            test_api()
        elif choice == '4':
            test_streamlit()
        elif choice == '5':
            print("\n✅ Running comprehensive test suite...\n")
            subprocess.run([sys.executable, "test_all_components.py"])
        elif choice == '6':
            print("\n👋 Goodbye!\n")
            break
        else:
            print("❌ Invalid choice. Try again.\n")

def test_pipeline():
    print_section("1. DATA PIPELINE TESTING")
    
    print("""
The data pipeline does 3 main things:
  • Loads 9 raw CSV files
  • Consolidates and cleans data
  • Extracts pricing features
    """)
    
    print("Current outputs:")
    print("  ✓ olist_full_cleaned.csv (52.0 MB)")
    print("  ✓ olist_pricing_features.csv (39.2 MB)\n")
    
    run_command(
        "python data_pipeline.py --raw-dir . --output-dir . --cleaned-filename olist_full_cleaned.csv --features-filename olist_pricing_features.csv",
        "Run data pipeline (full processing)"
    )
    
    run_command(
        "python -c \"import pandas as pd; df = pd.read_csv('olist_full_cleaned.csv'); print(f'Loaded {len(df):,} rows with {len(df.columns)} columns'); print('\\nColumns:', df.columns.tolist()[:10])\"",
        "Inspect cleaned dataset"
    )
    
    run_command(
        "python -c \"import pandas as pd; df = pd.read_csv('olist_pricing_features.csv'); print(f'Features: {len(df)} rows'); print('\\nFeature columns:', [c for c in df.columns if 'mois' in c or 'elasticite' in c or 'panier' in c])\"",
        "Check extracted pricing features"
    )

def test_model():
    print_section("2. DQN MODEL TESTING")
    
    print("""
The DQN model does dynamic pricing decisions:
  • Takes market conditions as input
  • Recommends one of 5 pricing actions
  • Scores all actions
    """)
    
    print("Model artifacts:")
    print("  ✓ artifacts/dqn_pricing_checkpoint.pt (1.1 MB)")
    print("  ✓ artifacts/dqn_scaler.pkl (0.8 KB)\n")
    
    run_command(
        "python -c \"import torch; from pathlib import Path; ckpt = torch.load('artifacts/dqn_pricing_checkpoint.pt', map_location='cpu'); print('Checkpoint keys:'); [print(f'  - {k}') for k in ckpt.keys()]\"",
        "Inspect checkpoint structure"
    )
    
    run_command(
        "python -c \"import pickle; from pathlib import Path; scaler = pickle.load(open('artifacts/dqn_scaler.pkl', 'rb')); print(f'Scaler type: {type(scaler).__name__}'); print(f'Scaler parameters: {scaler.get_params() if hasattr(scaler, \\\"get_params\\\") else \\\"N/A\\\"}')\"",
        "Inspect feature scaler"
    )

def test_api():
    print_section("3. API SERVICE TESTING")
    
    print("""
The API exposes the DQN model via REST endpoints:
  • GET /health - Service health check
  • GET /categories - List available product categories
  • POST /predict - Get pricing recommendations
    """)
    
    print("API Status: http://localhost:8000\n")
    
    run_command(
        "python -c \"import requests; r = requests.get('http://localhost:8000/health'); print(r.json())\"",
        "Test /health endpoint"
    )
    
    run_command(
        "python -c \"import requests; r = requests.get('http://localhost:8000/categories'); cats = r.json()['categories']; print(f'Found {len(cats)} categories'); print('Sample:', cats[:5])\"",
        "Test /categories endpoint"
    )
    
    run_command(
        """python -c \"
import requests, json

data = {
    'category': 'informatica',
    'current_price': 500.0,
    'month': 6,
    'review_score': 4.5,
    'marketing_boost': 0.0,
    'demand_shock': 0.0,
    'stock_pressure': 0.0,
    'cost_shock': 0.0
}

r = requests.post('http://localhost:8000/predict', json=data)
result = r.json()
print(f'Status: {r.status_code}')
print(f'Suggested Price: \${result.get(\\\"suggested_price\\\")}')
print(f'Recommended Action: {result.get(\\\"recommended_action\\\")}')
print(f'Reward Score: {result.get(\\\"reward_proxy\\\"):.2f}')
print('\\nAll 5 pricing actions:')
for i, score in enumerate(result['scores']):
    print(f'  Action {i}: {score[\\\"delta_pct\\\"]:+.0f}% → Profit: \${score[\\\"gross_profit\\\"]:,.2f}')
\"\"",
        "Test /predict endpoint with sample data"
    )

def test_streamlit():
    print_section("4. STREAMLIT APPLICATION TESTING")
    
    print("""
The Streamlit app provides an interactive interface:
  • Product category selection
  • Dynamic pricing simulation
  • Real-time model predictions
  • Performance analytics
    """)
    
    print("Application file: simulateur_web.py (35.7 KB)\n")
    
    run_command(
        "python -m py_compile simulateur_web.py && echo '✓ Syntax valid'",
        "Validate Streamlit app syntax"
    )
    
    print("Launch the Streamlit app:")
    print("   streamlit run simulateur_web.py --server.port 8501\n")
    print("Then open your browser to:")
    print("   http://localhost:8501\n")
    
    response = input("Launch Streamlit now? (y/n): ").strip().lower()
    if response == 'y':
        try:
            subprocess.run(["streamlit", "run", "simulateur_web.py", "--server.port", "8501"])
        except FileNotFoundError:
            print("❌ Streamlit not found. Install with: pip install streamlit")
        except KeyboardInterrupt:
            print("\n✓ Streamlit stopped")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Exiting...\n")
