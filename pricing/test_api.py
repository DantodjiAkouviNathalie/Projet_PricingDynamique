import requests
import json

# Test data for prediction
request_data = {
    "category": "informatica",
    "current_price": 500.0,
    "month": 6,
    "review_score": 4.5,
    "marketing_boost": 0.0,
    "demand_shock": 0.0,
    "stock_pressure": 0.0,
    "cost_shock": 0.0
}

response = requests.post('http://localhost:8000/predict', json=request_data)
result = response.json()

print("Response Status:", response.status_code)
print("\nPrediction Result (truncated):")
result_str = json.dumps(result, indent=2, default=str)
print(result_str[:1500])

print("\n\n=== Full Response Structure ===")
print(f"Suggested Price: {result.get('suggested_price')}")
print(f"Recommended Action: {result.get('recommended_action')}")
print(f"Reward Proxy: {result.get('reward_proxy')}")
print(f"Number of action scores: {len(result.get('scores', []))}")

if result.get('scores'):
    print("\nAction Scores:")
    for score in result['scores']:
        print(f"  - Delta: {score['delta_pct']}%, Expected Orders: {score['expected_orders']:.1f}, "
              f"Gross Profit: ${score['gross_profit']:.2f}")
