import requests
import json

# First get list of available categories
response = requests.get('http://localhost:8000/categories')
categories = response.json()['categories']
print(f"Total categories: {len(categories)}")
print(f"First 10: {categories[:10]}")

# Try with first category
test_category = categories[0]
print(f"\nTesting with category: {test_category}")

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

response = requests.post('http://localhost:8000/predict', json=request_data)
result = response.json()

print(f"\nResponse Status: {response.status_code}")
if response.status_code == 200:
    print(f"Suggested Price: {result.get('suggested_price')}")
    print(f"Recommended Action: {result.get('recommended_action')}")
    print(f"Reward Proxy: {result.get('reward_proxy'):.4f}")
    print(f"\nAction Scores:")
    for i, score in enumerate(result['scores']):
        print(f"  Action {i}: Delta={score['delta_pct']:+.1f}%, Orders={score['expected_orders']:.1f}, "
              f"Profit=${score['gross_profit']:.2f}")
else:
    print(f"Error: {result}")
