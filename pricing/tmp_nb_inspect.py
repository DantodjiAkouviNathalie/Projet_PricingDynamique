import json
from pathlib import Path

path = Path('chargement_datasets.ipynb')
nb = json.loads(path.read_text(encoding='utf-8'))
print('cells', len(nb['cells']))
for i, cell in enumerate(nb['cells'], 1):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    if any(term in src for term in ['to_csv(', 'olist_full_cleaned.csv', 'pricing_features.to_csv', 'pd.read_csv(', 'fillna(', 'groupby(', 'dropna', 'order_purchase_timestamp']):
        print('CELL', i, '---')
        print(src)
        print('---')
