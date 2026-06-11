from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
import numpy as np

DATASET_FILES = {
    'customers': 'olist_customers_dataset.csv',
    'geolocation': 'olist_geolocation_dataset.csv',
    'order_items': 'olist_order_items_dataset.csv',
    'order_payments': 'olist_order_payments_dataset.csv',
    'order_reviews': 'olist_order_reviews_dataset.csv',
    'orders': 'olist_orders_dataset.csv',
    'products': 'olist_products_dataset.csv',
    'sellers': 'olist_sellers_dataset.csv',
    'category_translation': 'product_category_name_translation.csv',
}

DATE_COLUMNS = {
    'orders': [
        'order_purchase_timestamp',
        'order_approved_at',
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date',
    ],
    'order_reviews': ['review_creation_date', 'review_answer_timestamp'],
}


def load_raw_datasets(base_path: Path) -> dict[str, pd.DataFrame]:
    datasets: dict[str, pd.DataFrame] = {}

    for name, filename in DATASET_FILES.items():
        file_path = base_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier introuvable: {file_path}")

        parse_dates = DATE_COLUMNS.get(name)
        datasets[name] = pd.read_csv(file_path, parse_dates=parse_dates)

    return datasets


def consolidate_orders(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    orders = datasets['orders'].copy()
    customers = datasets['customers'].copy()
    order_items = datasets['order_items'].copy()
    products = datasets['products'].copy()
    sellers = datasets['sellers'].copy()
    order_payments = datasets['order_payments'].copy()
    order_reviews = datasets['order_reviews'].copy()

    payments_agg = (
        order_payments
        .groupby('order_id', as_index=False)
        .agg(
            payment_sequential_max=('payment_sequential', 'max'),
            payment_installments_max=('payment_installments', 'max'),
            payment_value_total=('payment_value', 'sum'),
            payment_type_nunique=('payment_type', 'nunique'),
        )
    )

    reviews_agg = (
        order_reviews
        .sort_values('review_answer_timestamp')
        .groupby('order_id', as_index=False)
        .agg(
            review_score_mean=('review_score', 'mean'),
            review_creation_date_max=('review_creation_date', 'max'),
            review_answer_timestamp_max=('review_answer_timestamp', 'max'),
        )
    )

    olist_full = (
        orders
        .merge(customers, on='customer_id', how='left')
        .merge(order_items, on='order_id', how='left')
        .merge(products, on='product_id', how='left')
        .merge(sellers, on='seller_id', how='left')
        .merge(payments_agg, on='order_id', how='left')
        .merge(reviews_agg, on='order_id', how='left')
    )

    return olist_full


def clean_olist_full(olist_full: pd.DataFrame) -> pd.DataFrame:
    olist_full = olist_full.copy()

    duplicate_count = int(olist_full.duplicated().sum())
    if duplicate_count > 0:
        olist_full = olist_full.drop_duplicates().copy()
        print(f"Suppression de {duplicate_count} doublons.")
    else:
        print("Aucun doublon trouvé.")

    cat_cols = olist_full.select_dtypes(include=['object', 'category', 'string']).columns
    for col in cat_cols:
        olist_full.loc[:, col] = olist_full[col].fillna('unknown')

    num_cols = olist_full.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        olist_full.loc[:, col] = olist_full[col].fillna(olist_full[col].median())

    date_cols = [col for col in olist_full.columns if ('date' in col or 'timestamp' in col)]
    for col in date_cols:
        olist_full[col] = pd.to_datetime(olist_full[col], errors='coerce')

    olist_full['order_approved_at'] = olist_full['order_approved_at'].fillna(
        olist_full['order_purchase_timestamp']
    )
    olist_full['order_delivered_carrier_date'] = olist_full['order_delivered_carrier_date'].fillna(
        olist_full['order_approved_at']
    )
    olist_full['order_delivered_customer_date'] = olist_full['order_delivered_customer_date'].fillna(
        olist_full['order_estimated_delivery_date']
    )
    olist_full['review_creation_date_max'] = olist_full['review_creation_date_max'].fillna(
        olist_full['order_delivered_customer_date']
    )
    olist_full['review_answer_timestamp_max'] = olist_full['review_answer_timestamp_max'].fillna(
        olist_full['review_creation_date_max']
    )
    olist_full['shipping_limit_date'] = olist_full['shipping_limit_date'].fillna(
        olist_full['order_purchase_timestamp']
    )

    return olist_full


def extract_pricing_features(olist_full: pd.DataFrame) -> pd.DataFrame:
    olist_features = olist_full.copy()

    olist_features['annee_mois'] = olist_features['order_purchase_timestamp'].dt.to_period('M').astype(str)
    olist_features['mois'] = olist_features['order_purchase_timestamp'].dt.month
    olist_features['trimestre'] = olist_features['order_purchase_timestamp'].dt.quarter
    olist_features['jour_semaine'] = olist_features['order_purchase_timestamp'].dt.dayofweek
    olist_features['is_weekend'] = (olist_features['jour_semaine'] >= 5).astype(int)
    olist_features['mois_sin'] = np.sin(2 * np.pi * olist_features['mois'] / 12)
    olist_features['mois_cos'] = np.cos(2 * np.pi * olist_features['mois'] / 12)

    order_basket = (
        olist_features
        .groupby(['order_id', 'customer_unique_id'], as_index=False)
        .agg(
            nb_items_commande=('order_item_id', 'count'),
            panier_commande=('price', 'sum'),
            fret_commande=('freight_value', 'sum'),
        )
    )
    order_basket['panier_commande_avec_fret'] = (
        order_basket['panier_commande'] + order_basket['fret_commande']
    )

    customer_history = (
        order_basket
        .groupby('customer_unique_id', as_index=False)
        .agg(
            nb_commandes_client=('order_id', 'nunique'),
            panier_moyen_client=('panier_commande_avec_fret', 'mean'),
            depense_totale_client=('panier_commande_avec_fret', 'sum'),
        )
    )

    olist_features['delai_approbation_h'] = (
        olist_features['order_approved_at'] - olist_features['order_purchase_timestamp']
    ).dt.total_seconds() / 3600

    olist_features['delai_livraison_j'] = (
        olist_features['order_delivered_customer_date'] - olist_features['order_purchase_timestamp']
    ).dt.days

    olist_features['retard_livraison_j'] = (
        olist_features['order_delivered_customer_date'] - olist_features['order_estimated_delivery_date']
    ).dt.days

    elasticity_base = (
        olist_features
        .groupby(['product_category_name', 'annee_mois'], as_index=False)
        .agg(
            demande_qte=('order_id', 'count'),
            prix_moyen_categorie_mois=('price', 'mean'),
        )
        .sort_values(['product_category_name', 'annee_mois'])
    )

    elasticity_base['pct_change_demande'] = (
        elasticity_base
        .groupby('product_category_name')['demande_qte']
        .pct_change()
    )
    elasticity_base['pct_change_prix'] = (
        elasticity_base
        .groupby('product_category_name')['prix_moyen_categorie_mois']
        .pct_change()
    )
    elasticity_base['elasticite_demande'] = (
        elasticity_base['pct_change_demande'] /
        elasticity_base['pct_change_prix'].replace(0, np.nan)
    )
    elasticity_base['elasticite_demande'] = (
        elasticity_base['elasticite_demande']
        .replace([np.inf, -np.inf], np.nan)
    )

    selected_cols = [
        'order_id',
        'customer_unique_id',
        'product_id',
        'product_category_name',
        'price',
        'freight_value',
        'payment_value_total',
        'review_score_mean',
        'order_purchase_timestamp',
        'order_approved_at',
        'order_delivered_customer_date',
        'order_estimated_delivery_date',
        'annee_mois',
        'mois',
        'trimestre',
        'jour_semaine',
        'is_weekend',
        'mois_sin',
        'mois_cos',
        'delai_approbation_h',
        'delai_livraison_j',
        'retard_livraison_j',
    ]

    pricing_features = (
        olist_features[selected_cols]
        .merge(
            order_basket[
                ['order_id', 'nb_items_commande', 'panier_commande', 'fret_commande', 'panier_commande_avec_fret']
            ],
            on='order_id',
            how='left',
        )
        .merge(customer_history, on='customer_unique_id', how='left')
        .merge(
            elasticity_base[['product_category_name', 'annee_mois', 'elasticite_demande']],
            on=['product_category_name', 'annee_mois'],
            how='left',
        )
    )

    for col in ['delai_approbation_h', 'delai_livraison_j', 'retard_livraison_j', 'elasticite_demande']:
        pricing_features[col] = pricing_features[col].fillna(pricing_features[col].median())

    return pricing_features


def save_outputs(
    cleaned_df: pd.DataFrame,
    features_df: pd.DataFrame,
    output_dir: Path,
    cleaned_filename: str,
    features_filename: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = output_dir / cleaned_filename
    features_path = output_dir / features_filename

    cleaned_df.to_csv(cleaned_path, index=False)
    features_df.to_csv(features_path, index=False)

    print(f"Saved cleaned dataset to {cleaned_path}")
    print(f"Saved pricing features to {features_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Pipeline de préparation des données Olist')
    parser.add_argument(
        '--raw-dir',
        type=Path,
        default=Path('.'),
        help='Dossier contenant les fichiers CSV bruts',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('.'),
        help='Dossier de sortie pour les fichiers nettoyés et les features',
    )
    parser.add_argument(
        '--cleaned-filename',
        type=str,
        default='olist_full_cleaned.csv',
        help='Nom du fichier CSV nettoyé produit',
    )
    parser.add_argument(
        '--features-filename',
        type=str,
        default='olist_pricing_features.csv',
        help='Nom du fichier CSV des features produit',
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print('Chargement des données brutes...')
    datasets = load_raw_datasets(args.raw_dir)
    print('Consolidation des données...')
    olist_full = consolidate_orders(datasets)
    print('Nettoyage du dataset consolidé...')
    olist_full_cleaned = clean_olist_full(olist_full)
    print('Extraction des features de pricing dynamique...')
    pricing_features = extract_pricing_features(olist_full_cleaned)
    save_outputs(
        cleaned_df=olist_full_cleaned,
        features_df=pricing_features,
        output_dir=args.output_dir,
        cleaned_filename=args.cleaned_filename,
        features_filename=args.features_filename,
    )


if __name__ == '__main__':
    main()
