from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "olist_full_cleaned.csv"
CHECKPOINT_FILE = BASE_DIR / "artifacts" / "dqn_pricing_checkpoint.pt"
SCALER_FILE = BASE_DIR / "artifacts" / "dqn_scaler.pkl"

ACTION_MULTIPLIERS = np.array([-0.30, -0.15, 0.00, 0.15, 0.30], dtype=np.float64)

app = FastAPI(
    title="Pricing RL API",
    description="API de service pour recommendations de prix basées sur le checkpoint DQN du projet.",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    category: str = Field(..., description="Nom de la catégorie produit")
    current_price: float = Field(..., gt=0.0, description="Prix actuel du produit")
    month: int = Field(..., ge=1, le=12, description="Mois de l'année (1-12)")
    review_score: float = Field(4.0, ge=1.0, le=5.0, description="Score moyen des avis")
    marketing_boost: float = Field(0.0, description="Effet marketing en pourcentage")
    demand_shock: float = Field(0.0, description="Choc de demande en pourcentage")
    stock_pressure: float = Field(1.0, description="Couverture de stock (1=normal)")
    cost_shock: float = Field(0.0, description="Choc de coût unitaire en pourcentage")


class ActionScore(BaseModel):
    action_id: int
    action: str
    delta_pct: float
    suggested_price: float
    expected_orders: float
    gross_profit: float
    churn_risk: float
    satisfaction: float
    reward_proxy: float


class PredictResponse(BaseModel):
    category: str
    month: int
    current_price: float
    recommended_action: str
    suggested_price: float
    reward_proxy: float
    q_values: List[float]
    scores: List[ActionScore]


@dataclass
class CategoryProfile:
    category: str
    base_price: float
    base_freight: float
    base_demand: float
    elasticity: float
    base_review: float
    unit_cost: float


@dataclass
class DQNBundledModel:
    online_net: object
    scaler: object
    state_cols: list[str]
    action_multipliers: np.ndarray


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


def _safe_clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def load_dataset() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable: {DATA_FILE}. Exécutez le pipeline de données d'abord."
        )

    df = pd.read_csv(DATA_FILE)
    required = {
        "order_id",
        "product_category_name",
        "price",
        "order_purchase_timestamp",
    }
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le dataset: {missing}")

    df = df.copy()
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")
    df = df.dropna(subset=["order_purchase_timestamp", "product_category_name", "price"])
    df["product_category_name"] = df["product_category_name"].astype(str)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])

    if "review_score_mean" not in df.columns:
        df["review_score_mean"] = 4.0
    df["review_score_mean"] = pd.to_numeric(df["review_score_mean"], errors="coerce").fillna(4.0)

    return df


def build_monthly_profiles(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["year_month"] = work["order_purchase_timestamp"].dt.to_period("M").astype(str)
    work["month"] = work["order_purchase_timestamp"].dt.month

    monthly = (
        work.groupby(["product_category_name", "year_month", "month"], as_index=False)
        .agg(
            demand_qty=("order_id", "nunique"),
            avg_price=("price", "mean"),
            avg_review=("review_score_mean", "mean"),
        )
        .sort_values(["product_category_name", "year_month"])
    )

    monthly["pct_demand"] = monthly.groupby("product_category_name")["demand_qty"].pct_change()
    monthly["pct_price"] = monthly.groupby("product_category_name")["avg_price"].pct_change()
    monthly["elasticity_raw"] = monthly["pct_demand"] / monthly["pct_price"].replace(0, np.nan)

    profile = (
        monthly.groupby("product_category_name", as_index=False)
        .agg(
            base_price=("avg_price", "median"),
            base_demand=("demand_qty", "median"),
            base_review=("avg_review", "mean"),
            elasticity=("elasticity_raw", "median"),
        )
        .sort_values("product_category_name")
    )

    if "freight_value" in work.columns:
        freight_proxy = work.groupby("product_category_name")["freight_value"].median().rename("base_freight")
        profile = profile.merge(freight_proxy, on="product_category_name", how="left")
        profile["base_freight"] = pd.to_numeric(profile["base_freight"], errors="coerce").fillna(0.0)
    else:
        profile["base_freight"] = 0.0

    profile["elasticity"] = (
        pd.to_numeric(profile["elasticity"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(-1.2)
    )
    profile["elasticity"] = -np.abs(profile["elasticity"]).clip(lower=0.20, upper=3.50)

    cost_proxy = work.groupby("product_category_name")["price"].quantile(0.35).rename("unit_cost")
    profile = profile.merge(cost_proxy, on="product_category_name", how="left")
    profile["unit_cost"] = np.minimum(profile["unit_cost"], 0.92 * profile["base_price"]).clip(lower=1.0)

    seasonality = (
        monthly.groupby(["product_category_name", "month"], as_index=False)
        .agg(monthly_demand=("demand_qty", "mean"))
    )
    seasonality["cat_avg"] = seasonality.groupby("product_category_name")["monthly_demand"].transform("mean")
    seasonality["seasonality_index"] = (seasonality["monthly_demand"] / seasonality["cat_avg"]).clip(0.5, 1.8)

    price_grid = (
        work.groupby("product_category_name", as_index=False)
        .agg(
            p10=("price", lambda s: float(np.percentile(s, 10))),
            p90=("price", lambda s: float(np.percentile(s, 90))),
        )
    )

    return profile, seasonality, price_grid


def load_dqn_model() -> DQNBundledModel:
    if not CHECKPOINT_FILE.exists() or not SCALER_FILE.exists():
        raise FileNotFoundError(f"Checkpoint ou scaler manquant: {CHECKPOINT_FILE}, {SCALER_FILE}")

    checkpoint = torch.load(CHECKPOINT_FILE, map_location="cpu")
    with open(SCALER_FILE, "rb") as f:
        scaler = pd.read_pickle(f)

    state_dim = int(checkpoint["state_dim"])
    action_dim = int(checkpoint["action_dim"])
    state_cols = list(checkpoint["state_cols"])
    action_multipliers = np.asarray(checkpoint["action_multipliers"], dtype=np.float64)

    net = QNetwork(state_dim=state_dim, action_dim=action_dim)
    net.load_state_dict(checkpoint["online_net_state_dict"])
    net.eval()

    return DQNBundledModel(
        online_net=net,
        scaler=scaler,
        state_cols=state_cols,
        action_multipliers=action_multipliers,
    )


def get_display_categories(df: pd.DataFrame, profile_df: pd.DataFrame) -> list[str]:
    invalid_names = {"", "unknown", "nan", "none", "null"}
    raw_categories = df["product_category_name"].astype(str).str.strip().tolist()
    raw_set = {c for c in raw_categories if c and c.lower() not in invalid_names}
    profile_categories = profile_df["product_category_name"].astype(str).str.strip().tolist()
    return [c for c in profile_categories if c in raw_set]


def get_profile(profile_df: pd.DataFrame, category: str) -> CategoryProfile:
    row = profile_df.loc[profile_df["product_category_name"] == category]
    if row.empty:
        raise ValueError(f"Categorie inconnue: {category}")
    r = row.iloc[0]
    return CategoryProfile(
        category=category,
        base_price=float(r["base_price"]),
        base_freight=float(r["base_freight"]),
        base_demand=float(r["base_demand"]),
        elasticity=float(r["elasticity"]),
        base_review=float(r["base_review"]),
        unit_cost=float(r["unit_cost"]),
    )


def get_seasonality(seasonality_df: pd.DataFrame, category: str, month: int) -> float:
    row = seasonality_df[
        (seasonality_df["product_category_name"] == category) &
        (seasonality_df["month"] == month)
    ]
    if row.empty:
        return 1.0
    return float(row.iloc[0]["seasonality_index"])


def infer_dqn_action(
    dqn_model: DQNBundledModel,
    profile: CategoryProfile,
    current_price: float,
    month: int,
) -> tuple[int, np.ndarray]:
    mois_sin = float(np.sin(2.0 * np.pi * month / 12.0))
    mois_cos = float(np.cos(2.0 * np.pi * month / 12.0))
    base_state = {
        "price": float(current_price),
        "freight_value": float(profile.base_freight),
        "base_demand": float(profile.base_demand),
        "elasticite_demande": float(profile.elasticity),
        "historical_review_score": float(profile.base_review),
        "mois_sin": mois_sin,
        "mois_cos": mois_cos,
        "is_weekend": 0.0,
    }
    x = np.array([base_state.get(col, 0.0) for col in dqn_model.state_cols], dtype=np.float64).reshape(1, -1)
    x_scaled = dqn_model.scaler.transform(x)
    with torch.no_grad():
        tensor_state = torch.tensor(x_scaled, dtype=torch.float32)
        q_values = dqn_model.online_net(tensor_state).squeeze(0).cpu().numpy()
    action_id = int(np.argmax(q_values))
    return action_id, q_values


def simulate_customer_response(
    profile: CategoryProfile,
    target_price: float,
    month: int,
    seasonality_idx: float,
    review_score: float,
    marketing_boost_pct: float,
    demand_shock_pct: float,
    stock_pressure: float,
    unit_cost_shock_pct: float,
) -> dict[str, float]:
    target_price = max(1.0, float(target_price))
    review_score = _safe_clip(float(review_score), 1.0, 5.0)
    marketing_boost = float(marketing_boost_pct) / 100.0
    demand_shock = float(demand_shock_pct) / 100.0
    rel_change = (target_price - profile.base_price) / (profile.base_price + 1e-6)
    price_factor = float(np.exp(profile.elasticity * rel_change))
    price_factor = _safe_clip(price_factor, 0.05, 3.0)
    review_gap = review_score - profile.base_review
    review_factor = _safe_clip(1.0 + 0.06 * review_gap, 0.70, 1.30)
    seasonality_factor = _safe_clip(seasonality_idx, 0.5, 1.8)
    marketing_factor = _safe_clip(1.0 + marketing_boost, 0.75, 1.8)
    demand_shock_factor = _safe_clip(1.0 + demand_shock, 0.5, 2.2)
    stock_factor = _safe_clip(stock_pressure, 0.35, 1.2)
    expected_orders = profile.base_demand * price_factor * review_factor * seasonality_factor * marketing_factor
    expected_orders *= demand_shock_factor
    expected_orders *= stock_factor
    expected_orders = max(0.0, float(expected_orders))
    unit_cost = profile.unit_cost * (1.0 + float(unit_cost_shock_pct) / 100.0)
    unit_cost = _safe_clip(unit_cost, 1.0, 0.98 * target_price)
    revenue = expected_orders * target_price
    margin_unit = target_price - unit_cost
    gross_profit = expected_orders * margin_unit
    overpricing_ratio = max(0.0, target_price / (profile.base_price + 1e-6) - 1.0)
    churn_risk = overpricing_ratio * max(0.0, abs(profile.elasticity)) * (1.0 + max(0.0, 3.8 - review_score))
    churn_risk += max(0.0, 0.9 - stock_factor) * 0.30
    churn_risk = _safe_clip(churn_risk, 0.0, 1.0)
    satisfaction_score = 75.0 + 10.0 * (review_score - 3.5)
    satisfaction_score += 8.0 * min(0.0, rel_change)
    satisfaction_score -= 28.0 * max(0.0, overpricing_ratio)
    satisfaction_score -= 10.0 * max(0.0, 0.85 - stock_factor)
    satisfaction_score = _safe_clip(satisfaction_score, 0.0, 100.0)
    return {
        "price": float(target_price),
        "expected_orders": float(expected_orders),
        "revenue": float(revenue),
        "gross_profit": float(gross_profit),
        "margin_unit": float(margin_unit),
        "unit_cost": float(unit_cost),
        "churn_risk": float(churn_risk),
        "satisfaction_score": float(satisfaction_score),
    }


def score_actions(
    dqn_model: DQNBundledModel,
    profile: CategoryProfile,
    current_price: float,
    month: int,
    seasonality_idx: float,
    review_score: float,
    marketing_boost_pct: float,
    demand_shock_pct: float,
    stock_pressure: float,
    unit_cost_shock_pct: float,
) -> pd.DataFrame:
    _, q_values = infer_dqn_action(
        dqn_model=dqn_model,
        profile=profile,
        current_price=current_price,
        month=month,
    )

    rows = []
    for action_id, delta in enumerate(ACTION_MULTIPLIERS):
        new_price = max(1.0, float(current_price) * (1.0 + float(delta)))
        sim = simulate_customer_response(
            profile=profile,
            target_price=new_price,
            month=month,
            seasonality_idx=seasonality_idx,
            review_score=review_score,
            marketing_boost_pct=marketing_boost_pct,
            demand_shock_pct=demand_shock_pct,
            stock_pressure=stock_pressure,
            unit_cost_shock_pct=unit_cost_shock_pct,
        )

        rows.append(
            {
                "action_id": action_id,
                "action": f"{'DECREASE_LARGE' if action_id == 0 else 'DECREASE_SMALL' if action_id == 1 else 'MAINTAIN' if action_id == 2 else 'INCREASE_SMALL' if action_id == 3 else 'INCREASE_LARGE'}",
                "delta_pct": float(delta * 100.0),
                "suggested_price": float(new_price),
                "expected_orders": float(sim["expected_orders"]),
                "gross_profit": float(sim["gross_profit"]),
                "churn_risk": float(sim["churn_risk"]),
                "satisfaction": float(sim["satisfaction_score"]),
                "reward_proxy": float(q_values[action_id]),
            }
        )

    return pd.DataFrame(rows).sort_values("reward_proxy", ascending=False).reset_index(drop=True)


@app.on_event("startup")
def startup_event() -> None:
    global df, profile_df, seasonality_df, model
    df = load_dataset()
    profile_df, seasonality_df, _ = build_monthly_profiles(df)
    model = load_dqn_model()


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/categories")
def list_categories() -> dict[str, List[str]]:
    categories = get_display_categories(df, profile_df)
    return {"categories": categories}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        profile = get_profile(profile_df, request.category)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if request.month < 1 or request.month > 12:
        raise HTTPException(status_code=400, detail="Le mois doit être entre 1 et 12.")

    seasonality_idx = get_seasonality(seasonality_df, request.category, request.month)
    scores_df = score_actions(
        dqn_model=model,
        profile=profile,
        current_price=request.current_price,
        month=request.month,
        seasonality_idx=seasonality_idx,
        review_score=request.review_score,
        marketing_boost_pct=request.marketing_boost,
        demand_shock_pct=request.demand_shock,
        stock_pressure=request.stock_pressure,
        unit_cost_shock_pct=request.cost_shock,
    )

    best = scores_df.iloc[0]
    return PredictResponse(
        category=request.category,
        month=request.month,
        current_price=request.current_price,
        recommended_action=str(best["action"]),
        suggested_price=float(best["suggested_price"]),
        reward_proxy=float(best["reward_proxy"]),
        q_values=[float(v) for v in infer_dqn_action(model, profile, request.current_price, request.month)[1].tolist()],
        scores=[ActionScore(**row) for row in scores_df.to_dict(orient="records")],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_service:app", host="0.0.0.0", port=8000, reload=False)
