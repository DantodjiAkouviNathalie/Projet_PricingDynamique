from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - fallback runtime if torch is unavailable
    torch = None
    nn = None


DATA_FILE = Path("olist_full_cleaned.csv")
CHECKPOINT_FILE = Path("artifacts/dqn_pricing_checkpoint.pt")
SCALER_FILE = Path("artifacts/dqn_scaler.pkl")
ACTION_MULTIPLIERS = np.array([-0.30, -0.15, 0.00, 0.15, 0.30], dtype=np.float64)
ACTION_NAMES = {
    0: "DECREASE_LARGE (-30%)",
    1: "DECREASE_SMALL (-15%)",
    2: "MAINTAIN (0%)",
    3: "INCREASE_SMALL (+15%)",
    4: "INCREASE_LARGE (+30%)",
}

SCENARIO_PRESETS = {
    "Personnalise": {"demand_shock": 0.0, "stock_pressure": 1.0, "cost_shock": 0.0, "marketing": 0},
    "Forte demande": {"demand_shock": 35.0, "stock_pressure": 1.0, "cost_shock": 0.0, "marketing": 8},
    "Faible stock": {"demand_shock": 0.0, "stock_pressure": 0.55, "cost_shock": 18.0, "marketing": 0},
    "Forte demande + stock faible": {"demand_shock": 45.0, "stock_pressure": 0.55, "cost_shock": 20.0, "marketing": 10},
    "Marche degrade": {"demand_shock": -20.0, "stock_pressure": 1.0, "cost_shock": 8.0, "marketing": -5},
}

SIMULATOR_ALLOWED_CATEGORIES = [
    "cama_mesa_banho",
    "beleza_saude",
    "esporte_lazer",
    "moveis_decoracao",
    "informatica_acessorios",
    "utilidades_domesticas",
    "relogios_presentes",
    "telefonia",
    "ferramentas_jardim",
    "automotivo",
]


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
class AgentDecision:
    action_id: int
    action_name: str
    suggested_price: float
    reward_proxy: float
    rationale: list[str]


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

    def forward(self, x):
        features = self.feature_extractor(x)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


def _safe_clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


@st.cache_resource(show_spinner=False)
def load_dqn_model() -> tuple[DQNBundledModel | None, str]:
    if torch is None:
        return None, "Torch non disponible."
    if not CHECKPOINT_FILE.exists() or not SCALER_FILE.exists():
        return None, "Checkpoint/scaler absents."

    try:
        checkpoint = torch.load(CHECKPOINT_FILE, map_location="cpu")
        with open(SCALER_FILE, "rb") as f:
            scaler = pickle.load(f)

        state_dim = int(checkpoint["state_dim"])
        action_dim = int(checkpoint["action_dim"])
        state_cols = list(checkpoint["state_cols"])
        action_multipliers = np.asarray(checkpoint["action_multipliers"], dtype=np.float64)

        net = QNetwork(state_dim=state_dim, action_dim=action_dim)
        net.load_state_dict(checkpoint["online_net_state_dict"])
        net.eval()

        if len(action_multipliers) != len(ACTION_MULTIPLIERS):
            return None, "Checkpoint incompatible (action space)."

        bundled = DQNBundledModel(
            online_net=net,
            scaler=scaler,
            state_cols=state_cols,
            action_multipliers=action_multipliers,
        )
        return bundled, "Mode DQN actif (checkpoint charge)."
    except Exception as exc:
        return None, f"Echec chargement checkpoint ({exc})."


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


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Fichier introuvable: olist_full_cleaned.csv. "
            "Execute le notebook de preparation avant le simulateur web."
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


@st.cache_data(show_spinner=False)
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


def get_display_categories(df: pd.DataFrame, profile_df: pd.DataFrame) -> list[str]:
    invalid_names = {"", "unknown", "nan", "none", "null"}

    raw_categories = (
        df["product_category_name"]
        .astype(str)
        .str.strip()
        .tolist()
    )
    raw_set = {c for c in raw_categories if c and c.lower() not in invalid_names}

    profile_categories = (
        profile_df["product_category_name"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    categories = [c for c in profile_categories if c in raw_set]
    allowed = [c for c in SIMULATOR_ALLOWED_CATEGORIES if c in set(categories)]
    return allowed


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
        (seasonality_df["product_category_name"] == category)
        & (seasonality_df["month"] == month)
    ]
    if row.empty:
        return 1.0
    return float(row.iloc[0]["seasonality_index"])


def simulate_customer_response(
    profile: CategoryProfile,
    target_price: float,
    month: int,
    seasonality_idx: float,
    review_score: float,
    marketing_boost_pct: float,
    demand_shock_pct: float = 0.0,
    stock_pressure: float = 1.0,
    unit_cost_shock_pct: float = 0.0,
    rng: np.random.Generator | None = None,
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

    noise = 1.0
    if rng is not None:
        noise = float(rng.normal(loc=1.0, scale=0.035))
        noise = _safe_clip(noise, 0.85, 1.15)
    expected_orders = max(0.0, float(expected_orders * noise))

    unit_cost = profile.unit_cost * (1.0 + float(unit_cost_shock_pct) / 100.0)
    unit_cost = _safe_clip(unit_cost, 1.0, 0.98 * target_price)

    revenue = expected_orders * target_price
    margin_unit = target_price - unit_cost
    gross_profit = expected_orders * margin_unit

    overpricing_ratio = max(0.0, target_price / (profile.base_price + 1e-6) - 1.0)
    churn_risk = overpricing_ratio * max(0.4, abs(profile.elasticity)) * (1.0 + max(0.0, 3.8 - review_score))
    churn_risk += max(0.0, 0.9 - stock_factor) * 0.30
    churn_risk = _safe_clip(churn_risk, 0.0, 1.0)

    satisfaction_score = 75.0 + 10.0 * (review_score - 3.5)
    satisfaction_score += 8.0 * min(0.0, rel_change)
    satisfaction_score -= 28.0 * max(0.0, overpricing_ratio)
    satisfaction_score -= 10.0 * max(0.0, 0.85 - stock_factor)
    satisfaction_score = _safe_clip(satisfaction_score, 0.0, 100.0)

    return {
        "month": float(month),
        "price": float(target_price),
        "expected_orders": float(expected_orders),
        "revenue": float(revenue),
        "gross_profit": float(gross_profit),
        "margin_unit": float(margin_unit),
        "unit_cost": float(unit_cost),
        "price_factor": float(price_factor),
        "review_factor": float(review_factor),
        "seasonality_factor": float(seasonality_factor),
        "marketing_factor": float(marketing_factor),
        "demand_shock_factor": float(demand_shock_factor),
        "stock_factor": float(stock_factor),
        "churn_risk": float(churn_risk),
        "satisfaction_score": float(satisfaction_score),
        "elasticity": float(profile.elasticity),
        "overpricing_ratio": float(overpricing_ratio),
    }


def build_comparison_table(base: dict[str, float], test: dict[str, float]) -> pd.DataFrame:
    metrics = [
        ("Commandes attendues", "expected_orders"),
        ("Revenu", "revenue"),
        ("Profit brut", "gross_profit"),
        ("Marge unitaire", "margin_unit"),
        ("Risque churn", "churn_risk"),
        ("Score satisfaction", "satisfaction_score"),
    ]
    rows = []
    for label, key in metrics:
        b = float(base[key])
        t = float(test[key])
        delta = t - b
        delta_pct = (delta / (abs(b) + 1e-9)) * 100.0
        rows.append(
            {
                "Metrique": label,
                "Baseline": b,
                "Scenario": t,
                "Delta": delta,
                "Delta %": delta_pct,
            }
        )
    return pd.DataFrame(rows)


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
                "action": ACTION_NAMES[action_id],
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


def explain_decision(scores: pd.DataFrame, profile: CategoryProfile, stock_pressure: float, demand_shock_pct: float) -> AgentDecision:
    best = scores.iloc[0]
    second = scores.iloc[1] if len(scores) > 1 else best
    margin_adv = float(best["reward_proxy"] - second["reward_proxy"])

    rationale = [
        f"Avantage de score vs 2eme action: {margin_adv:+.3f}",
        f"Elasticite estimee: {profile.elasticity:.2f} (demande sensible au prix)",
        f"Risque churn estime: {100.0 * float(best['churn_risk']):.1f}%",
    ]
    if stock_pressure < 0.8:
        rationale.append("Contrainte de stock detectee: le policy favorise la protection de marge.")
    if demand_shock_pct > 20:
        rationale.append("Pic de demande detecte: le policy autorise des hausses moderees.")

    return AgentDecision(
        action_id=int(best["action_id"]),
        action_name=str(best["action"]),
        suggested_price=float(best["suggested_price"]),
        reward_proxy=float(best["reward_proxy"]),
        rationale=rationale,
    )


def build_price_curve(
    profile: CategoryProfile,
    month: int,
    seasonality_idx: float,
    review_score: float,
    marketing_boost_pct: float,
    demand_shock_pct: float,
    stock_pressure: float,
    unit_cost_shock_pct: float,
    p10: float,
    p90: float,
) -> pd.DataFrame:
    lo = max(1.0, min(p10, 0.7 * profile.base_price))
    hi = max(lo + 1.0, max(p90, 1.4 * profile.base_price))
    prices = np.linspace(lo, hi, 50)

    rows = []
    for p in prices:
        sim = simulate_customer_response(
            profile=profile,
            target_price=float(p),
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
                "price": sim["price"],
                "expected_orders": sim["expected_orders"],
                "revenue": sim["revenue"],
                "gross_profit": sim["gross_profit"],
            }
        )

    curve = pd.DataFrame(rows)
    curve["price_bin"] = pd.qcut(curve["price"], q=10, duplicates="drop")
    return curve


def simulate_horizon(
    dqn_model: DQNBundledModel,
    profile: CategoryProfile,
    start_price: float,
    start_month: int,
    horizon_months: int,
    seasonality_df: pd.DataFrame,
    review_score: float,
    marketing_boost_pct: float,
    demand_shock_pct: float,
    stock_pressure: float,
    unit_cost_shock_pct: float,
) -> pd.DataFrame:
    current_price = float(start_price)
    rows = []

    for step in range(horizon_months):
        month = ((start_month - 1 + step) % 12) + 1
        seasonality_idx = get_seasonality(seasonality_df, profile.category, month)

        scores = score_actions(
            dqn_model=dqn_model,
            profile=profile,
            current_price=current_price,
            month=month,
            seasonality_idx=seasonality_idx,
            review_score=review_score,
            marketing_boost_pct=marketing_boost_pct,
            demand_shock_pct=demand_shock_pct,
            stock_pressure=stock_pressure,
            unit_cost_shock_pct=unit_cost_shock_pct,
        )
        best = scores.iloc[0]

        sim = simulate_customer_response(
            profile=profile,
            target_price=float(best["suggested_price"]),
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
                "mois_simule": step + 1,
                "mois_calendaire": month,
                "action": best["action"],
                "action_pct": float(best["delta_pct"]),
                "price": float(best["suggested_price"]),
                "expected_orders": float(sim["expected_orders"]),
                "gross_profit": float(sim["gross_profit"]),
                "revenue": float(sim["revenue"]),
                "churn_risk": float(sim["churn_risk"]),
                "reward_proxy": float(best["reward_proxy"]),
            }
        )

        current_price = 0.65 * float(best["suggested_price"]) + 0.35 * float(profile.base_price)

    return pd.DataFrame(rows)


def run_monte_carlo(
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
    n_runs: int,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for _ in range(n_runs):
        shock_demand = demand_shock_pct + float(rng.normal(0.0, 7.0))
        shock_review = review_score + float(rng.normal(0.0, 0.20))
        shock_marketing = marketing_boost_pct + float(rng.normal(0.0, 4.0))
        shock_stock = stock_pressure + float(rng.normal(0.0, 0.08))
        shock_cost = unit_cost_shock_pct + float(rng.normal(0.0, 3.5))

        scores = score_actions(
            dqn_model=dqn_model,
            profile=profile,
            current_price=current_price,
            month=month,
            seasonality_idx=seasonality_idx,
            review_score=shock_review,
            marketing_boost_pct=shock_marketing,
            demand_shock_pct=shock_demand,
            stock_pressure=shock_stock,
            unit_cost_shock_pct=shock_cost,
        )
        best = scores.iloc[0]

        sim = simulate_customer_response(
            profile=profile,
            target_price=float(best["suggested_price"]),
            month=month,
            seasonality_idx=seasonality_idx,
            review_score=shock_review,
            marketing_boost_pct=shock_marketing,
            demand_shock_pct=shock_demand,
            stock_pressure=shock_stock,
            unit_cost_shock_pct=shock_cost,
            rng=rng,
        )

        rows.append(
            {
                "action_id": int(best["action_id"]),
                "action": best["action"],
                "suggested_price": float(best["suggested_price"]),
                "gross_profit": float(sim["gross_profit"]),
                "revenue": float(sim["revenue"]),
                "expected_orders": float(sim["expected_orders"]),
                "churn_risk": float(sim["churn_risk"]),
                "reward_proxy": float(best["reward_proxy"]),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Simulateur Pricing RL", layout="wide")
    st.title("Simulateur Web Interactif de Pricing Dynamique")
    st.caption("Integration agent de pricing, simulation business multi-scenarios, robustesse et interpretation")

    try:
        df = load_dataset()
        profile_df, seasonality_df, price_grid_df = build_monthly_profiles(df)
        dqn_model, dqn_status = load_dqn_model()
    except Exception as exc:
        st.error(f"Erreur de chargement/calibration: {exc}")
        st.stop()

    if dqn_model is None:
        st.error(f"Mode DQN obligatoire: {dqn_status}")
        st.stop()
    st.success(dqn_status)

    categories = get_display_categories(df, profile_df)
    if not categories:
        st.error("Aucune categorie exploitable dans les donnees historiques.")
        st.stop()

    with st.sidebar:
        st.header("Configuration")
        category = st.selectbox("Categorie produit", categories)
        preset_name = st.selectbox("Scenario business", list(SCENARIO_PRESETS.keys()), index=0)
        preset = SCENARIO_PRESETS[preset_name]

        profile = get_profile(profile_df, category)
        pgrid = price_grid_df.loc[price_grid_df["product_category_name"] == category]
        if pgrid.empty:
            p10, p90 = max(1.0, 0.7 * profile.base_price), 1.4 * profile.base_price
        else:
            p10 = float(pgrid.iloc[0]["p10"])
            p90 = float(pgrid.iloc[0]["p90"])

        month = st.slider("Mois", min_value=1, max_value=12, value=6, step=1)
        current_price = st.number_input(
            "Prix actuel (BRL)",
            min_value=1.0,
            value=float(round(profile.base_price, 2)),
            step=1.0,
        )
        manual_price = st.number_input(
            "Prix manuel de comparaison (BRL)",
            min_value=1.0,
            value=float(round(profile.base_price * 1.08, 2)),
            step=1.0,
        )

        review_score = st.slider(
            "Review score moyen",
            min_value=1.0,
            max_value=5.0,
            value=float(round(profile.base_review, 1)),
            step=0.1,
        )
        marketing_boost = st.slider("Boost marketing (%)", min_value=-20, max_value=80, value=0, step=1)
        demand_shock = st.slider("Choc de demande (%)", min_value=-50, max_value=120, value=0, step=1)
        stock_pressure = st.slider("Couverture stock (1=normal)", min_value=0.35, max_value=1.20, value=1.0, step=0.01)
        cost_shock = st.slider("Choc de cout unitaire (%)", min_value=-10, max_value=40, value=0, step=1)

        apply_preset = st.toggle("Appliquer automatiquement le preset", value=True)
        if apply_preset:
            demand_shock = int(demand_shock + preset["demand_shock"])
            marketing_boost = int(marketing_boost + preset["marketing"])
            stock_pressure = float(stock_pressure * preset["stock_pressure"])
            cost_shock = int(cost_shock + preset["cost_shock"])

    seasonality_idx = get_seasonality(seasonality_df, category, month)

    base_sim = simulate_customer_response(
        profile=profile,
        target_price=current_price,
        month=month,
        seasonality_idx=seasonality_idx,
        review_score=review_score,
        marketing_boost_pct=marketing_boost,
        demand_shock_pct=demand_shock,
        stock_pressure=stock_pressure,
        unit_cost_shock_pct=cost_shock,
    )
    manual_sim = simulate_customer_response(
        profile=profile,
        target_price=manual_price,
        month=month,
        seasonality_idx=seasonality_idx,
        review_score=review_score,
        marketing_boost_pct=marketing_boost,
        demand_shock_pct=demand_shock,
        stock_pressure=stock_pressure,
        unit_cost_shock_pct=cost_shock,
    )

    scores = score_actions(
        dqn_model=dqn_model,
        profile=profile,
        current_price=current_price,
        month=month,
        seasonality_idx=seasonality_idx,
        review_score=review_score,
        marketing_boost_pct=marketing_boost,
        demand_shock_pct=demand_shock,
        stock_pressure=stock_pressure,
        unit_cost_shock_pct=cost_shock,
    )
    decision = explain_decision(scores, profile, stock_pressure=stock_pressure, demand_shock_pct=demand_shock)
    dqn_action_id, q_values = infer_dqn_action(
        dqn_model=dqn_model,
        profile=profile,
        current_price=current_price,
        month=month,
    )
    dqn_row = scores.loc[scores["action_id"] == dqn_action_id]
    if not dqn_row.empty:
        dqn_best = dqn_row.iloc[0]
        decision = AgentDecision(
            action_id=int(dqn_best["action_id"]),
            action_name=str(dqn_best["action"]),
            suggested_price=float(dqn_best["suggested_price"]),
            reward_proxy=float(dqn_best["reward_proxy"]),
            rationale=decision.rationale + ["Action selectionnee par le reseau DQN entraine."],
        )

    st.subheader("Synthese du contexte")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Prix actuel", f"R${current_price:.2f}")
    k2.metric("Elasticite", f"{profile.elasticity:.2f}")
    k3.metric("Indice saisonnier", f"{seasonality_idx:.2f}")
    k4.metric("Choc demande", f"{demand_shock:+d}%")
    k5.metric("Couverture stock", f"{stock_pressure:.2f}")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Decision agent",
        "Comparaison scenario",
        "Trajectoire multi-mois",
        "Robustesse et interpretation",
    ])

    with tab1:
        st.markdown("### Recommandation de l'agent")
        t1, t2, t3 = st.columns(3)
        t1.metric("Action recommandee", decision.action_name)
        t2.metric("Prix recommande", f"R${decision.suggested_price:.2f}")
        t3.metric("Reward proxy", f"{decision.reward_proxy:.3f}")

        st.markdown("### Justification")
        for reason in decision.rationale:
            st.write(f"- {reason}")

        st.markdown("### Classement des actions")
        scores = scores.copy()
        scores["q_value_dqn"] = scores["action_id"].map(lambda a: float(q_values[int(a)]))
        scores["selected_by_dqn"] = scores["action_id"] == int(decision.action_id)
        st.dataframe(scores, use_container_width=True)

    with tab2:
        st.markdown("### Baseline vs prix manuel")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Commandes attendues",
            f"{manual_sim['expected_orders']:.1f}",
            f"{manual_sim['expected_orders'] - base_sim['expected_orders']:+.1f}",
        )
        c2.metric(
            "Revenu",
            f"R${manual_sim['revenue']:.0f}",
            f"{manual_sim['revenue'] - base_sim['revenue']:+.0f}",
        )
        c3.metric(
            "Profit brut",
            f"R${manual_sim['gross_profit']:.0f}",
            f"{manual_sim['gross_profit'] - base_sim['gross_profit']:+.0f}",
        )
        c4.metric(
            "Risque churn",
            f"{100.0 * manual_sim['churn_risk']:.1f}%",
            f"{100.0 * (manual_sim['churn_risk'] - base_sim['churn_risk']):+.1f} pts",
        )
        st.dataframe(build_comparison_table(base_sim, manual_sim), use_container_width=True)

    with tab3:
        st.markdown("### Projection agent sur horizon")
        horizon = st.slider("Horizon (mois)", min_value=3, max_value=18, value=12, step=1)
        horizon_df = simulate_horizon(
            dqn_model=dqn_model,
            profile=profile,
            start_price=current_price,
            start_month=month,
            horizon_months=horizon,
            seasonality_df=seasonality_df,
            review_score=review_score,
            marketing_boost_pct=marketing_boost,
            demand_shock_pct=demand_shock,
            stock_pressure=stock_pressure,
            unit_cost_shock_pct=cost_shock,
        )

        st.dataframe(horizon_df, use_container_width=True)
        st.line_chart(
            horizon_df.set_index("mois_simule")[["price", "gross_profit", "expected_orders"]],
            use_container_width=True,
        )

        total_profit = float(horizon_df["gross_profit"].sum())
        avg_churn = float(horizon_df["churn_risk"].mean())
        m1, m2 = st.columns(2)
        m1.metric("Profit cumule horizon", f"R${total_profit:,.0f}")
        m2.metric("Risque churn moyen", f"{100.0 * avg_churn:.1f}%")

    with tab4:
        st.markdown("### Test de robustesse Monte Carlo")
        n_runs = st.slider("Nombre de simulations", min_value=100, max_value=1500, value=400, step=100)
        mc_df = run_monte_carlo(
            dqn_model=dqn_model,
            profile=profile,
            current_price=current_price,
            month=month,
            seasonality_idx=seasonality_idx,
            review_score=review_score,
            marketing_boost_pct=marketing_boost,
            demand_shock_pct=demand_shock,
            stock_pressure=stock_pressure,
            unit_cost_shock_pct=cost_shock,
            n_runs=n_runs,
            seed=42,
        )

        p5 = float(mc_df["gross_profit"].quantile(0.05))
        p50 = float(mc_df["gross_profit"].quantile(0.50))
        p95 = float(mc_df["gross_profit"].quantile(0.95))
        win_rate = float((mc_df["gross_profit"] > 0).mean())

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Profit P5", f"R${p5:,.0f}")
        r2.metric("Profit median", f"R${p50:,.0f}")
        r3.metric("Profit P95", f"R${p95:,.0f}")
        r4.metric("Probabilite profit > 0", f"{100.0 * win_rate:.1f}%")

        action_mix = mc_df.groupby("action", as_index=False).size().rename(columns={"size": "count"})
        action_mix["share"] = action_mix["count"] / action_mix["count"].sum()
        st.bar_chart(action_mix.set_index("action")["share"], use_container_width=True)

        st.markdown("### Interpretation prix vs demande")
        curve = build_price_curve(
            profile=profile,
            month=month,
            seasonality_idx=seasonality_idx,
            review_score=review_score,
            marketing_boost_pct=marketing_boost,
            demand_shock_pct=demand_shock,
            stock_pressure=stock_pressure,
            unit_cost_shock_pct=cost_shock,
            p10=p10,
            p90=p90,
        )

        correlation = float(curve["price"].corr(curve["expected_orders"]))
        st.metric("Correlation prix-demande", f"{correlation:.3f}")
        st.line_chart(curve.set_index("price")[["expected_orders", "gross_profit"]], use_container_width=True)

    with st.expander("Details calibration categorie"):
        st.write(
            {
                "category": profile.category,
                "base_price": round(profile.base_price, 2),
                "base_demand_monthly": round(profile.base_demand, 2),
                "elasticity": round(profile.elasticity, 3),
                "base_review": round(profile.base_review, 3),
                "unit_cost_proxy": round(profile.unit_cost, 2),
                "seasonality_index_month": round(seasonality_idx, 3),
                "scenario_preset": preset_name,
            }
        )

    st.caption(
        "Ce simulateur web utilise uniquement le checkpoint DQN entraine pour la selection d'action."
    )


if __name__ == "__main__":
    main()
