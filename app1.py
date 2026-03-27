#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flight Delay Modeling + PSO + NLP Query Interface (CLI Version)

- Uses your HFlights-style dataset (data.csv by default).
- Baseline RandomForest + XGBoost/GradientBoosting.
- PSO (Particle Swarm Optimization) to tune RF hyperparameters.
- Baseline vs Optimized metrics (Accuracy, Precision, Recall, F1, AUC).
- NLP query engine using Sentence-BERT:
    - Statistics (mean/median/count/distribution/trend)
    - Optimization (best/least congested departure slots)
    - Prediction (delay probability)

Run:
    python flight_cli_nlp.py --data data.csv
"""

import argparse
import os
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# Sentence-BERT
from sentence_transformers import SentenceTransformer, util

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


# ------------------------------
# Basic utilities
# ------------------------------
def parse_time_hhmm(x):
    """Convert numeric HHMM (e.g., 1405) into (hour, minute)."""
    if pd.isna(x):
        return (np.nan, np.nan)
    try:
        s = str(int(x)).zfill(4)
        hh = int(s[:-2])
        mm = int(s[-2:])
        if 0 <= hh < 24 and 0 <= mm < 60:
            return hh, mm
    except Exception:
        pass
    return (np.nan, np.nan)


# ------------------------------
# Airport Configuration
# ------------------------------
AIRPORT_CONFIG = {
    "IAH": {  # Houston George Bush Intercontinental
        "runways": 5,
        "capacity_per_hour": 70,
        "peak_hours": [7, 8, 9, 16, 17, 18, 19],
        "weather_delays": 0.10,
        "ground_congestion": 0.7,
    },
    "DFW": {  # Dallas/Fort Worth
        "runways": 7,
        "capacity_per_hour": 90,
        "peak_hours": [7, 8, 9, 17, 18, 19, 20],
        "weather_delays": 0.12,
        "ground_congestion": 0.8,
    },
    "HOU": {  # Houston Hobby
        "runways": 4,
        "capacity_per_hour": 40,
        "peak_hours": [7, 8, 16, 17, 18],
        "weather_delays": 0.09,
        "ground_congestion": 0.5,
    },
}


# ------------------------------
# Column descriptions (for NLP)
# ------------------------------
COLUMN_DESCRIPTIONS = {
    "Year": "year of flight",
    "Month": "month number 1..12",
    "DayofMonth": "day of the month 1..31",
    "DayOfWeek": "day of week as integer (1=Mon, 7=Sun)",
    "DepTime": "actual departure time in HHMM format",
    "ArrTime": "actual arrival time in HHMM format",
    "UniqueCarrier": "airline carrier code",
    "FlightNum": "unique flight number",
    "TailNum": "aircraft tail number identifier",
    "ActualElapsedTime": "total elapsed time in minutes",
    "AirTime": "time spent airborne in minutes",
    "ArrDelay": "difference between scheduled and actual arrival time in minutes (negative for early)",
    "DepDelay": "difference between scheduled and actual departure time in minutes",
    "Origin": "origin airport code",
    "Dest": "destination airport code",
    "Distance": "distance flown (miles)",
    "TaxiIn": "taxi-in time in minutes",
    "TaxiOut": "taxi-out time in minutes",
    "Cancelled": "1 if flight cancelled else 0",
    "CancellationCode": "reason code for cancellation",
    "Diverted": "1 if flight diverted else 0",
}

OPERATIONS = {
    "mean": "average / mean",
    "median": "median",
    "max": "maximum",
    "min": "minimum",
    "count": "count / how many",
    "top": "top N / most frequent",
    "distribution": "distribution / histogram",
    "percent": "percentage (for binary columns)",
    "trend": "time trend (by hour or by day)",
    "optimize": "find optimal schedule slots",
    "capacity": "runway capacity analysis",
    "congestion": "congestion analysis",
}


# ------------------------------
# SBERT + embeddings
# ------------------------------
def load_sbert(model_name: str = "all-MiniLM-L6-v2"):
    print("[INFO] Loading Sentence-BERT model:", model_name)
    return SentenceTransformer(model_name)


def precompute_embeddings(model):
    col_keys = list(COLUMN_DESCRIPTIONS.keys())
    col_texts = [f"{k}: {v}" for k, v in COLUMN_DESCRIPTIONS.items()]
    col_emb = model.encode(col_texts, convert_to_tensor=True)

    op_keys = list(OPERATIONS.keys())
    op_texts = [f"{k}: {v}" for k, v in OPERATIONS.items()]
    op_emb = model.encode(op_texts, convert_to_tensor=True)

    return col_keys, col_emb, op_keys, op_emb


# ------------------------------
# Query parsing helpers
# ------------------------------
MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        start=1,
    )
}
WEEKDAYS = {
    "monday": 1,
    "tuesday": 2,
    "wednesday": 3,
    "thursday": 4,
    "friday": 5,
    "saturday": 6,
    "sunday": 7,
}


def detect_month(q: str):
    qq = q.lower()
    for k, v in MONTHS.items():
        if k in qq:
            return v
    m = re.search(
        r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b",
        qq,
    )
    if m:
        short = m.group(1)
        return {
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }[short]
    return None


def detect_dayofweek(q: str):
    qq = q.lower()
    for k, v in WEEKDAYS.items():
        if k in qq:
            return v
    if "weekday" in qq:
        return "weekday"
    if "weekend" in qq:
        return "weekend"
    return None


def detect_hour_range(q: str):
    qq = q.lower()
    # "4 to 7", "16-19", "4pm to 7pm"
    m = re.search(
        r"(\d{1,2})(?:[:h]?| ?(?:am|pm)?)\s*(?:-|to|and|until)\s*(\d{1,2})(?:[:h]?| ?(?:am|pm)?)",
        qq,
    )
    if m:
        a = int(m.group(1))
        b = int(m.group(2))
        return min(a, b), max(a, b)
    m2 = re.search(r"\b(at|around|@ )?(\d{1,2})(?:am|pm)?\b", qq)
    if m2:
        return int(m2.group(2)), int(m2.group(2))
    return None


def detect_origin_dest_carrier(q: str, df: pd.DataFrame):
    qq = q.lower()
    origin = None
    dest = None
    carrier = None

    city_to_code = {
        "houston": "IAH",
        "dallas": "DFW",
        "hobby": "HOU",
    }

    for city, code in city_to_code.items():
        if city in qq:
            if re.search(r"from\s+" + re.escape(city), qq):
                origin = code
            elif re.search(r"to\s+" + re.escape(city), qq):
                dest = code
            elif origin is None:
                origin = code

    airport_codes = list(pd.concat([df["Origin"], df["Dest"]]).dropna().unique())
    for code in airport_codes:
        code_l = str(code).lower()
        if code_l in qq:
            if re.search(r"from\s+" + re.escape(code_l), qq):
                origin = code
            elif re.search(r"to\s+" + re.escape(code_l), qq):
                dest = code
            elif origin is None:
                origin = code

    if "UniqueCarrier" in df.columns:
        for c in df["UniqueCarrier"].dropna().astype(str).unique():
            if c.lower() in qq:
                carrier = c
                break

    return origin, dest, carrier


def _determine_intent(query: str, top_op: str, op_score: float) -> str:
    q_lower = query.lower()
    optimization_keywords = [
        "optimize",
        "best time",
        "optimal slot",
        "recommend",
        "suggest",
        "when should",
        "best departure",
        "least congested",
    ]
    if any(k in q_lower for k in optimization_keywords) or (
        top_op == "optimize" and op_score > 0.4
    ):
        return "optimize"

    prediction_keywords = [
        "will",
        "predict",
        "probability",
        "chance",
        "delayed?",
        "what are the chances",
    ]
    if any(k in q_lower for k in prediction_keywords):
        if "delay" in q_lower or "late" in q_lower or "on time" in q_lower:
            return "predict"

    return "statistics"


def map_query(query: str, model, col_keys, col_emb, op_keys, op_emb, df):
    q_emb = model.encode(query, convert_to_tensor=True)
    col_scores = util.pytorch_cos_sim(q_emb, col_emb)[0].cpu().tolist()
    col_with_scores = sorted(
        [(col_keys[i], float(col_scores[i])) for i in range(len(col_keys))],
        key=lambda x: -x[1],
    )
    top_cols = [c for c, s in col_with_scores[:3] if s > 0.28]

    op_scores = util.pytorch_cos_sim(q_emb, op_emb)[0].cpu().tolist()
    op_with_scores = sorted(
        [(op_keys[i], float(op_scores[i])) for i in range(len(op_keys))],
        key=lambda x: -x[1],
    )
    top_op = op_with_scores[0][0] if op_with_scores and op_with_scores[0][1] > 0.22 else None
    top_op_score = op_with_scores[0][1] if op_with_scores and op_with_scores[0][1] > 0.22 else 0.0

    month = detect_month(query)
    dow = detect_dayofweek(query)
    hour_range = detect_hour_range(query)
    origin, dest, carrier = detect_origin_dest_carrier(query, df)

    intent = _determine_intent(query, top_op, top_op_score)

    groupby = None
    q_lower = query.lower()
    if "by " in q_lower or " per " in q_lower or "each " in q_lower:
        if "carrier" in q_lower:
            groupby = "UniqueCarrier"
        elif "origin" in q_lower:
            groupby = "Origin"
        elif "dest" in q_lower:
            groupby = "Dest"
        else:
            for c in top_cols:
                if c in ["UniqueCarrier", "Origin", "Dest", "TailNum", "Month", "DayOfWeek"]:
                    groupby = c
                    break

    return {
        "top_cols": top_cols,
        "col_expl": col_with_scores[:5],
        "top_op": top_op,
        "op_expl": op_with_scores[:4],
        "filters": {
            "month": month,
            "dayofweek": dow,
            "hour_range": hour_range,
            "origin": origin,
            "dest": dest,
            "carrier": carrier,
        },
        "groupby": groupby,
        "intent": intent,
    }


# ------------------------------
# Statistics execution
# ------------------------------
def execute_stats(mapping, df, raw_query):
    top_cols = mapping["top_cols"]
    op = mapping["top_op"]
    grp = mapping["groupby"]
    filters = mapping["filters"]

    df2 = df.copy()

    if filters["month"]:
        df2 = df2[df2["Month"] == filters["month"]]

    if filters["dayofweek"]:
        if filters["dayofweek"] == "weekday":
            df2 = df2[df2["DayOfWeek"].isin([1, 2, 3, 4, 5])]
        elif filters["dayofweek"] == "weekend":
            df2 = df2[df2["DayOfWeek"].isin([6, 7])]
        else:
            df2 = df2[df2["DayOfWeek"] == filters["dayofweek"]]

    if filters["origin"]:
        df2 = df2[df2["Origin"] == filters["origin"]]
    if filters["dest"]:
        df2 = df2[df2["Dest"] == filters["dest"]]
    if filters["carrier"]:
        df2 = df2[df2["UniqueCarrier"] == filters["carrier"]]

    if filters["hour_range"]:
        a, b = filters["hour_range"]
        if "DepHour" not in df2.columns and "DepTime" in df2.columns:
            dh, dm = zip(*df2["DepTime"].apply(parse_time_hhmm))
            df2["DepHour"] = pd.Series(dh, index=df2.index)
        df2 = df2[df2["DepHour"].between(a, b)]

    agg_col = top_cols[0] if top_cols else "ArrDelay"
    if agg_col not in df2.columns:
        agg_col = "ArrDelay"

    try:
        if op in ["mean", "median", "max", "min", "percent"]:
            if grp:
                series = df2.groupby(grp)[agg_col]
                res = series.mean() * 100 if op == "percent" else getattr(series, op)()
                result = res.reset_index()
            else:
                val = df2[agg_col].mean() * 100 if op == "percent" else getattr(df2[agg_col], op)()
                result = pd.DataFrame([{f"{op}_{agg_col}": val}])
        elif op == "count":
            result = df2.groupby(grp).size().reset_index(name="count") if grp else pd.DataFrame(
                [{"count": len(df2)}]
            )
        elif op == "top":
            result = df2[agg_col].value_counts().head(10).reset_index()
        elif op == "distribution":
            result = {"hist_values": df2[agg_col].dropna().tolist()}
        elif op == "trend":
            if "DepHour" not in df2.columns and "DepTime" in df2.columns:
                dh, dm = zip(*df2["DepTime"].apply(parse_time_hhmm))
                df2["DepHour"] = pd.Series(dh, index=df2.index)
            result = (
                df2.groupby("DepHour")[agg_col].mean()
                .reindex(range(0, 24))
                .reset_index()
            )
        else:
            # fallback: show a sample
            cols = [agg_col] + [c for c in ["UniqueCarrier", "Origin", "Dest", "DepHour"] if c in df2.columns]
            result = df2[cols].head(20)
    except Exception as e:
        result = pd.DataFrame([{"error": str(e)}])

    return result


# ------------------------------
# Data prep for ML
# ------------------------------
def prepare_ml_data(df: pd.DataFrame):
    df = df.copy()

    if "DepHour" not in df.columns and "DepTime" in df.columns:
        dh, dm = zip(*df["DepTime"].apply(parse_time_hhmm))
        df["DepHour"] = pd.Series(dh, index=df.index)

    if "GroundTime" not in df.columns:
        df["GroundTime"] = df["TaxiIn"].fillna(0) + df["TaxiOut"].fillna(0)

    df["IsPeakHour"] = df["DepHour"].apply(
        lambda x: 1 if x in [7, 8, 9, 16, 17, 18, 19] else 0
    )
    df["IsWeekend"] = df["DayOfWeek"].apply(lambda x: 1 if x in [6, 7] else 0)
    df["IsCongestedAirport"] = df["Origin"].apply(
        lambda x: 1 if x in ["IAH", "DFW"] else 0
    )
    df["ArrDelayBinary"] = (df["ArrDelay"].fillna(0) > 15).astype(int)

    feats = [
        "Month",
        "DayOfWeek",
        "DepHour",
        "Distance",
        "AirTime",
        "GroundTime",
        "IsPeakHour",
        "IsWeekend",
        "IsCongestedAirport",
    ]
    X = df[feats].fillna(0)
    y = df["ArrDelayBinary"]
    return X, y, feats


# ------------------------------
# Baseline ML models
# ------------------------------
def train_baseline_models(df: pd.DataFrame, random_state: int = 42):
    X, y, feats = prepare_ml_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    # RandomForest (baseline)
    rf_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    max_features=1.0,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf_pipe.fit(X_train, y_train)
    y_pred_rf = rf_pipe.predict(X_test)
    y_prob_rf = rf_pipe.predict_proba(X_test)[:, 1]

    rf_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred_rf),
        "Precision": precision_score(y_test, y_pred_rf, zero_division=0),
        "Recall": recall_score(y_test, y_pred_rf, zero_division=0),
        "F1": f1_score(y_test, y_pred_rf, zero_division=0),
    }
    try:
        if len(np.unique(y_test)) == 2:
            rf_metrics["AUC"] = roc_auc_score(y_test, y_prob_rf)
        else:
            rf_metrics["AUC"] = np.nan
    except Exception:
        rf_metrics["AUC"] = np.nan

    # Second model: XGBoost or GradientBoosting
    if XGBOOST_AVAILABLE:
        model_b = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            n_estimators=100,
            random_state=random_state,
            max_depth=5,
        )
        model_b_name = "XGBoost"
    else:
        model_b = GradientBoostingClassifier(
            n_estimators=100, random_state=random_state, max_depth=5
        )
        model_b_name = "GradientBoosting"

    model_b.fit(X_train, y_train)
    y_pred_b = model_b.predict(X_test)
    y_prob_b = model_b.predict_proba(X_test)[:, 1]

    b_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred_b),
        "Precision": precision_score(y_test, y_pred_b, zero_division=0),
        "Recall": recall_score(y_test, y_pred_b, zero_division=0),
        "F1": f1_score(y_test, y_pred_b, zero_division=0),
    }
    try:
        if len(np.unique(y_test)) == 2:
            b_metrics["AUC"] = roc_auc_score(y_test, y_prob_b)
        else:
            b_metrics["AUC"] = np.nan
    except Exception:
        b_metrics["AUC"] = np.nan

    return {
        "rf_pipe": rf_pipe,
        "model_b": model_b,
        "meta": {
            "features": feats,
            "rf_metrics": rf_metrics,
            "b_metrics": b_metrics,
            "model_b_name": model_b_name,
        },
    }


def ensemble_predict(rf_pipe, model_b, meta, input_row: pd.DataFrame):
    Xrow = input_row[meta["features"]].fillna(0)
    rf_prob = rf_pipe.predict_proba(Xrow)[:, 1]
    b_prob = model_b.predict_proba(Xrow)[:, 1]
    avg_prob = (rf_prob * 0.5 + b_prob * 0.5)
    return {
        "rf_prob": float(rf_prob[0]),
        "b_prob": float(b_prob[0]),
        "avg_prob": float(avg_prob[0]),
        "confidence": float(abs(avg_prob[0] - 0.5) * 2),
    }


# ------------------------------
# PSO to tune RF hyperparameters
# ------------------------------
def _rf_accuracy_for_params(X_train, X_val, y_train, y_val, params):
    n_estimators = int(round(params["n_estimators"]))
    max_depth = int(round(params["max_depth"]))
    max_features = float(params["max_features"])

    n_estimators = max(10, min(300, n_estimators))
    max_depth = max(3, min(30, max_depth))
    max_features = max(0.2, min(1.0, max_features))

    rf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    max_features=max_features,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    return accuracy_score(y_val, y_pred)


def pso_optimize_rf(
    df: pd.DataFrame, n_particles: int = 10, n_iters: int = 15, random_state: int = 42
):
    rng = np.random.RandomState(random_state)
    X, y, feats = prepare_ml_data(df)

    # Subsample for speed if dataset is large
    if len(X) > 20000:
        idx = rng.choice(len(X), size=20000, replace=False)
        X = X.iloc[idx]
        y = y.iloc[idx]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    bounds = {
        "n_estimators": (50, 200),
        "max_depth": (5, 20),
        "max_features": (0.3, 1.0),
    }
    keys = list(bounds.keys())
    dim = len(bounds)

    lb = np.array([bounds[k][0] for k in keys], dtype=float)
    ub = np.array([bounds[k][1] for k in keys], dtype=float)

    w = 0.7
    c1 = 1.5
    c2 = 1.5

    positions = lb + (ub - lb) * rng.rand(n_particles, dim)
    velocities = rng.uniform(-1, 1, size=(n_particles, dim))

    personal_best_positions = positions.copy()
    personal_best_scores = np.zeros(n_particles)

    for i in range(n_particles):
        params = {k: positions[i, j] for j, k in enumerate(keys)}
        personal_best_scores[i] = _rf_accuracy_for_params(
            X_train, X_val, y_train, y_val, params
        )

    gbest_idx = int(np.argmax(personal_best_scores))
    global_best_position = personal_best_positions[gbest_idx].copy()
    global_best_score = personal_best_scores[gbest_idx]

    convergence = [global_best_score]

    for _ in range(n_iters):
        for i in range(n_particles):
            r1 = rng.rand(dim)
            r2 = rng.rand(dim)

            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (personal_best_positions[i] - positions[i])
                + c2 * r2 * (global_best_position - positions[i])
            )
            positions[i] = positions[i] + velocities[i]
            positions[i] = np.clip(positions[i], lb, ub)

            params = {k: positions[i, j] for j, k in enumerate(keys)}
            score = _rf_accuracy_for_params(X_train, X_val, y_train, y_val, params)

            if score > personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = positions[i].copy()
                if score > global_best_score:
                    global_best_score = score
                    global_best_position = positions[i].copy()

        convergence.append(global_best_score)

    best_params = {k: global_best_position[j] for j, k in enumerate(keys)}
    best_params["n_estimators"] = int(round(best_params["n_estimators"]))
    best_params["max_depth"] = int(round(best_params["max_depth"]))
    best_params["max_features"] = float(best_params["max_features"])

    return best_params, float(global_best_score), convergence


def compute_rf_metrics(df: pd.DataFrame, params: Dict[str, float], random_state: int = 42):
    X, y, feats = prepare_ml_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    n_estimators = int(round(params.get("n_estimators", 100)))
    max_depth = int(round(params.get("max_depth", 10)))
    max_features = float(params.get("max_features", 1.0))

    rf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    max_features=max_features,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
    }
    try:
        if len(np.unique(y_test)) == 2:
            metrics["AUC"] = roc_auc_score(y_test, y_prob)
        else:
            metrics["AUC"] = np.nan
    except Exception:
        metrics["AUC"] = np.nan

    return metrics


# ------------------------------
# Schedule optimizer (text-only)
# ------------------------------
class ScheduleOptimizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._prepare_data()

    def _prepare_data(self):
        if "DepHour" not in self.df.columns and "DepTime" in self.df.columns:
            dh, dm = zip(*self.df["DepTime"].apply(parse_time_hhmm))
            self.df["DepHour"] = pd.Series(dh, index=self.df.index)
        self._calculate_congestion_metrics()
        self._calculate_delay_patterns()

    def _calculate_congestion_metrics(self):
        congestion_data = []
        for airport in AIRPORT_CONFIG.keys():
            if (
                "Origin" in self.df.columns
                and "Dest" in self.df.columns
                and (
                    airport in self.df["Origin"].values
                    or airport in self.df["Dest"].values
                )
            ):
                flights = self.df[
                    (self.df["Origin"] == airport) | (self.df["Dest"] == airport)
                ]
                hourly_counts = flights.groupby("DepHour").size()
                config = AIRPORT_CONFIG[airport]
                max_capacity = config["runways"] * config["capacity_per_hour"]
                for hour in range(24):
                    count = hourly_counts.get(hour, 0)
                    util = count / max_capacity if max_capacity > 0 else 0
                    congestion_data.append(
                        {
                            "Airport": airport,
                            "Hour": hour,
                            "FlightCount": count,
                            "Utilization": min(util, 1.0),
                            "CongestionLevel": self._get_congestion_level(
                                util, config, hour
                            ),
                        }
                    )
        self.congestion_df = (
            pd.DataFrame(congestion_data)
            if congestion_data
            else pd.DataFrame(
                columns=["Airport", "Hour", "FlightCount", "Utilization", "CongestionLevel"]
            )
        )

    def _get_congestion_level(self, utilization, config, hour):
        base = utilization
        if hour in config["peak_hours"]:
            base *= 1.3
        base *= (1 + config["weather_delays"])
        base *= config["ground_congestion"]
        return min(base, 1.0)

    def _calculate_delay_patterns(self):
        delay_col = "ArrDelay" if "ArrDelay" in self.df.columns else "DepDelay"
        patterns = []
        for airport in AIRPORT_CONFIG.keys():
            dfa = self.df[self.df["Origin"] == airport]
            if len(dfa) == 0 or "DepHour" not in dfa.columns:
                continue
            stats = dfa.groupby("DepHour")[delay_col].agg(
                ["mean", "median", "std", "count"]
            ).fillna(0)
            for hour in range(24):
                if hour in stats.index:
                    s = stats.loc[hour]
                    patterns.append(
                        {
                            "Airport": airport,
                            "Hour": hour,
                            "AvgDelay": s["mean"],
                            "MedianDelay": s["median"],
                            "DelayStd": s["std"],
                            "FlightCount": s["count"],
                            "DelayRisk": self._delay_risk(s),
                        }
                    )
        self.delay_patterns_df = pd.DataFrame(patterns)

    @staticmethod
    def _delay_risk(stats_row):
        if stats_row["count"] == 0:
            return 0.0
        avg_delay = max(0, stats_row["mean"])
        std_delay = stats_row["std"]
        risk = (avg_delay + std_delay) / 100.0
        return min(risk, 1.0)

    def find_optimal_slots(self, airport: str) -> List[Dict]:
        recs = []
        for hour in range(24):
            c = self.congestion_df[
                (self.congestion_df["Airport"] == airport)
                & (self.congestion_df["Hour"] == hour)
            ]
            d = self.delay_patterns_df[
                (self.delay_patterns_df["Airport"] == airport)
                & (self.delay_patterns_df["Hour"] == hour)
            ]
            if c.empty or d.empty:
                continue
            c = c.iloc[0]
            d = d.iloc[0]
            score = (
                c["CongestionLevel"] * 0.4
                + d["DelayRisk"] * 0.4
                + c["Utilization"] * 0.2
            )
            recs.append(
                {
                    "Hour": hour,
                    "Score": score,
                    "FlightCount": c["FlightCount"],
                    "AvgDelay": d["AvgDelay"],
                }
            )
        recs.sort(key=lambda x: x["Score"])
        return recs


# ------------------------------
# CLI / REPL helpers
# ------------------------------
def print_metrics_table(title: str, metrics: Dict[str, float]):
    print(f"\n=== {title} ===")
    for k, v in metrics.items():
        print(f"{k:10s}: {v:.4f}")


def handle_optimize_query(mapping, df, optimizer: ScheduleOptimizer):
    filters = mapping["filters"]
    airport = filters["origin"] or "IAH"
    if airport not in AIRPORT_CONFIG:
        airport = "IAH"
    print(f"\n[OPTIMIZATION] Finding best departure slots for {airport} ...")
    recs = optimizer.find_optimal_slots(airport)
    if not recs:
        print("No sufficient data to compute recommendations.")
        return
    print("Top 5 recommended hours (lower score = better):")
    for r in recs[:5]:
        print(
            f"  Hour {r['Hour']:02d}: Score={r['Score']:.3f}, "
            f"Flights={int(r['FlightCount'])}, AvgDelay={r['AvgDelay']:.2f} min"
        )


def handle_predict_query(mapping, df, artifacts):
    print("\n[PREDICTION] Delay probability query")

    filters = mapping["filters"]
    origin = filters["origin"] or "IAH"
    hour_range = filters["hour_range"]
    dow = filters["dayofweek"]

    if hour_range:
        dep_hour = hour_range[0]
    else:
        dep_hour = int(input("Departure hour (0-23)? "))

    if isinstance(dow, int):
        day_of_week = dow
    else:
        day_of_week = int(input("Day of week (1=Mon..7=Sun)? "))

    rf_pipe = artifacts["rf_pipe"]
    model_b = artifacts["model_b"]
    meta = artifacts["meta"]

    input_features = pd.DataFrame(
        [
            {
                "Month": 6,
                "DayOfWeek": day_of_week,
                "DepHour": dep_hour,
                "Distance": df["Distance"].median(),
                "AirTime": df["AirTime"].median(),
                "GroundTime": df["TaxiIn"].median() + df["TaxiOut"].median(),
                "IsPeakHour": 1
                if dep_hour in AIRPORT_CONFIG.get(origin, {}).get("peak_hours", [])
                else 0,
                "IsWeekend": 1 if day_of_week > 5 else 0,
                "IsCongestedAirport": 1 if origin in ["IAH", "DFW"] else 0,
            }
        ]
    )

    result = ensemble_predict(rf_pipe, model_b, meta, input_features)
    p = result["avg_prob"]
    print(f"Predicted delay probability: {p:.2%}")
    if p < 0.3:
        print("Risk level: LOW")
    elif p < 0.6:
        print("Risk level: MODERATE")
    else:
        print("Risk level: HIGH")


def handle_stats_query(mapping, df, raw_query):
    print("\n[STATISTICS] Running statistical analysis ...")
    result = execute_stats(mapping, df, raw_query)
    if isinstance(result, pd.DataFrame):
        print(result.head(20).to_string(index=False))
        if len(result) > 20:
            print(f"... ({len(result)} rows total, showing first 20)")
    elif isinstance(result, dict) and "hist_values" in result:
        vals = result["hist_values"]
        print(f"Histogram over {len(vals)} values.")
        print(f"  mean: {np.mean(vals):.3f}")
        print(f"  std : {np.std(vals):.3f}")
        print(f"  min : {np.min(vals):.3f}")
        print(f"  max : {np.max(vals):.3f}")
    else:
        print("No results found / unknown format.")


# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.csv", help="Path to CSV dataset")
    parser.add_argument("--particles", type=int, default=10, help="PSO particles")
    parser.add_argument("--iters", type=int, default=15, help="PSO iterations")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"[ERROR] Data file not found: {args.data}")
        return

    print(f"[INFO] Loading dataset: {args.data}")
    df = pd.read_csv(args.data)
    print(f"[INFO] Rows: {len(df)}, Columns: {len(df.columns)}")
    print("[INFO] Columns:", ", ".join(df.columns.astype(str).tolist()))

    # Ensure DepHour exists
    if "DepHour" not in df.columns and "DepTime" in df.columns:
        dh, dm = zip(*df["DepTime"].apply(parse_time_hhmm))
        df["DepHour"] = pd.Series(dh, index=df.index)

    # Baseline ML
    print("\n>>> Training baseline ML models (RandomForest + XGBoost/GB) ...")
    artifacts = train_baseline_models(df)
    meta = artifacts["meta"]
    print_metrics_table("Baseline RandomForest", meta["rf_metrics"])
    print_metrics_table(f"Baseline {meta['model_b_name']}", meta["b_metrics"])

    # PSO optimization for RF
    print(
        f"\n>>> Running PSO for RF hyperparameters (particles={args.particles}, iterations={args.iters}) ..."
    )
    best_params, best_acc, convergence = pso_optimize_rf(
        df, n_particles=args.particles, n_iters=args.iters
    )
    print("\n[PSO] Best RF hyperparameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"[PSO] Best validation accuracy: {best_acc:.4f}")

    baseline_params = {"n_estimators": 100, "max_depth": 10, "max_features": 1.0}
    baseline_rf_metrics = compute_rf_metrics(df, baseline_params)
    optimized_rf_metrics = compute_rf_metrics(df, best_params)

    print("\n=== Baseline RF vs Optimized RF (Test metrics) ===")
    header = f"{'Metric':10s} | {'Baseline':>10s} | {'Optimized':>10s}"
    print(header)
    print("-" * len(header))
    for k in baseline_rf_metrics.keys():
        print(
            f"{k:10s} | {baseline_rf_metrics[k]:10.4f} | {optimized_rf_metrics[k]:10.4f}"
        )

    # PSO convergence plot
    plt.figure(figsize=(6, 4))
    plt.plot(convergence, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Best Validation Accuracy")
    plt.title("PSO Convergence (RF Hyperparameters)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("pso_convergence.png")
    plt.close()
    print("\n[INFO] Saved PSO convergence plot: pso_convergence.png")

    # Schedule optimizer
    optimizer = ScheduleOptimizer(df)

    # NLP model & embeddings
    sbert = load_sbert()
    col_keys, col_emb, op_keys, op_emb = precompute_embeddings(sbert)

    print(
        "\n=== NLP Query Interface ===\n"
        "Type natural-language queries like:\n"
        "  - 'What is the average arrival delay from IAH?'\n"
        "  - 'When is the best time to depart from IAH?'\n"
        "  - 'Predict delay for a flight from IAH at 8 AM on Sunday'\n"
        "Type 'exit' to quit.\n"
    )

    while True:
        q = input("Query> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        mapping = map_query(q, sbert, col_keys, col_emb, op_keys, op_emb, df)
        print("\n[DEBUG] Intent:", mapping["intent"])
        print("[DEBUG] Filters:", mapping["filters"])
        print("[DEBUG] Top cols:", mapping["top_cols"], "Top op:", mapping["top_op"])

        if mapping["intent"] == "optimize":
            handle_optimize_query(mapping, df, optimizer)
        elif mapping["intent"] == "predict":
            handle_predict_query(mapping, df, artifacts)
        else:
            handle_stats_query(mapping, df, q)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
