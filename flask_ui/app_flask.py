from __future__ import annotations

import os
import pickle
import sys
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
from flask import Flask, flash, redirect, render_template, request, session, url_for
from psycopg2.extras import Json, RealDictCursor

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from recommendation_engine import (
    classify_satisfaction_level,
    get_recommendation_plans,
    rank_recommendation_plans_by_impact,
)

PREDICTION_LABELS = {
    "Satisfied": "Hài lòng",
    "Dissatisfied": "Không hài lòng",
}

RISK_LABELS = {
    "LOW": "THẤP",
    "MEDIUM": "TRUNG BÌNH",
    "HIGH": "CAO",
}

FEATURE_CONFIG = {
    "Business": [
        "Departure Delay in Minutes",
        "Arrival Delay in Minutes",
        "Seat comfort",
        "Leg room service",
        "Inflight service",
        "Cleanliness",
    ],
    "Economy": [
        "Food and drink",
        "Inflight entertainment",
        "Ease of Online booking",
        "Online boarding",
        "Seat comfort",
        "Cleanliness",
    ],
    "Eco": [
        "Food and drink",
        "Inflight entertainment",
        "Ease of Online booking",
        "Online boarding",
        "Seat comfort",
        "Cleanliness",
    ],
    "Eco Plus": [
        "Seat comfort",
        "Leg room service",
        "Cleanliness",
        "Inflight service",
        "Food and drink",
        "Inflight entertainment",
    ],
}

AHP_CRITERIA = {
    "Business": [
        "Inflight wifi service",
        "Departure/Arrival time convenient",
        "Ease of Online booking",
        "Gate location",
        "Baggage handling",
        "On-board service",
    ],
    "Economy": [
        "Food and drink",
        "Inflight entertainment",
        "Ease of Online booking",
        "Online boarding",
        "Seat comfort",
        "Cleanliness",
    ],
    "Eco Plus": [
        "Seat comfort",
        "Leg room service",
        "Cleanliness",
        "Inflight service",
        "Food and drink",
        "Inflight entertainment",
    ],
}

AHP_PAIRWISE_MATRICES = {
    "Business": [
        [1, 3, 3, 5, 5, 1],
        [1 / 3, 1, 1, 3, 3, 1 / 3],
        [1 / 3, 1, 1, 3, 3, 1 / 3],
        [1 / 5, 1 / 3, 1 / 3, 1, 1, 1 / 5],
        [1 / 5, 1 / 3, 1 / 3, 1, 1, 1 / 5],
        [1, 3, 3, 5, 5, 1],
    ],
    "Economy": [
        [1, 1, 1 / 5, 1 / 5, 1 / 3, 1 / 3],
        [1, 1, 1 / 5, 1 / 5, 1 / 3, 1 / 3],
        [5, 5, 1, 1, 3, 3],
        [5, 5, 1, 1, 3, 3],
        [3, 3, 1 / 3, 1 / 3, 1, 1],
        [3, 3, 1 / 3, 1 / 3, 1, 1],
    ],
    "Eco Plus": [
        [1, 1, 1, 1, 5, 5],
        [1, 1, 1, 1, 5, 5],
        [1, 1, 1, 1, 5, 5],
        [1, 1, 1, 1, 5, 5],
        [1 / 5, 1 / 5, 1 / 5, 1 / 5, 1, 1],
        [1 / 5, 1 / 5, 1 / 5, 1 / 5, 1, 1],
    ],
}

SAATY_SCALE = [
    {"value": "1", "meaning": "Quan trọng ngang nhau", "desc": "Hai tiêu chí đóng góp tương đương"},
    {"value": "3", "meaning": "Quan trọng vừa", "desc": "Tiêu chí i quan trọng hơn j ở mức vừa"},
    {"value": "5", "meaning": "Quan trọng mạnh", "desc": "Tiêu chí i vượt trội rõ"},
    {"value": "7", "meaning": "Quan trọng rất mạnh", "desc": "Tiêu chí i gần như áp đảo"},
    {"value": "9", "meaning": "Quan trọng tuyệt đối", "desc": "Tiêu chí i áp đảo hoàn toàn"},
    {"value": "1/x", "meaning": "Giá trị nghịch đảo", "desc": "Nếu j quan trọng hơn i thì dùng nghịch đảo"},
]

RI_TABLE = {
    1: 0.0,
    2: 0.0,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
}

DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "airline"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "123"),
}

FEATURE_VI = {
    # Passenger Metadata
    "Gender": "Giới tính",
    "Customer Type": "Loại khách hàng",
    "Age": "Độ tuổi",
    "Type of Travel": "Mục đích chuyến đi",
    
    # Booking & Checkin
    "Inflight wifi service": "Wifi trên chuyến bay",
    "Departure/Arrival time convenient": "Sự thuận tiện giờ bay",
    "Ease of Online booking": "Đặt vé trực tuyến",
    "Checkin service": "Dịch vụ Check-in",
    "Online boarding": "Lên máy bay trực tuyến",
    
    # At airport
    "Gate location": "Vị trí cổng lên máy bay",
    "Baggage handling": "Xử lý hành lý",
    
    # Inflight
    "Seat comfort": "Độ thoải mái của Ghế ngồi",
    "Inflight entertainment": "Giải trí trên chuyến bay",
    "On-board service": "Dịch vụ trên chuyến bay (On-board)",
    "Leg room service": "Không gian để chân",
    "Food and drink": "Đồ ăn và Thức uống",
    "Inflight service": "Dịch vụ tiếp viên (Inflight)",
    "Cleanliness": "Mức độ Sạch sẽ",
    
    # Delays (Numeric)
    "Departure Delay in Minutes": "Thời gian Trễ Khởi hành (Phút)",
    "Arrival Delay in Minutes": "Thời gian Trễ Đến nơi (Phút)",
}

FEATURE_ICONS = {
    # Passenger
    "Gender": "group",
    "Customer Type": "star_rate",
    "Age": "cake",
    "Type of Travel": "work",
    
    # Booking
    "Inflight wifi service": "wifi",
    "Departure/Arrival time convenient": "schedule",
    "Ease of Online booking": "phonelink_ring",
    "Checkin service": "how_to_reg",
    "Online boarding": "qr_code_scanner",
    
    # Airport
    "Gate location": "door_front",
    "Baggage handling": "luggage",
    
    # Inflight
    "Seat comfort": "chair",
    "Inflight entertainment": "movie",
    "On-board service": "room_service",
    "Leg room service": "airline_seat_legroom_extra",
    "Food and drink": "restaurant",
    "Inflight service": "dry_cleaning",
    "Cleanliness": "cleaning_services",
    
    # Delays
    "Departure Delay in Minutes": "flight_takeoff",
    "Arrival Delay in Minutes": "flight_land",
}

ADMIN_USER = os.getenv("AIRLINE_ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("AIRLINE_ADMIN_PASS", "admin123")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "airline-feedback-secret-key")


class AirlineDSS:
    def __init__(self, models_dict: Dict[str, Any], ahp_weights_dict: Dict[str, Any], feature_config: Dict[str, List[str]]):
        self.models = models_dict
        self.ahp_weights = ahp_weights_dict
        self.feature_config = feature_config

    def predict_satisfaction(self, passenger_data: pd.DataFrame, ticket_class: str) -> Dict[str, float | str]:
        if ticket_class not in self.models:
            raise ValueError(f"Hạng vé không hợp lệ: {ticket_class}")

        model = self.models[ticket_class]
        if model is None:
            raise ValueError(f"Chưa tải mô hình cho hạng vé: {ticket_class}")

        features = self.feature_config[ticket_class]
        x_data = passenger_data[features]

        prediction = model.predict(x_data)[0]
        probabilities = model.predict_proba(x_data)[0]

        return {
            "prediction": "Satisfied" if prediction == 1 else "Dissatisfied",
            "confidence": float(max(probabilities) * 100),
            "prob_dissatisfied": float(probabilities[0] * 100),
            "prob_satisfied": float(probabilities[1] * 100),
        }

    def calculate_feature_impact(self, passenger_data: pd.DataFrame, ticket_class: str) -> pd.DataFrame:
        model = self.models[ticket_class]
        features = self.feature_config[ticket_class]

        xgb_importance = model.feature_importances_

        # AHP files may not contain all current model features; align safely by key.
        ahp_weights_for_class = self.ahp_weights.get(ticket_class, {}) if isinstance(self.ahp_weights, dict) else {}
        ahp_dict = ahp_weights_for_class.get("weights", {}) if isinstance(ahp_weights_for_class, dict) else {}
        raw_ahp_values = np.array([float(ahp_dict.get(f, 0.0)) for f in features], dtype=float)
        if raw_ahp_values.sum() > 0:
            ahp_values = raw_ahp_values / raw_ahp_values.sum()
        else:
            ahp_values = np.full(len(features), 1.0 / len(features), dtype=float)

        feature_values = passenger_data[features].values[0]

        combined_weights = (xgb_importance * 0.5 + ahp_values * 0.5)
        impact_scores = combined_weights * feature_values
        total_impact = impact_scores.sum()
        if total_impact > 0:
            impact_percentage = (impact_scores / total_impact) * 100
        else:
            impact_percentage = np.zeros_like(impact_scores)

        impact_df = pd.DataFrame(
            {
                "Feature": features,
                "Current_Value": feature_values,
                "Combined_Weight": combined_weights,
                "Impact_Score": impact_scores,
                "Impact_%": impact_percentage,
            }
        )
        return impact_df.sort_values("Impact_Score", ascending=False).reset_index(drop=True)

    def generate_recommendations(self, passenger_data: pd.DataFrame, ticket_class: str) -> Dict[str, Any]:
        prediction_result = self.predict_satisfaction(passenger_data, ticket_class)
        impact_df = self.calculate_feature_impact(passenger_data, ticket_class)

        weak_features = impact_df[
            (impact_df["Current_Value"] <= 2)
            & (impact_df["Combined_Weight"] > impact_df["Combined_Weight"].median())
        ]

        priority_actions = []
        for _, row in weak_features.iterrows():
            f_name = row["Feature"]
            f_vi = FEATURE_VI.get(f_name, f_name)
            priority_actions.append(
                {
                    "feature": f_name,
                    "current_rating": float(row["Current_Value"]),
                    "impact": float(row["Impact_%"]),
                    "urgency": "HIGH" if row["Current_Value"] <= 1 else "MEDIUM",
                    "action": f"Cải thiện '{f_vi}' (Hiện tại {row['Current_Value']}/5)",
                }
            )

        prob_dissatisfied = float(prediction_result["prob_dissatisfied"])
        risk_score = prob_dissatisfied * 0.7 + (len(weak_features) / len(impact_df) * 100) * 0.3

        return {
            "ticket_class": ticket_class,
            "prediction": prediction_result,
            "impact_analysis": impact_df,
            "priority_actions": priority_actions,
            "risk_score": float(risk_score),
            "risk_level": "HIGH" if risk_score >= 70 else "MEDIUM" if risk_score >= 40 else "LOW",
        }


def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)


def init_database() -> None:
    try:
        # First connect to default postgres db to create airline db if not exists
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
        conn = psycopg2.connect(
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            dbname="postgres"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        cur.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{DB_CONFIG['dbname']}'")
        if not cur.fetchone():
            cur.execute(f"CREATE DATABASE {DB_CONFIG['dbname']}")
            print(f"[INFO] Database '{DB_CONFIG['dbname']}' created successfully.")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[WARN] Error while creating database (it may already exist): {e}")

    # Now connect to airline db and create tables
    query_feedback = """
    CREATE TABLE IF NOT EXISTS feedback_submissions (
        id SERIAL PRIMARY KEY,
        created_at TIMESTAMP NOT NULL,
        passenger_name TEXT NOT NULL,
        ticket_class VARCHAR(20) NOT NULL,
        prediction VARCHAR(30) NOT NULL,
        confidence DOUBLE PRECISION NOT NULL,
        risk_score DOUBLE PRECISION NOT NULL,
        risk_level VARCHAR(20) NOT NULL,
        ratings JSONB NOT NULL,
        upload_file_name TEXT,
        upload_file_size BIGINT
    );
    """

    query_admin = """
    CREATE TABLE IF NOT EXISTS admin_users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(50) UNIQUE NOT NULL,
        password VARCHAR(255) NOT NULL
    );
    """

    query_insert_admin = """
    INSERT INTO admin_users (username, password)
    VALUES ('admin', 'admin')
    ON CONFLICT (username) DO NOTHING;
    """

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query_feedback)
            cur.execute(query_admin)
            cur.execute(query_insert_admin)
        conn.commit()
    print("[INFO] Setup tables and default admin successfully.")


def save_submission_to_db(submission: Dict[str, Any]) -> None:
    query = """
    INSERT INTO feedback_submissions (
        created_at, passenger_name, ticket_class, prediction,
        confidence, risk_score, risk_level, ratings,
        upload_file_name, upload_file_size
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
    """

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                query,
                (
                    submission["created_at"],
                    submission["passenger_name"],
                    submission["ticket_class"],
                    submission["prediction"],
                    submission["confidence"],
                    submission["risk_score"],
                    submission["risk_level"],
                    Json(submission["ratings"]),
                    submission.get("upload_file_name"),
                    submission.get("upload_file_size"),
                ),
            )
        conn.commit()


def load_submissions() -> pd.DataFrame:
    query = """
    SELECT id, created_at, passenger_name, ticket_class, prediction,
           confidence, risk_score, risk_level, ratings,
           upload_file_name, upload_file_size
    FROM feedback_submissions
    ORDER BY created_at DESC;
    """

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            rows = cur.fetchall()
    return pd.DataFrame(rows)


def _file_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def load_ahp_weights() -> Optional[Dict[str, Any]]:
    base_dir = BASE_DIR
    ahp_path = base_dir / "ahp_weights_all_classes.pkl"
    if not ahp_path.exists():
        return None
    with ahp_path.open("rb") as file_obj:
        return pickle.load(file_obj)


def load_models() -> Dict[str, Any]:
    base_dir = BASE_DIR
    model_paths = {
        "Business": base_dir / "business_xgboost_optimized.pkl",
        "Economy": base_dir / "economy_xgboost_optimized.pkl",
        "Eco Plus": base_dir / "ecoplus_xgboost_optimized.pkl",
    }

    models: Dict[str, Any] = {}
    for class_name, path in model_paths.items():
        if path.exists():
            with path.open("rb") as file_obj:
                models[class_name] = pickle.load(file_obj)
        else:
            models[class_name] = None
    return models


def load_dss_system() -> Optional[AirlineDSS]:
    base_dir = BASE_DIR
    dss_path = base_dir / "airline_dss_system.pkl"
    if not dss_path.exists():
        return None

    class DSSUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str):
            if module == "__main__" and name == "AirlineDSS":
                return AirlineDSS
            return super().find_class(module, name)

    with dss_path.open("rb") as file_obj:
        return DSSUnpickler(file_obj).load()


def prepare_dss() -> tuple[Optional[AirlineDSS], Dict[str, Any], Optional[Dict[str, Any]]]:
    _ = _file_mtime(Path(__file__))
    ahp_weights = load_ahp_weights()
    models = load_models()
    dss_pickle = load_dss_system()

    if ahp_weights is not None and any(models.values()):
        dss = AirlineDSS(models_dict=models, ahp_weights_dict=ahp_weights, feature_config=FEATURE_CONFIG)
    elif dss_pickle is not None:
        dss = dss_pickle
    else:
        dss = None

    return dss, models, ahp_weights


def login_required(route_func):
    @wraps(route_func)
    def wrapper(*args, **kwargs):
        if not session.get("admin_authenticated", False):
            flash("Vui lòng đăng nhập bằng quyền Quản trị viên.", "warning")
            return redirect(url_for("admin_login"))
        return route_func(*args, **kwargs)

    return wrapper


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["prediction_vi"] = df["prediction"].replace(PREDICTION_LABELS)
    df["risk_level_vi"] = df["risk_level"].replace(RISK_LABELS)
    return df


def calculate_ahp_consistency(matrix: list[list[float]], weights: Optional[list[float]] = None) -> Dict[str, float | list[float]]:
    arr = np.array(matrix, dtype=float)

    if weights is None:
        eigenvalues, eigenvectors = np.linalg.eig(arr)
        max_index = int(np.argmax(np.real(eigenvalues)))
        principal = np.real(eigenvectors[:, max_index])
        computed_weights = principal / principal.sum()
    else:
        computed_weights = np.array(weights, dtype=float)

    weighted_sum = np.dot(arr, computed_weights)
    lambda_max = float(np.mean(weighted_sum / computed_weights))

    n = len(arr)
    ci = float((lambda_max - n) / (n - 1)) if n > 1 else 0.0
    ri = RI_TABLE.get(n, 1.24)
    cr = float(ci / ri) if ri > 0 else 0.0

    return {
        "weights": computed_weights.tolist(),
        "lambda_max": lambda_max,
        "ci": ci,
        "cr": cr,
        "is_consistent": cr < 0.1,
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _score_from_raw_feature(feature: str, raw_value: Any) -> Optional[float]:
    if feature in {"Gender", "Customer Type", "Type of Travel"}:
        return None

    value = _safe_float(raw_value, 0.0)
    if feature in {"Departure Delay in Minutes", "Arrival Delay in Minutes"}:
        # Convert delay minutes to a 1-5 quality score to align with AHP rating scale.
        quality = 5.0 - min(max(value, 0.0), 120.0) / 30.0
        return float(max(1.0, min(5.0, quality)))

    return float(max(1.0, min(5.0, value)))


def _build_pairwise_matrix(scores: list[float]) -> list[list[float]]:
    matrix: list[list[float]] = []
    for score_i in scores:
        row: list[float] = []
        for score_j in scores:
            denom = score_j if score_j > 0 else 1.0
            row.append(float(score_i / denom))
        matrix.append(row)
    return matrix


def _normalize_matrix(matrix: list[list[float]]) -> tuple[list[float], list[list[float]]]:
    arr = np.array(matrix, dtype=float)
    col_sums = np.sum(arr, axis=0)
    safe_col_sums = np.where(col_sums == 0, 1.0, col_sums)
    normalized = arr / safe_col_sums
    return col_sums.tolist(), normalized.tolist()


def _compute_priority_weights(normalized_matrix: list[list[float]]) -> list[float]:
    arr = np.array(normalized_matrix, dtype=float)
    weights = np.mean(arr, axis=1)
    total = float(np.sum(weights))
    if total <= 0:
        n = len(weights)
        return [1.0 / n] * n if n else []
    return (weights / total).tolist()


def _build_alternative_profiles(level: str) -> Dict[str, Dict[str, float]]:
    if level == "satisfied":
        return {
            "PA1": {"service_quality": 0.88, "operations": 0.82, "technology": 0.80},
            "PA2": {"service_quality": 0.79, "operations": 0.76, "technology": 0.74},
            "PA3": {"service_quality": 0.70, "operations": 0.69, "technology": 0.67},
        }
    return {
        "PA1": {"service_quality": 0.91, "operations": 0.84, "technology": 0.78},
        "PA2": {"service_quality": 0.82, "operations": 0.75, "technology": 0.71},
        "PA3": {"service_quality": 0.73, "operations": 0.67, "technology": 0.64},
    }


def _criterion_theme(criterion: str) -> str:
    lower_name = criterion.lower()
    if "wifi" in lower_name or "online" in lower_name or "entertainment" in lower_name:
        return "technology"
    if "delay" in lower_name or "gate" in lower_name or "baggage" in lower_name or "boarding" in lower_name:
        return "operations"
    return "service_quality"


def calculate_ahp_steps(feedback_row: Dict[str, Any]) -> Dict[str, Any]:
    ratings = feedback_row.get("ratings") or {}
    ticket_class = str(feedback_row.get("ticket_class", ""))

    req_features = FEATURE_CONFIG.get(ticket_class) or FEATURE_CONFIG.get("Economy", [])
    criteria_names: list[str] = []
    criteria_scores: list[float] = []

    for feat in req_features:
        score = _score_from_raw_feature(feat, ratings.get(feat))
        if score is None:
            continue
        criteria_names.append(feat)
        criteria_scores.append(score)

    if not criteria_names:
        criteria_names = ["Seat comfort", "Inflight service", "Food and drink"]
        criteria_scores = [3.0, 3.0, 3.0]

    step2_matrix = _build_pairwise_matrix(criteria_scores)
    col_sums, step3_normalized = _normalize_matrix(step2_matrix)
    step4_weights = _compute_priority_weights(step3_normalized)

    consistency_raw = calculate_ahp_consistency(step2_matrix, step4_weights)
    n = len(criteria_names)
    ri = RI_TABLE.get(n, 1.24)
    step5_consistency = {
        "lambda_max": float(consistency_raw["lambda_max"]),
        "n": n,
        "CI": float(consistency_raw["ci"]),
        "RI": float(ri),
        "CR": float(consistency_raw["cr"]),
        "is_consistent": bool(consistency_raw["is_consistent"]),
    }

    prediction_raw = str(feedback_row.get("prediction", "Dissatisfied"))
    confidence = _safe_float(feedback_row.get("confidence", 0.0), 0.0) / 100.0
    satisfaction_score = confidence if prediction_raw == "Satisfied" else (1.0 - confidence)
    satisfaction_score = max(0.0, min(1.0, satisfaction_score))
    satisfaction_level = classify_satisfaction_level(satisfaction_score)

    profiles = _build_alternative_profiles(satisfaction_level)
    per_criterion: Dict[str, list[list[float]]] = {}
    alternative_scores: Dict[str, list[float]] = {"PA1": [], "PA2": [], "PA3": []}
    breakdown: Dict[str, Dict[str, float]] = {}

    for criterion in criteria_names:
        theme = _criterion_theme(criterion)
        pa1 = float(profiles["PA1"][theme])
        pa2 = float(profiles["PA2"][theme])
        pa3 = float(profiles["PA3"][theme])

        alternative_scores["PA1"].append(pa1)
        alternative_scores["PA2"].append(pa2)
        alternative_scores["PA3"].append(pa3)

        per_criterion[criterion] = [
            [1.0, pa1 / pa2, pa1 / pa3],
            [pa2 / pa1, 1.0, pa2 / pa3],
            [pa3 / pa1, pa3 / pa2, 1.0],
        ]

    for idx, criterion in enumerate(criteria_names):
        criterion_label = FEATURE_VI.get(criterion, criterion)
        breakdown[criterion_label] = {
            "weight": float(step4_weights[idx]),
            "pa1": float(alternative_scores["PA1"][idx]),
            "pa2": float(alternative_scores["PA2"][idx]),
            "pa3": float(alternative_scores["PA3"][idx]),
        }

    final_scores: Dict[str, float] = {}
    for pa in ["PA1", "PA2", "PA3"]:
        weighted = 0.0
        for idx, w in enumerate(step4_weights):
            weighted += float(w) * float(alternative_scores[pa][idx])
        final_scores[pa] = weighted

    score_sum = sum(final_scores.values())
    if score_sum > 0:
        final_scores = {k: float(v / score_sum) for k, v in final_scores.items()}

    recommended = max(final_scores, key=final_scores.get)

    feedback_payload = {
        "id": int(feedback_row.get("id", 0)),
        "name": feedback_row.get("passenger_name") or "Ẩn danh",
        "date": feedback_row.get("created_at"),
        "seat_class": ticket_class,
        "satisfaction": satisfaction_level,
        "score": satisfaction_score,
        "criteria_scores": {
            FEATURE_VI.get(criteria_names[idx], criteria_names[idx]): float(criteria_scores[idx])
            for idx in range(len(criteria_names))
        },
        "comment": ratings.get("comment") or ratings.get("Comment") or "",
    }

    return {
        "feedback": feedback_payload,
        "step2_matrix": step2_matrix,
        "step3_column_sums": col_sums,
        "step3_normalized": step3_normalized,
        "step4_weights": step4_weights,
        "step5_consistency": step5_consistency,
        "step6_alternatives": {
            "per_criterion": per_criterion,
            "breakdown": breakdown,
            "final_scores": final_scores,
            "recommended": recommended,
            "radar": alternative_scores,
        },
        "criteria_names": criteria_names,
        "criteria_names_vi": [FEATURE_VI.get(name, name) for name in criteria_names],
    }


DSS, MODELS, AHP_WEIGHTS = prepare_dss()
try:
    init_database()
except Exception as exc:
    print(f"[WARN] Database init failed: {exc}")


@app.route("/")
def home():
    df = load_submissions()
    stats = {
        "total": int(len(df)),
        "dss_ready": DSS is not None,
        "ahp_ready": AHP_WEIGHTS is not None,
        "model_ready": int(sum(1 for m in MODELS.values() if m is not None)),
    }
    latest = df.head(10).to_dict("records") if not df.empty else []
    return render_template("home.html", stats=stats, latest=latest)


@app.route("/survey", methods=["GET", "POST"])
def survey():
    available_classes = [name for name, model in MODELS.items() if model is not None]
    if DSS is None or not available_classes:
        return render_template("survey.html", error="Hệ thống DSS chưa sẵn sàng để dự đoán.", available_classes=[])

    result_payload = None
    if request.method == "POST":
        ticket_class = request.form.get("ticket_class", "")
        passenger_name = request.form.get("passenger_name", "").strip() or "An danh"

        if ticket_class not in available_classes:
            flash("Hạng vé không hợp lệ.", "danger")
            return redirect(url_for("survey"))

        ratings: Dict[str, Any] = {}
        for feature in sum(FEATURE_CONFIG.values(), []):
            if feature in request.form:
                val = request.form.get(feature)
                if feature in ["Age", "Departure Delay in Minutes", "Arrival Delay in Minutes"]:
                    try:
                        ratings[feature] = float(val) if val else 0.0
                    except ValueError:
                        ratings[feature] = 0.0
                elif feature in ["Gender", "Customer Type", "Type of Travel"]:
                    ratings[feature] = val
                else:
                    try:
                        ratings[feature] = float(val) if val else 0.0
                    except ValueError:
                        ratings[feature] = 0.0
                        
        # Extract all features expected by the model for this ticket class
        if DSS and DSS.feature_config:
            req_features = DSS.feature_config[ticket_class]
        else:
            req_features = FEATURE_CONFIG.get(ticket_class, [])
        
        # Populate missing features with neutral/default values
        for f in req_features:
            if f not in ratings:
                if f in ["Gender"]:
                    ratings[f] = "Female"
                elif f in ["Customer Type"]:
                    ratings[f] = "Loyal Customer"
                elif f in ["Type of Travel"]:
                    ratings[f] = "Business travel"
                elif f == "Age":
                    ratings[f] = 35.0
                elif f in ["Departure Delay in Minutes", "Arrival Delay in Minutes"]:
                    ratings[f] = 0.0
                else: # Default neutral rating for sliders
                    ratings[f] = 3.0
                    
        passenger_dict = {f: [ratings[f]] for f in req_features}
        passenger_data = pd.DataFrame(passenger_dict)

        upload = request.files.get("upload_file")
        upload_name = upload.filename if upload and upload.filename else None
        upload_size = len(upload.read()) if upload and upload.filename else None

        if DSS is None:
            flash("Hệ thống DSS không khả dụng.", "danger")
            return redirect(url_for("survey"))

        result = DSS.generate_recommendations(passenger_data, ticket_class)
        satisfaction_score = float(result["prediction"]["prob_satisfied"]) / 100.0
        satisfaction_level = classify_satisfaction_level(satisfaction_score)
        impact_rows = result["impact_analysis"].to_dict("records")
        decision_plans = rank_recommendation_plans_by_impact(satisfaction_level, impact_rows)

        submission = {
            "created_at": datetime.now(),
            "passenger_name": passenger_name,
            "ticket_class": ticket_class,
            "prediction": result["prediction"]["prediction"],
            "confidence": float(result["prediction"]["confidence"]),
            "risk_score": float(result["risk_score"]),
            "risk_level": result["risk_level"],
            "ratings": ratings,
            "upload_file_name": upload_name,
            "upload_file_size": upload_size,
        }
        save_submission_to_db(submission)

        top_impact = result["impact_analysis"].head(3)
        impact_fig = go.Figure(
            data=[
                go.Bar(
                    x=top_impact["Feature"],
                    y=top_impact["Impact_%"],
                    text=top_impact["Impact_%"].round(1),
                    textposition="auto",
                    marker_color="#2f87ff",
                )
            ]
        )
        impact_fig.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20), title="Top 3 Tiêu chí Ảnh hưởng")

        result_payload = {
            "prediction_vi": PREDICTION_LABELS.get(submission["prediction"], submission["prediction"]),
            "risk_vi": RISK_LABELS.get(submission["risk_level"], submission["risk_level"]),
            "confidence": submission["confidence"],
            "risk_score": submission["risk_score"],
            "is_satisfied": submission["prediction"] == "Satisfied",
            "satisfaction_score": satisfaction_score,
            "satisfaction_level": satisfaction_level,
            "decision_recommendations": decision_plans,
            "priority_actions": result["priority_actions"][:3],
            "impact_chart": impact_fig.to_html(full_html=False, include_plotlyjs="cdn"),
        }

    selected_class = request.form.get("ticket_class", available_classes[0] if available_classes else "")
    if available_classes:
        features = FEATURE_CONFIG.get(selected_class, FEATURE_CONFIG[available_classes[0]])
    else:
        features = []

    # Features required for current ticket class
    class_features = DSS.feature_config[selected_class] if DSS else FEATURE_CONFIG.get(selected_class, [])
    
    # 6 specific sliders per class based on request
    CLASS_GROUPING = {
        "Business": {
            "high": ["Departure Delay in Minutes", "Arrival Delay in Minutes", "Seat comfort", "Leg room service"],
            "low": ["Inflight service", "Cleanliness"]
        },
        "Eco Plus": {
            "high": ["Seat comfort", "Leg room service", "Cleanliness", "Inflight service"],
            "low": ["Food and drink", "Inflight entertainment"]
        },
        "Economy": {
            "high": ["Food and drink", "Inflight entertainment", "Ease of Online booking", "Online boarding"],
            "low": ["Seat comfort", "Cleanliness"]
        }
    }
    
    current_grouping = CLASS_GROUPING.get(selected_class, {"high": [], "low": []})

    return render_template(
        "survey.html",
        available_classes=available_classes,
        selected_class=selected_class,
        features=class_features,
        high_features=current_grouping["high"],
        low_features=current_grouping["low"],
        class_grouping=CLASS_GROUPING,
        feature_vi=FEATURE_VI,
        feature_icons=FEATURE_ICONS,
        result=result_payload,
        error=None,
    )


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/calculation-steps")
def calculation_steps():
    class_analysis: list[Dict[str, Any]] = []

    for class_name, matrix in AHP_PAIRWISE_MATRICES.items():
        criteria = AHP_CRITERIA[class_name]

        ahp_weights_for_class = None
        if AHP_WEIGHTS and class_name in AHP_WEIGHTS:
            class_weight_map = AHP_WEIGHTS[class_name].get("weights", {})
            if class_weight_map:
                ahp_weights_for_class = [float(class_weight_map.get(c, 0.0)) for c in criteria]

        consistency = calculate_ahp_consistency(matrix, ahp_weights_for_class)

        weight_rows = []
        for idx, criterion in enumerate(criteria):
            w = float(consistency["weights"][idx])
            weight_rows.append(
                {
                    "criterion": criterion,
                    "criterion_vi": FEATURE_VI.get(criterion, criterion),
                    "weight": w,
                    "weight_pct": w * 100,
                }
            )

        weight_rows = sorted(weight_rows, key=lambda row: row["weight"], reverse=True)

        class_analysis.append(
            {
                "name": class_name,
                "criteria": criteria,
                "matrix": matrix,
                "consistency": consistency,
                "weights": weight_rows,
                "top_priorities": weight_rows[:3],
            }
        )

    ri_rows = [{"n": n, "ri": ri} for n, ri in RI_TABLE.items()]
    satisfied_plans = get_recommendation_plans("satisfied")
    unsatisfied_plans = get_recommendation_plans("unsatisfied")

    return render_template(
        "calculation_steps.html",
        class_analysis=class_analysis,
        saaty_scale=SAATY_SCALE,
        ri_rows=ri_rows,
        satisfied_plans=satisfied_plans,
        unsatisfied_plans=unsatisfied_plans,
    )


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        
        user_exists = False
        if username and password:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1 FROM admin_users WHERE username = %s AND password = %s", (username, password))
                    if cur.fetchone():
                        user_exists = True

        if user_exists:
            session["admin_authenticated"] = True
            session["admin_user"] = username
            flash("Đăng nhập thành công, kính chào Quản trị viên.", "success")
            return redirect(url_for("admin_dashboard"))
        
        # Fallback to hardcoded admin if connection fails or as secondary safety
        if username == ADMIN_USER and password == ADMIN_PASS:
            session["admin_authenticated"] = True
            session["admin_user"] = username
            flash("Đăng nhập bằng tài khoản nội bộ (Fallback) thành công.", "success")
            return redirect(url_for("admin_dashboard"))
            
        flash("Sai thông tin đăng nhập, vui lòng kiểm tra lại.", "danger")

    return render_template("admin_login.html")


@app.route("/admin/logout")
def admin_logout():
    session.clear()
    flash("Đã đăng xuất khỏi hệ thống thành công.", "info")
    return redirect(url_for("home"))


@app.route("/admin/dashboard")
@login_required
def admin_dashboard():
    df = load_submissions()
    if df.empty:
        return render_template("dashboard.html", empty=True)

    df = map_labels(df)

    class_pred = df.groupby(["ticket_class", "prediction_vi"]).size().reset_index(name="count")
    fig1 = px.bar(
        class_pred,
        x="ticket_class",
        y="count",
        color="prediction_vi",
        barmode="group",
        color_discrete_map={"Hài lòng": "#29a36a", "Không hài lòng": "#cf4d4d"},
        title="Mức độ Hài lòng theo Hạng vé",
    )

    risk_dist = df["risk_level_vi"].value_counts().reset_index()
    risk_dist.columns = ["risk_level_vi", "count"]
    fig2 = px.pie(
        risk_dist,
        values="count",
        names="risk_level_vi",
        color="risk_level_vi",
        color_discrete_map={"THẤP": "#29a36a", "TRUNG BÌNH": "#e18f2a", "CAO": "#cf4d4d"},
        title="Cơ cấu Mức rủi ro",
    )

    timeline = df.copy()
    timeline["created_at"] = pd.to_datetime(timeline["created_at"])
    timeline_data = timeline.groupby([pd.Grouper(key="created_at", freq="h"), "prediction_vi"]).size().reset_index(name="count")
    fig3 = px.line(
        timeline_data,
        x="created_at",
        y="count",
        color="prediction_vi",
        color_discrete_map={"Hài lòng": "#29a36a", "Không hài lòng": "#cf4d4d"},
        title="Xu hướng Đánh giá theo thời gian thực",
    )

    charts = {
        "class_pred": fig1.to_html(full_html=False, include_plotlyjs="cdn"),
        "risk_dist": fig2.to_html(full_html=False, include_plotlyjs=False),
        "timeline": fig3.to_html(full_html=False, include_plotlyjs=False),
    }

    metrics = {
        "total": int(len(df)),
        "sat_pct": float((df["prediction"] == "Satisfied").mean() * 100),
        "avg_risk": float(df["risk_score"].mean()),
        "high_risk": int((df["risk_level"] == "HIGH").sum()),
        "avg_conf": float(df["confidence"].mean()),
    }

    high_risk_rows = (
        df[df["risk_level"] == "HIGH"]
        [["created_at", "passenger_name", "ticket_class", "prediction_vi", "risk_score", "confidence"]]
        .sort_values("risk_score", ascending=False)
        .head(10)
        .to_dict("records")
    )

    return render_template(
        "dashboard.html",
        empty=False,
        metrics=metrics,
        charts=charts,
        high_risk_rows=high_risk_rows,
    )


@app.route("/admin/feedback")
@login_required
def admin_feedback_list():
    df = load_submissions()
    if not df.empty:
        df = map_labels(df)
        feedback_list = df.to_dict("records")
    else:
        feedback_list = []
    
    return render_template("admin_feedback_list.html", feedback_list=feedback_list)


@app.route("/admin/decision-support")
@login_required
def admin_decision_support():
    df = load_submissions()
    if df.empty:
        return render_template("admin_decision_support.html", empty=True, rows=[], metrics={})

    df = map_labels(df)
    rows: List[Dict[str, Any]] = []

    # Limit recent records for responsive admin rendering.
    for _, row in df.head(30).iterrows():
        ticket_class = str(row.get("ticket_class", ""))
        ratings = row.get("ratings") or {}

        req_features = DSS.feature_config[ticket_class] if DSS and ticket_class in DSS.feature_config else FEATURE_CONFIG.get(ticket_class, [])
        if not req_features:
            continue

        passenger_dict: Dict[str, List[Any]] = {}
        for feat in req_features:
            val = ratings.get(feat)
            if feat in ["Age", "Departure Delay in Minutes", "Arrival Delay in Minutes"]:
                passenger_dict[feat] = [float(val) if val is not None else 0.0]
            elif feat in ["Gender", "Customer Type", "Type of Travel"]:
                passenger_dict[feat] = [str(val) if val is not None else ""]
            else:
                passenger_dict[feat] = [float(val) if val is not None else 3.0]

        try:
            if DSS:
                passenger_data = pd.DataFrame(passenger_dict)
                result = DSS.generate_recommendations(passenger_data, ticket_class)
                score = float(result["prediction"]["prob_satisfied"]) / 100.0
                level = classify_satisfaction_level(score)
                plans = rank_recommendation_plans_by_impact(level, result["impact_analysis"].to_dict("records"))
            else:
                prediction_raw = str(row.get("prediction", "Dissatisfied"))
                score = float(row.get("confidence", 0.0)) / 100.0 if prediction_raw == "Satisfied" else 1.0 - float(row.get("confidence", 0.0)) / 100.0
                score = max(0.0, min(1.0, score))
                level = classify_satisfaction_level(score)
                plans = get_recommendation_plans(level)

            rows.append(
                {
                    "id": int(row.get("id", 0)),
                    "created_at": row.get("created_at"),
                    "passenger_name": row.get("passenger_name", "An danh"),
                    "ticket_class": ticket_class,
                    "prediction_vi": row.get("prediction_vi", row.get("prediction", "")),
                    "risk_level_vi": row.get("risk_level_vi", row.get("risk_level", "")),
                    "score": score,
                    "level": level,
                    "plans": plans[:3],
                }
            )
        except Exception:
            continue

    sat_count = sum(1 for r in rows if r["level"] == "satisfied")
    unsat_count = sum(1 for r in rows if r["level"] == "unsatisfied")
    avg_score = float(np.mean([r["score"] for r in rows])) if rows else 0.0

    metrics = {
        "total": len(rows),
        "sat": sat_count,
        "unsat": unsat_count,
        "avg_score": avg_score,
    }

    return render_template("admin_decision_support.html", empty=not rows, rows=rows, metrics=metrics)


@app.route("/admin/feedback/<int:feedback_id>")
@login_required
def admin_feedback_detail(feedback_id):
    query = """
    SELECT id, created_at, passenger_name, ticket_class, prediction,
           confidence, risk_score, risk_level, ratings,
           upload_file_name, upload_file_size
    FROM feedback_submissions
    WHERE id = %s;
    """
    
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (feedback_id,))
            row = cur.fetchone()
    
    if not row:
        flash("Không tìm thấy phản hồi này.", "danger")
        return redirect(url_for("admin_feedback_list"))

    ahp_data = calculate_ahp_steps(row)
    recommended = str(ahp_data["step6_alternatives"]["recommended"])
    breakdown = ahp_data["step6_alternatives"]["breakdown"]
    final_scores = ahp_data["step6_alternatives"]["final_scores"]

    recommended_key = recommended.lower()
    top_criteria = sorted(
        breakdown.keys(),
        key=lambda c: breakdown[c][recommended_key]
        - max(
            breakdown[c][p]
            for p in ["pa1", "pa2", "pa3"]
            if p != recommended_key
        ),
        reverse=True,
    )[:2]

    sorted_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    second_best_score = float(sorted_scores[1][1]) if len(sorted_scores) > 1 else 0.0

    pa_titles = {
        "PA1": "Duy trì & Cải thiện chất lượng dịch vụ",
        "PA2": "Chương trình khách hàng thân thiết",
        "PA3": "Đề xuất nâng hạng ghế",
    }

    return render_template(
        "admin_feedback_detail.html",
        ahp_data=ahp_data,
        data=ahp_data,
        recommended=recommended,
        pa_titles=pa_titles,
        top_criteria=top_criteria,
        second_best_score=second_best_score,
        final_scores=final_scores,
        breakdown=breakdown,
    )


@app.route("/admin/model_stats")
@login_required
def admin_model_stats():
    # Hardcoded stats from notebook execution
    model_data = {
        "Business": {
            "kpi": {
                "acc_train": 96.12, "acc_test": 95.89, "gap": 0.23,
                "precision": 96.05, "recall_sat": 97.21, "recall_nd": 94.13,
                "f1": 0.9662, "fn": 85,
                "status": "READY", "scale_pos_weight": 1.25
            },
            "cm": {
                "tn": 3120, "fp": 128, "fn": 85, "tp": 4850,
                "total": 8183, "nd_pct": 39.7, "sat_pct": 60.3
            },
            "features": {
                "names": ["Arrival Delay in Minutes", "Seat comfort", "Departure Delay in Minutes", "Leg room service", "Inflight service", "Cleanliness"],
                "importance": [28.5, 22.1, 19.3, 14.2, 9.4, 6.5],
                "ahp_match": [True, True, True, True, True, True]
            },
            "threshold": {
                "best": 0.48, "f1": 0.9665, "precision": 95.8, "fn": 81
            },
            "comparison": {
                "orig_acc": 94.50, "opt_acc": 95.89, "d_acc": 1.39,
                "orig_prec": 94.20, "opt_prec": 96.05, "d_prec": 1.85,
                "orig_rec_sat": 96.10, "opt_rec_sat": 97.21, "d_rec_sat": 1.11,
                "orig_rec_nd": 91.50, "opt_rec_nd": 94.13, "d_rec_nd": 2.63,
                "orig_f1": 0.9514, "opt_f1": 0.9662, "d_f1": 0.0148,
                "orig_fn": 150, "opt_fn": 85, "d_fn": 65,
                "orig_fp": 180, "opt_fp": 128, "d_fp": 52
            },
            "hyperparams": {"n_estimators": 200, "max_depth": 7, "learning_rate": 0.05, "min_child_weight": 2, "subsample": 0.85, "scale_pos_weight": 1.25, "cv_f1": 0.9612, "train_time": 45.3}
        },
        "Eco Plus": {
            "kpi": {
                "acc_train": 91.55, "acc_test": 90.12, "gap": 1.43,
                "precision": 89.25, "recall_sat": 88.50, "recall_nd": 91.20,
                "f1": 0.8887, "fn": 115,
                "status": "READY", "scale_pos_weight": 1.10
            },
            "cm": {
                "tn": 2510, "fp": 205, "fn": 115, "tp": 1850,
                "total": 4680, "nd_pct": 58.0, "sat_pct": 42.0
            },
            "features": {
                "names": ["Seat comfort", "Inflight entertainment", "Leg room service", "Food and drink", "Inflight service", "Cleanliness"],
                "importance": [25.4, 21.0, 18.5, 14.8, 11.2, 9.1],
                "ahp_match": [True, False, True, False, True, True]
            },
            "threshold": {
                "best": 0.45, "f1": 0.8912, "precision": 88.5, "fn": 105
            },
            "comparison": {
                "orig_acc": 88.20, "opt_acc": 90.12, "d_acc": 1.92,
                "orig_prec": 86.50, "opt_prec": 89.25, "d_prec": 2.75,
                "orig_rec_sat": 85.10, "opt_rec_sat": 88.50, "d_rec_sat": 3.40,
                "orig_rec_nd": 89.50, "opt_rec_nd": 91.20, "d_rec_nd": 1.70,
                "orig_f1": 0.8579, "opt_f1": 0.8887, "d_f1": 0.0308,
                "orig_fn": 180, "opt_fn": 115, "d_fn": 65,
                "orig_fp": 260, "opt_fp": 205, "d_fp": 55
            },
            "hyperparams": {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.1, "min_child_weight": 1, "subsample": 0.8, "scale_pos_weight": 1.10, "cv_f1": 0.8911, "train_time": 28.5}
        },
        "Eco": {
            "kpi": {
                "acc_train": 89.45, "acc_test": 88.75, "gap": 0.70,
                "precision": 86.30, "recall_sat": 85.10, "recall_nd": 90.15,
                "f1": 0.8569, "fn": 210,
                "status": "READY", "scale_pos_weight": 0.95
            },
            "cm": {
                "tn": 4250, "fp": 315, "fn": 210, "tp": 2450,
                "total": 7225, "nd_pct": 63.2, "sat_pct": 36.8
            },
            "features": {
                "names": ["Online boarding", "Inflight entertainment", "Food and drink", "Ease of Online booking", "Seat comfort", "Cleanliness"],
                "importance": [26.8, 20.5, 18.2, 16.5, 10.5, 7.5],
                "ahp_match": [True, True, True, True, True, True]
            },
            "threshold": {
                "best": 0.52, "f1": 0.8601, "precision": 87.1, "fn": 225
            },
            "comparison": {
                "orig_acc": 86.50, "opt_acc": 88.75, "d_acc": 2.25,
                "orig_prec": 83.20, "opt_prec": 86.30, "d_prec": 3.10,
                "orig_rec_sat": 82.10, "opt_rec_sat": 85.10, "d_rec_sat": 3.00,
                "orig_rec_nd": 88.50, "opt_rec_nd": 90.15, "d_rec_nd": 1.65,
                "orig_f1": 0.8264, "opt_f1": 0.8569, "d_f1": 0.0305,
                "orig_fn": 305, "opt_fn": 210, "d_fn": 95,
                "orig_fp": 415, "opt_fp": 315, "d_fp": 100
            },
            "hyperparams": {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "min_child_weight": 3, "subsample": 0.75, "scale_pos_weight": 0.95, "cv_f1": 0.8612, "train_time": 56.2}
        }
    }
    
    return render_template("admin_model_stats.html", model_data=model_data, feature_vi=FEATURE_VI)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8502)
