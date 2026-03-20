"""
GIAI DOAN 6: PHAT TRIEN SAN PHAM WEB DSS
He thong ho tro quyet dinh danh gia hai long hanh khach - ung dung Streamlit
"""

from __future__ import annotations

import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from recommendation_engine import (
    classify_satisfaction_level,
    get_recommendation_plans,
    rank_recommendation_plans_by_impact,
)

try:
    import psycopg2
    from psycopg2.extras import Json, RealDictCursor
except Exception:
    psycopg2 = None
    Json = None
    RealDictCursor = None


PREDICTION_LABELS = {
    "Satisfied": "Hai long",
    "Dissatisfied": "Khong hai long",
}

RISK_LABELS = {
    "LOW": "THAP",
    "MEDIUM": "TRUNG BINH",
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
    "Eco Plus": [
        "Seat comfort",
        "Leg room service",
        "Cleanliness",
        "Inflight service",
        "Food and drink",
        "Inflight entertainment",
    ],
}

DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "airline"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "123"),
}

ADMIN_USER = os.getenv("AIRLINE_ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("AIRLINE_ADMIN_PASS", "admin123")


class AirlineDSS:
    """He thong DSS ket hop XGBoost va AHP."""

    def __init__(self, models_dict: Dict[str, Any], ahp_weights_dict: Dict[str, Any], feature_config: Dict[str, List[str]]):
        self.models = models_dict
        self.ahp_weights = ahp_weights_dict
        self.feature_config = feature_config

    def predict_satisfaction(self, passenger_data: pd.DataFrame, ticket_class: str) -> Dict[str, float | str]:
        if ticket_class not in self.models:
            raise ValueError(f"Hang ve khong hop le: {ticket_class}")

        model = self.models[ticket_class]
        if model is None:
            raise ValueError(f"Chua tai mo hinh cho hang {ticket_class}")

        features = self.feature_config[ticket_class]
        X = passenger_data[features]

        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]

        return {
            "prediction": "Satisfied" if prediction == 1 else "Dissatisfied",
            "confidence": max(probabilities) * 100,
            "prob_dissatisfied": probabilities[0] * 100,
            "prob_satisfied": probabilities[1] * 100,
        }

    def calculate_feature_impact(self, passenger_data: pd.DataFrame, ticket_class: str) -> pd.DataFrame:
        model = self.models[ticket_class]
        features = self.feature_config[ticket_class]

        xgb_importance = model.feature_importances_

        ahp_dict = self.ahp_weights[ticket_class]["weights"]
        ahp_values = np.array([ahp_dict[f] for f in features])

        feature_values = passenger_data[features].values[0]

        combined_weights = (xgb_importance * 0.5 + ahp_values * 0.5)
        impact_scores = combined_weights * feature_values
        impact_percentage = (impact_scores / impact_scores.sum()) * 100

        impact_df = pd.DataFrame(
            {
                "Feature": features,
                "Current_Value": feature_values,
                "XGBoost_Importance": xgb_importance,
                "AHP_Weight": ahp_values,
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

        strong_features = impact_df[
            (impact_df["Current_Value"] >= 4)
            & (impact_df["Combined_Weight"] > impact_df["Combined_Weight"].median())
        ]

        priority_actions = []
        for _, row in weak_features.iterrows():
            priority_actions.append(
                {
                    "feature": row["Feature"],
                    "current_rating": row["Current_Value"],
                    "importance": row["Combined_Weight"],
                    "impact": row["Impact_%"],
                    "urgency": "HIGH" if row["Current_Value"] <= 1 else "MEDIUM",
                    "action": f"Cai thien '{row['Feature']}' (hien tai {row['Current_Value']}/5)",
                }
            )

        risk_score = prediction_result["prob_dissatisfied"] * 0.7 + (len(weak_features) / len(impact_df) * 100) * 0.3

        return {
            "ticket_class": ticket_class,
            "prediction": prediction_result,
            "impact_analysis": impact_df,
            "weak_features": weak_features,
            "strong_features": strong_features,
            "priority_actions": priority_actions,
            "risk_score": risk_score,
            "risk_level": "HIGH" if risk_score >= 70 else "MEDIUM" if risk_score >= 40 else "LOW",
        }


def _file_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_resource
def load_ahp_weights(_mtime: float) -> Optional[Dict[str, Any]]:
    base_dir = Path(__file__).resolve().parent
    ahp_path = base_dir / "ahp_weights_all_classes.pkl"
    if not ahp_path.exists():
        return None
    with ahp_path.open("rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_models(_mtimes: tuple[float, float, float]) -> Dict[str, Any]:
    base_dir = Path(__file__).resolve().parent
    model_paths = {
        "Business": base_dir / "business_xgboost_optimized.pkl",
        "Economy": base_dir / "economy_xgboost_optimized.pkl",
        "Eco Plus": base_dir / "ecoplus_xgboost_optimized.pkl",
    }

    models: Dict[str, Any] = {}
    for class_name, path in model_paths.items():
        if path.exists():
            with path.open("rb") as f:
                models[class_name] = pickle.load(f)
        else:
            models[class_name] = None
    return models


@st.cache_resource
def load_dss_system(_mtime: float) -> Optional[AirlineDSS]:
    base_dir = Path(__file__).resolve().parent
    dss_path = base_dir / "airline_dss_system.pkl"
    if not dss_path.exists():
        return None

    class DSSUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str):
            if module == "__main__" and name == "AirlineDSS":
                return AirlineDSS
            return super().find_class(module, name)

    with dss_path.open("rb") as f:
        return DSSUnpickler(f).load()


def db_enabled() -> bool:
    return psycopg2 is not None and Json is not None and RealDictCursor is not None


def get_db_connection():
    if not db_enabled():
        raise RuntimeError("Thieu psycopg2. Vui long cai dat psycopg2-binary.")
    return psycopg2.connect(**DB_CONFIG)


def init_database() -> bool:
    if not db_enabled():
        return False
    query = """
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
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
            conn.commit()
        return True
    except Exception as exc:
        st.error(f"Khong tao duoc bang du lieu PostgreSQL: {exc}")
        return False


def save_submission_to_db(submission: Dict[str, Any]) -> bool:
    if not db_enabled():
        return False

    query = """
    INSERT INTO feedback_submissions (
        created_at, passenger_name, ticket_class, prediction,
        confidence, risk_score, risk_level, ratings,
        upload_file_name, upload_file_size
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
    """

    try:
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
        return True
    except Exception as exc:
        st.error(f"Khong luu duoc danh gia vao PostgreSQL: {exc}")
        return False


def load_submissions_from_db() -> pd.DataFrame:
    if not db_enabled():
        return pd.DataFrame()

    query = """
    SELECT id, created_at, passenger_name, ticket_class, prediction,
           confidence, risk_score, risk_level, ratings,
           upload_file_name, upload_file_size
    FROM feedback_submissions
    ORDER BY created_at DESC;
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                rows = cur.fetchall()
        return pd.DataFrame(rows)
    except Exception as exc:
        st.error(f"Khong doc du lieu danh gia tu PostgreSQL: {exc}")
        return pd.DataFrame()


def svg_icon(icon_name: str) -> str:
    icons = {
        "flight": """
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M2 14.5L10.5 12L18 2L22 4L15 13L16.5 20L13 22L10.5 14.5L4 18L2 14.5Z" fill="#73B3FF"/>
        </svg>
        """,
        "shield": """
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2L20 5V11C20 16.5 16.5 20.74 12 22C7.5 20.74 4 16.5 4 11V5L12 2Z" fill="#8AD3C8"/>
        </svg>
        """,
        "chart": """
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M4 19H20" stroke="#9DB6FF" stroke-width="2"/>
            <rect x="6" y="10" width="3" height="7" fill="#9DB6FF"/>
            <rect x="11" y="7" width="3" height="10" fill="#73B3FF"/>
            <rect x="16" y="4" width="3" height="13" fill="#4D92E5"/>
        </svg>
        """,
    }
    return icons.get(icon_name, "")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&family=Lexend:wght@400;600;700&display=swap');

            :root {
                --bg-main: #e9f1fb;
                --ink: #13325f;
                --ink-soft: #2b4a73;
                --hero-start: #16355f;
                --hero-end: #80b7d8;
                --card-bg: #f6f8fc;
                --line: #d7deea;
                --accent: #2f87ff;
                --accent-deep: #165aa5;
                --ok: #22a36e;
                --warn: #e18f2a;
                --bad: #d44747;
            }

            .stApp {
                font-family: 'Manrope', sans-serif;
                background: radial-gradient(circle at 20% 0%, #f4f9ff 0%, var(--bg-main) 45%, #dde7f4 100%);
            }

            .hero {
                border-radius: 24px;
                padding: 42px 30px;
                background: linear-gradient(115deg, var(--hero-start) 0%, #335a85 52%, var(--hero-end) 100%);
                color: #f8fbff;
                box-shadow: 0 16px 40px rgba(24, 49, 86, 0.24);
                margin: 6px 0 26px 0;
            }

            .hero h1 {
                font-family: 'Lexend', sans-serif;
                font-size: 2.35rem;
                margin: 0;
                letter-spacing: 0.2px;
            }

            .hero p {
                margin-top: 14px;
                max-width: 760px;
                font-size: 1.04rem;
                line-height: 1.5;
                color: #d8e8fb;
            }

            .panel {
                background: var(--card-bg);
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 18px 20px;
                box-shadow: 0 8px 22px rgba(26, 45, 78, 0.08);
                margin-bottom: 16px;
            }

            .panel h3 {
                margin: 0 0 8px 0;
                color: var(--ink);
                font-size: 1.08rem;
            }

            .panel p {
                margin: 0;
                color: var(--ink-soft);
                line-height: 1.5;
            }

            .section-title {
                font-family: 'Lexend', sans-serif;
                font-weight: 700;
                color: var(--ink);
                margin: 8px 0 16px 0;
                letter-spacing: 0.2px;
            }

            .metric-strip {
                border-radius: 14px;
                background: #ffffff;
                border: 1px solid #d8e4f0;
                padding: 12px 14px;
            }

            .stButton > button, .stDownloadButton > button {
                border-radius: 12px;
                border: 1px solid #13407a !important;
                background: linear-gradient(90deg, #113c73 0%, #23679f 100%);
                color: #f5f9ff;
                font-weight: 700;
                letter-spacing: 0.2px;
            }

            .stTextInput > div > div > input,
            .stSelectbox div[data-baseweb="select"] > div,
            .stNumberInput input,
            .stTextArea textarea {
                border-radius: 10px !important;
            }

            .result-good {
                border-left: 6px solid var(--ok);
                background: #ecfbf4;
                border-radius: 12px;
                padding: 14px;
                margin-top: 10px;
                color: #0d5b3a;
            }

            .result-bad {
                border-left: 6px solid var(--bad);
                background: #fff1f1;
                border-radius: 12px;
                padding: 14px;
                margin-top: 10px;
                color: #822020;
            }

            .sidebar-status {
                font-size: 0.9rem;
                color: #315070;
                background: #edf3fc;
                border: 1px solid #d4e0ef;
                border-radius: 10px;
                padding: 10px;
                margin-top: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    if "current_result" not in st.session_state:
        st.session_state.current_result = None
    if "page_choice" not in st.session_state:
        st.session_state.page_choice = "Trang chu"
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False
    if "admin_user" not in st.session_state:
        st.session_state.admin_user = ""


def render_admin_login_sidebar() -> None:
    st.sidebar.markdown("### Quan tri")
    if not st.session_state.admin_authenticated:
        with st.sidebar.form("admin_login_form"):
            username = st.text_input("Tai khoan admin")
            password = st.text_input("Mat khau admin", type="password")
            login_submit = st.form_submit_button("Dang nhap")

        if login_submit:
            if username == ADMIN_USER and password == ADMIN_PASS:
                st.session_state.admin_authenticated = True
                st.session_state.admin_user = username
                st.sidebar.success("Dang nhap thanh cong")
            else:
                st.sidebar.error("Sai thong tin dang nhap")
    else:
        st.sidebar.success(f"Da dang nhap: {st.session_state.admin_user}")
        if st.sidebar.button("Dang xuat", use_container_width=True):
            st.session_state.admin_authenticated = False
            st.session_state.admin_user = ""
            st.rerun()

    st.sidebar.markdown(
        f"""
        <div class="sidebar-status">
            PostgreSQL: {DB_CONFIG['host']}:{DB_CONFIG['port']}<br/>
            Database: {DB_CONFIG['dbname']}<br/>
            User: {DB_CONFIG['user']}
        </div>
        """,
        unsafe_allow_html=True,
    )


def prepare_dss() -> tuple[Optional[AirlineDSS], Dict[str, Any], Optional[Dict[str, Any]]]:
    base_dir = Path(__file__).resolve().parent
    dss_mtime = _file_mtime(base_dir / "airline_dss_system.pkl")
    ahp_mtime = _file_mtime(base_dir / "ahp_weights_all_classes.pkl")
    model_mtimes = (
        _file_mtime(base_dir / "business_xgboost_optimized.pkl"),
        _file_mtime(base_dir / "economy_xgboost_optimized.pkl"),
        _file_mtime(base_dir / "ecoplus_xgboost_optimized.pkl"),
    )

    ahp_weights = load_ahp_weights(ahp_mtime)
    models = load_models(model_mtimes)
    dss_pickle = load_dss_system(dss_mtime)

    if ahp_weights is not None and any(models.values()):
        dss = AirlineDSS(models_dict=models, ahp_weights_dict=ahp_weights, feature_config=FEATURE_CONFIG)
    elif dss_pickle is not None:
        dss = dss_pickle
    else:
        dss = None

    return dss, models, ahp_weights


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if "prediction" in data.columns:
        data["prediction_vi"] = data["prediction"].replace(PREDICTION_LABELS)
    if "risk_level" in data.columns:
        data["risk_level_vi"] = data["risk_level"].replace(RISK_LABELS)
    return data


def render_home_page(db_df: pd.DataFrame, dss: Optional[AirlineDSS], ahp_weights: Optional[Dict[str, Any]], models: Dict[str, Any]) -> None:
    render_hero(
        "Danh gia trai nghiem chuyen bay",
        "Phieu khao sat duoc tong hop tu 6 tieu chi cot loi va su dung AI de xac dinh muc do hai long cung muc rui ro phuc vu.",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="panel">
                <h3>{svg_icon('flight')} Khao sat hanh khach</h3>
                <p>Khach hang cham diem 6 tieu chi theo tung hang ve va nhan ket qua du doan ngay lap tuc.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="panel">
                <h3>{svg_icon('chart')} He thong DSS</h3>
                <p>Mang mo hinh XGBoost ket hop AHP de giai thich ly do hai long va uu tien cai thien.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="panel">
                <h3>{svg_icon('shield')} Quan tri chat luong</h3>
                <p>Du lieu duoc luu vao PostgreSQL de theo doi KPI, xu huong va canh bao rui ro theo thoi gian.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Trang thai he thong")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("DSS", "San sang" if dss is not None else "Chua san sang")
    with s2:
        st.metric("AHP", "San sang" if ahp_weights else "Chua san sang")
    with s3:
        model_ready = sum(1 for m in models.values() if m is not None)
        st.metric("Mo hinh", f"{model_ready}/3")
    with s4:
        st.metric("So danh gia", len(db_df))

    if len(db_df) > 0:
        st.markdown("### Danh gia moi nhat")
        preview = db_df[["created_at", "passenger_name", "ticket_class", "prediction", "risk_level", "risk_score"]].head(10)
        st.dataframe(preview, use_container_width=True)


def render_customer_form_page(dss: Optional[AirlineDSS], models: Dict[str, Any]) -> None:
    render_hero(
        "Bieu mau khao sat",
        "Vui long hoan tat danh gia de he thong du doan muc do hai long cua ban voi trai nghiem hang khong.",
    )

    available_classes = [k for k, v in models.items() if v is not None]
    if dss is None or len(available_classes) == 0:
        st.error("He thong DSS chua san sang. Vui long kiem tra model da duoc huan luyen.")
        return

    st.markdown('<div class="section-title">Thong tin hanh khach</div>', unsafe_allow_html=True)

    with st.form("customer_form", clear_on_submit=False):
        a, b = st.columns(2)
        with a:
            ticket_class = st.selectbox("Hang ve", options=available_classes)
        with b:
            passenger_name = st.text_input("Ten hanh khach", placeholder="Co the de trong")

        st.markdown('<div class="section-title">Danh gia 6 tieu chi dich vu</div>', unsafe_allow_html=True)

        features = FEATURE_CONFIG[ticket_class]
        ratings: Dict[str, int] = {}
        cols = st.columns(2)

        for idx, feature in enumerate(features):
            with cols[idx % 2]:
                ratings[feature] = st.slider(
                    feature,
                    min_value=1,
                    max_value=5,
                    value=3,
                    help="1 la rat te, 5 la rat tot",
                )

        uploaded_file = st.file_uploader(
            "Tep dinh kem (neu co)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=False,
            help="File tuy chon de bo sung minh chung trai nghiem",
        )

        submit = st.form_submit_button("Gui danh gia va du doan", use_container_width=True)

    if not submit:
        return

    try:
        passenger_data = pd.DataFrame([ratings])
        result = dss.generate_recommendations(passenger_data, ticket_class)

        satisfaction_score = float(result["prediction"]["prob_satisfied"]) / 100.0
        satisfaction_level = classify_satisfaction_level(satisfaction_score)
        impact_rows = result["impact_analysis"].to_dict("records") if "impact_analysis" in result else []
        decision_plans = rank_recommendation_plans_by_impact(satisfaction_level, impact_rows)
        result["satisfaction_score"] = satisfaction_score
        result["satisfaction_level"] = satisfaction_level
        result["decision_recommendations"] = decision_plans

        pred_raw = str(result["prediction"]["prediction"])
        risk_raw = str(result["risk_level"])

        submission = {
            "created_at": datetime.now(),
            "passenger_name": passenger_name.strip() if passenger_name.strip() else "An danh",
            "ticket_class": ticket_class,
            "prediction": pred_raw,
            "confidence": float(result["prediction"]["confidence"]),
            "risk_score": float(result["risk_score"]),
            "risk_level": risk_raw,
            "ratings": ratings,
            "upload_file_name": uploaded_file.name if uploaded_file else None,
            "upload_file_size": uploaded_file.size if uploaded_file else None,
        }

        saved = save_submission_to_db(submission)
        st.session_state.current_result = result

        pred_vi = PREDICTION_LABELS.get(pred_raw, pred_raw)
        risk_vi = RISK_LABELS.get(risk_raw, risk_raw)

        if pred_raw == "Satisfied":
            st.markdown(
                f"""
                <div class="result-good">
                    <strong>Ket qua du doan:</strong> {pred_vi}<br/>
                    <strong>Do tin cay:</strong> {submission['confidence']:.1f}%<br/>
                    <strong>Muc rui ro:</strong> {risk_vi}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-bad">
                    <strong>Ket qua du doan:</strong> {pred_vi}<br/>
                    <strong>Do tin cay:</strong> {submission['confidence']:.1f}%<br/>
                    <strong>Muc rui ro:</strong> {risk_vi}
                </div>
                """,
                unsafe_allow_html=True,
            )

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Do tin cay", f"{submission['confidence']:.1f}%")
        with m2:
            st.metric("Rui ro", f"{submission['risk_score']:.1f}/100")
        with m3:
            st.metric("Hanh dong uu tien", len(result["priority_actions"]))

        st.markdown("### Phuong an xu ly phan hoi khach hang")
        st.caption("Logic DSS: score >= 0.7 => satisfied, score < 0.7 => unsatisfied")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Customer Satisfaction Level", satisfaction_level)
        with c2:
            st.metric("Satisfaction Score", f"{satisfaction_score:.2f}")

        for idx, plan in enumerate(decision_plans, start=1):
            st.markdown(
                f"""
                <div class="panel" style="margin-bottom: 10px; border-left: 6px solid {'#22a36e' if satisfaction_level == 'satisfied' else '#d44747'};">
                    <h3 style="margin-bottom: 6px;">PA{idx}. {plan['title']}</h3>
                    <p style="margin-bottom: 6px;">{plan['description']}</p>
                    <p><strong>Ly do de xuat:</strong> {plan.get('reason', 'De xuat theo muc do hai long tong hop.')}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        top_impact = result["impact_analysis"].head(3)
        fig = go.Figure(
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
        fig.update_layout(
            title="Top 3 tieu chi anh huong",
            xaxis_title="Tieu chi",
            yaxis_title="Muc tac dong (%)",
            height=390,
        )
        st.plotly_chart(fig, use_container_width=True)

        if len(result["priority_actions"]) > 0:
            st.markdown("### Khu vuc can cai thien")
            for idx, action in enumerate(result["priority_actions"][:3], start=1):
                st.write(
                    f"{idx}. {action['action']} | Diem: {action['current_rating']}/5 | Tac dong: {action['impact']:.1f}%"
                )

        if saved:
            st.success("Danh gia da duoc luu vao PostgreSQL thanh cong.")
        else:
            st.warning("Danh gia da duoc xu ly nhung chua luu vao PostgreSQL. Kiem tra ket noi DB.")

    except Exception as exc:
        st.error(f"Co loi khi xu ly danh gia: {exc}")


def render_admin_dashboard_page(db_df: pd.DataFrame) -> None:
    render_hero(
        "Dashboard quan tri trai nghiem bay",
        "Trang nay chi danh cho admin: theo doi KPI hai long, rui ro va xu huong danh gia theo thoi gian.",
    )

    if not st.session_state.admin_authenticated:
        st.warning("Can dang nhap admin o thanh ben trai de vao dashboard quan tri.")
        return

    if db_df.empty:
        st.info("Chua co du lieu danh gia trong PostgreSQL.")
        return

    df = map_labels(db_df)

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Tong danh gia", len(df))
    with k2:
        sat_pct = (df["prediction"] == "Satisfied").mean() * 100
        st.metric("Ty le hai long", f"{sat_pct:.1f}%")
    with k3:
        st.metric("Rui ro trung binh", f"{df['risk_score'].mean():.1f}")
    with k4:
        high_count = (df["risk_level"] == "HIGH").sum()
        st.metric("Rui ro cao", int(high_count))
    with k5:
        st.metric("Do tin cay TB", f"{df['confidence'].mean():.1f}%")

    tab1, tab2, tab3, tab4 = st.tabs(["Tong quan", "Phan tich theo hang", "Canh bao", "Du lieu thuan"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            class_pred = (
                df.groupby(["ticket_class", "prediction_vi"]).size().reset_index(name="count")
            )
            fig1 = px.bar(
                class_pred,
                x="ticket_class",
                y="count",
                color="prediction_vi",
                barmode="group",
                title="Muc hai long theo hang ve",
                color_discrete_map={"Hai long": "#29a36a", "Khong hai long": "#cf4d4d"},
            )
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            risk_dist = df["risk_level_vi"].value_counts().reset_index()
            risk_dist.columns = ["risk_level_vi", "count"]
            fig2 = px.pie(
                risk_dist,
                values="count",
                names="risk_level_vi",
                title="Co cau muc rui ro",
                color="risk_level_vi",
                color_discrete_map={"THAP": "#29a36a", "TRUNG BINH": "#e18f2a", "CAO": "#cf4d4d"},
            )
            st.plotly_chart(fig2, use_container_width=True)

        timeline = df.copy()
        timeline["created_at"] = pd.to_datetime(timeline["created_at"])
        timeline_data = (
            timeline.groupby([pd.Grouper(key="created_at", freq="h"), "prediction_vi"]).size().reset_index(name="count")
        )
        fig3 = px.line(
            timeline_data,
            x="created_at",
            y="count",
            color="prediction_vi",
            title="Xu huong danh gia theo gio",
            color_discrete_map={"Hai long": "#29a36a", "Khong hai long": "#cf4d4d"},
        )
        st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        st.markdown("### Diem trung binh cua cac tieu chi")
        rating_rows: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            ratings = row.get("ratings") or {}
            for feature, score in ratings.items():
                rating_rows.append(
                    {
                        "ticket_class": row["ticket_class"],
                        "feature": feature,
                        "score": float(score),
                    }
                )

        if rating_rows:
            ratings_df = pd.DataFrame(rating_rows)
            pivot = ratings_df.pivot_table(
                index="feature",
                columns="ticket_class",
                values="score",
                aggfunc="mean",
            ).fillna(0)
            fig4 = px.imshow(
                pivot,
                text_auto=True,
                color_continuous_scale="Blues",
                aspect="auto",
                title="Ban do nhiet diem trung binh dich vu",
            )
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Chua co du lieu diem chi tiet.")

    with tab3:
        high_risk = df[df["risk_level"] == "HIGH"].sort_values("risk_score", ascending=False)
        if high_risk.empty:
            st.success("Khong co danh gia rui ro cao.")
        else:
            st.error(f"Phat hien {len(high_risk)} danh gia rui ro cao can uu tien xu ly.")
            show_cols = [
                "created_at",
                "passenger_name",
                "ticket_class",
                "prediction_vi",
                "risk_score",
                "confidence",
            ]
            st.dataframe(high_risk[show_cols], use_container_width=True)

    with tab4:
        view_df = df[
            [
                "id",
                "created_at",
                "passenger_name",
                "ticket_class",
                "prediction_vi",
                "confidence",
                "risk_score",
                "risk_level_vi",
                "upload_file_name",
                "upload_file_size",
            ]
        ].copy()
        st.dataframe(view_df, use_container_width=True)
        csv_data = view_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Tai du lieu CSV",
            data=csv_data,
            file_name=f"airline_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    st.markdown("---")
    if st.button("Decision Recommendations", use_container_width=True):
        st.session_state.page_choice = "Customer Decision Recommendations"
        st.rerun()


def _extract_score_and_level_from_result(result: Dict[str, Any]) -> tuple[float, str, str]:
    pred_block = result.get("prediction", {}) if isinstance(result, dict) else {}
    score = float(pred_block.get("prob_satisfied", 0.0)) / 100.0
    level = classify_satisfaction_level(score)
    raw_prediction = str(pred_block.get("prediction", ""))
    return score, level, raw_prediction


def _extract_score_and_level_from_latest_feedback(db_df: pd.DataFrame) -> tuple[Optional[float], Optional[str], Optional[str]]:
    if db_df.empty:
        return None, None, None

    latest = db_df.iloc[0]
    prediction = str(latest.get("prediction", ""))
    confidence = float(latest.get("confidence", 0.0)) / 100.0

    # Fallback score when probability distribution is not stored in DB.
    if prediction == "Satisfied":
        score = confidence
    else:
        score = 1.0 - confidence

    score = max(0.0, min(1.0, score))
    level = classify_satisfaction_level(score)
    return score, level, prediction


def render_decision_recommendations_page(db_df: pd.DataFrame) -> None:
    render_hero(
        "Customer Decision Recommendations",
        "Trang DSS de xuat phuong an xu ly phan hoi khach hang dua tren ket qua AHP + XGBoost.",
    )

    score: Optional[float] = None
    level: Optional[str] = None
    source_note = ""
    prediction_raw = ""

    current_result = st.session_state.get("current_result")
    if current_result is not None:
        score, level, prediction_raw = _extract_score_and_level_from_result(current_result)
        source_note = "Nguon du lieu: Ket qua phan tich moi nhat tu form danh gia hanh khach."
    else:
        score, level, prediction_raw = _extract_score_and_level_from_latest_feedback(db_df)
        if score is not None:
            source_note = "Nguon du lieu: Ban ghi feedback gan nhat trong PostgreSQL."

    if score is None or level is None:
        st.info("Chua co du lieu phan tich. Hay vao trang 'Danh gia hanh khach' de tao ket qua truoc.")
        return

    plans = []
    if current_result is not None and isinstance(current_result, dict) and "impact_analysis" in current_result:
        impact_rows = current_result["impact_analysis"].to_dict("records")
        plans = rank_recommendation_plans_by_impact(level, impact_rows)
    if not plans:
        plans = get_recommendation_plans(level)
    level_text = "satisfied" if level == "satisfied" else "unsatisfied"

    a, b, c = st.columns(3)
    with a:
        st.metric("Customer Satisfaction Level", level_text)
    with b:
        st.metric("Satisfaction Score", f"{score:.2f}")
    with c:
        st.metric("Model Prediction", prediction_raw or "N/A")

    st.caption(source_note)
    st.caption("Decision logic: score >= 0.7 => satisfied, score < 0.7 => unsatisfied")

    st.markdown("### Danh sach phuong an de xuat")
    for idx, plan in enumerate(plans, start=1):
        st.markdown(
            f"""
            <div class="panel" style="margin-bottom: 12px; border-left: 6px solid {'#22a36e' if level == 'satisfied' else '#d44747'};">
                <h3 style="margin-bottom: 6px;">PA{idx}. {plan['title']}</h3>
                <p style="margin-bottom: 6px;">{plan['description']}</p>
                <p><strong>Ly do de xuat:</strong> {plan.get('reason', 'De xuat theo muc do hai long tong hop.')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_about_page() -> None:
    render_hero(
        "Gioi thieu he thong thu thap danh gia hanh khach",
        "Nen tang nay thu thap danh gia, phan tich hai long va ho tro bo phan van hanh nang cao chat luong dich vu hang khong.",
    )

    st.markdown(
        """
        <div class="panel">
            <h3>Mo hinh va du lieu</h3>
            <p>He thong ket hop XGBoost va AHP de du doan hai long, dong thoi xac dinh tieu chi anh huong de uu tien hanh dong.</p>
        </div>
        <div class="panel">
            <h3>Thu thap va quan tri du lieu</h3>
            <p>Moi phieu danh gia duoc luu vao PostgreSQL de theo doi dai han, khong bi mat khi tai lai trang.</p>
        </div>
        <div class="panel">
            <h3>Bao mat dashboard</h3>
            <p>Dashboard quan tri duoc bao ve boi dang nhap admin va chi hien thi cho tai khoan da xac thuc.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Airline Feedback DSS",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_styles()
    init_session_state()

    if not db_enabled():
        st.warning("Chua tim thay psycopg2. Hay cai dat dependencies de su dung PostgreSQL.")
    else:
        init_database()

    dss, models, ahp_weights = prepare_dss()

    render_admin_login_sidebar()

    st.sidebar.markdown("### Dieu huong")
    page = st.sidebar.radio(
        "Chon trang",
        [
            "Trang chu",
            "Danh gia hanh khach",
            "Dashboard quan tri",
            "Customer Decision Recommendations",
            "Gioi thieu",
        ],
        key="page_choice",
    )

    db_df = load_submissions_from_db() if db_enabled() else pd.DataFrame()

    if page == "Trang chu":
        render_home_page(db_df, dss, ahp_weights, models)
    elif page == "Danh gia hanh khach":
        render_customer_form_page(dss, models)
    elif page == "Dashboard quan tri":
        render_admin_dashboard_page(db_df)
    elif page == "Customer Decision Recommendations":
        render_decision_recommendations_page(db_df)
    else:
        render_about_page()

    st.markdown("---")
    st.caption("Airline DSS Feedback Platform | PostgreSQL + Streamlit + XGBoost + AHP")


if __name__ == "__main__":
    main()
