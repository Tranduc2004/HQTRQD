from __future__ import annotations

from typing import Dict, List, Optional


Recommendation = Dict[str, str]


def _plan(title: str, description: str, theme: str) -> Recommendation:
    return {
        "title": title,
        "description": description,
        "theme": theme,
    }


SATISFIED_PLANS: List[Recommendation] = [
    _plan(
        "Duy trì chất lượng dịch vụ",
        "Tiếp tục duy trì các dịch vụ hiện tại vì khách hàng đang hài lòng.",
        "service_quality",
    ),
    _plan(
        "Chương trình khách hàng thân thiết",
        "Cung cấp điểm thưởng hoặc miles để giữ chân khách hàng.",
        "loyalty",
    ),
    _plan(
        "Đề xuất nâng hạng",
        "Gợi ý khách hàng nâng hạng ghế cho chuyến bay tiếp theo.",
        "upsell",
    ),
    _plan(
        "Chiến dịch marketing",
        "Sử dụng phản hồi tích cực để quảng bá thương hiệu.",
        "marketing",
    ),
    _plan(
        "Chiến lược giữ chân khách hàng",
        "Gửi ưu đãi cho chuyến bay tiếp theo.",
        "retention",
    ),
]


UNSATISFIED_PLANS: List[Recommendation] = [
    _plan(
        "Cải thiện chất lượng dịch vụ",
        "Phân tích tiêu chí có điểm thấp và cải thiện.",
        "service_quality",
    ),
    _plan(
        "Chính sách bồi thường",
        "Cung cấp voucher hoặc hoàn tiền cho khách.",
        "compensation",
    ),
    _plan(
        "Hỗ trợ khách hàng",
        "Liên hệ trực tiếp với khách hàng để giải quyết vấn đề.",
        "support",
    ),
    _plan(
        "Cải thiện vận hành",
        "Tối ưu quy trình boarding và xử lý hành lý.",
        "operations",
    ),
    _plan(
        "Nâng cấp công nghệ",
        "Cải thiện hệ thống wifi và đặt vé online.",
        "technology",
    ),
]


FEATURE_THEME_MAP: Dict[str, str] = {
    "Inflight wifi service": "technology",
    "Ease of Online booking": "technology",
    "Online boarding": "operations",
    "Departure Delay in Minutes": "operations",
    "Arrival Delay in Minutes": "operations",
    "Baggage handling": "operations",
    "Gate location": "operations",
    "Seat comfort": "service_quality",
    "Leg room service": "service_quality",
    "Inflight service": "service_quality",
    "On-board service": "service_quality",
    "Food and drink": "service_quality",
    "Inflight entertainment": "service_quality",
    "Cleanliness": "service_quality",
}


THEME_REASON_TEMPLATE: Dict[str, str] = {
    "technology": "Tiêu chí '{feature}' đang thấp ({value:.1f}/5), tác động {impact:.1f}% nên ưu tiên phương án công nghệ.",
    "operations": "Tiêu chí '{feature}' đang thấp ({value:.1f}/5), tác động {impact:.1f}% nên ưu tiên phương án vận hành.",
    "service_quality": "Tiêu chí '{feature}' đang thấp ({value:.1f}/5), tác động {impact:.1f}% nên ưu tiên nâng chất lượng dịch vụ.",
    "support": "Có dấu hiệu không hài lòng, cần tăng cường hỗ trợ trực tiếp để xử lý nhanh.",
    "compensation": "Khách hàng có nguy cơ rời bỏ, nên áp dụng chính sách bồi thường/ưu đãi phù hợp.",
    "loyalty": "Mức hài lòng tốt, có thể củng cố lòng trung thành bằng chương trình tích điểm.",
    "upsell": "Trải nghiệm tổng thể tích cực, phù hợp đề xuất nâng hạng ở chuyến tiếp theo.",
    "marketing": "Phản hồi tích cực có thể dùng như bằng chứng xã hội cho chiến dịch truyền thông.",
    "retention": "Duy trì kết nối sau chuyến bay để tăng tỉ lệ quay lại của khách hàng.",
}


def normalize_level(satisfaction_level: str) -> str:
    level = (satisfaction_level or "").strip().lower()
    if level not in {"satisfied", "unsatisfied"}:
        raise ValueError("satisfaction_level must be 'satisfied' or 'unsatisfied'")
    return level


def get_recommendation_plans(satisfaction_level: str) -> List[Recommendation]:
    level = normalize_level(satisfaction_level)
    source = SATISFIED_PLANS if level == "satisfied" else UNSATISFIED_PLANS
    return [dict(item) for item in source]


def classify_satisfaction_level(score: float) -> str:
    return "satisfied" if float(score) >= 0.7 else "unsatisfied"


def _extract_priority_feature_signals(
    impact_rows: Optional[List[Dict[str, float | str]]],
    max_features: int = 3,
) -> List[Dict[str, float | str]]:
    if not impact_rows:
        return []

    candidates: List[Dict[str, float | str]] = []
    for row in impact_rows:
        feature = str(row.get("Feature", ""))
        current_value = float(row.get("Current_Value", 0.0))
        impact_pct = float(row.get("Impact_%", 0.0))
        if feature and current_value <= 3.0:
            theme = FEATURE_THEME_MAP.get(feature, "service_quality")
            candidates.append(
                {
                    "feature": feature,
                    "value": current_value,
                    "impact": impact_pct,
                    "theme": theme,
                }
            )

    candidates.sort(key=lambda x: float(x["impact"]), reverse=True)
    return candidates[:max_features]


def rank_recommendation_plans_by_impact(
    satisfaction_level: str,
    impact_rows: Optional[List[Dict[str, float | str]]] = None,
) -> List[Recommendation]:
    plans = get_recommendation_plans(satisfaction_level)
    feature_signals = _extract_priority_feature_signals(impact_rows)

    theme_priority: Dict[str, int] = {}
    first_signal_by_theme: Dict[str, Dict[str, float | str]] = {}
    for idx, signal in enumerate(feature_signals):
        theme = str(signal["theme"])
        if theme not in theme_priority:
            theme_priority[theme] = idx
            first_signal_by_theme[theme] = signal

    ranked: List[Recommendation] = []
    for base_idx, plan in enumerate(plans):
        item = dict(plan)
        theme = item.get("theme", "")

        if theme in first_signal_by_theme:
            signal = first_signal_by_theme[theme]
            template = THEME_REASON_TEMPLATE.get(theme, "Đề xuất dựa trên tiêu chí ưu tiên từ AHP impact.")
            item["reason"] = template.format(
                feature=signal["feature"],
                value=float(signal["value"]),
                impact=float(signal["impact"]),
            )
            item["_rank_key"] = (0, theme_priority[theme], base_idx)
        else:
            item["reason"] = THEME_REASON_TEMPLATE.get(theme, "Đề xuất theo mức độ hài lòng tổng hợp.")
            item["_rank_key"] = (1, 999, base_idx)

        ranked.append(item)

    ranked.sort(key=lambda x: x["_rank_key"])
    for row in ranked:
        row.pop("_rank_key", None)
        row.pop("theme", None)
    return ranked
