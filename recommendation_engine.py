from __future__ import annotations

from typing import Dict, List, Optional


Recommendation = Dict[str, str]

# Tiêu chí có chiều ngược (giá trị càng nhỏ càng tốt)
INVERSE_CRITERIA = [
    'Departure Delay in Minutes',
    'Arrival Delay in Minutes'
]

# Ngưỡng delay chấp nhận được (phút)
DELAY_THRESHOLD_GOOD = 15      # <= 15 phút = tốt
DELAY_THRESHOLD_ACCEPTABLE = 30  # <= 30 phút = chấp nhận

# ── NGÂN HÀNG PA CHO TỪNG NHÓM TIÊU CHÍ ──────────────

PA_BANK = {

  # ── UNSATISFIED ──────────────────────────────────────

  'unsatisfied': {

    # Nhóm 1: Delay (Departure/Arrival Delay)
    'delay': {
      'PA1': {
        'name': 'Cải thiện đúng giờ bay',
        'desc': 'Tối ưu lịch bay, giảm thời gian chờ tại cổng, cải thiện quy trình boarding để giảm delay.',
        'icon': '⏱️',
        'apply_when': 'Delay > 30 phút, ảnh hưởng trực tiếp đến trải nghiệm khách hàng'
      },
      'PA2': {
        'name': 'Hỗ trợ & chăm sóc khách hàng khi delay',
        'desc': 'Cung cấp suất ăn nhẹ, nước uống tại sân bay, bố trí khu vực chờ thoải mái, cập nhật thông tin chuyến bay liên tục.',
        'icon': '🎧',
        'apply_when': 'Delay đã xảy ra, cần duy trì trải nghiệm tốt trong thời gian chờ'
      },
      'PA3': {
        'name': 'Thông báo chủ động',
        'desc': 'Gửi SMS/email sớm khi phát hiện nguy cơ delay, cung cấp lựa chọn đổi chuyến bay.',
        'icon': '📢',
        'apply_when': 'Cải thiện trải nghiệm khi delay không tránh khỏi'
      }
    },

    # Nhóm 2: Dịch vụ trên chuyến bay
    'onboard_service': {
      'PA1': {
        'name': 'Nâng cao chất lượng dịch vụ cabin',
        'desc': 'Cải thiện tiêu chí yếu nhất: vệ sinh cabin, ghế ngồi, thái độ tiếp viên, chất lượng đồ ăn theo phản hồi thực tế.',
        'icon': '✈️',
        'apply_when': 'Khách unsatisfied với dịch vụ cơ bản trong cabin'
      },
      'PA2': {
        'name': 'Phản hồi & cam kết cải thiện',
        'desc': 'Gửi email/tin nhắn cảm ơn phản hồi, thông báo các bước hãng sẽ cải thiện, mời khách tham gia khảo sát lần sau.',
        'icon': '📩',
        'apply_when': 'Khách unsatisfied, cần giữ mối quan hệ'
      },
      'PA3': {
        'name': 'Nâng cấp trang thiết bị cabin',
        'desc': 'Đầu tư ghế mới, hệ thống giải trí, cải thiện không gian để chân theo hạng vé.',
        'icon': '🛋️',
        'apply_when': 'Seat Comfort hoặc Leg Room dưới 3.0/5'
      }
    },

    # Nhóm 3: Công nghệ & Tiện ích
    'technology': {
      'PA1': {
        'name': 'Nâng cấp hệ thống wifi & giải trí',
        'desc': 'Đầu tư băng thông wifi tốc độ cao, cập nhật nội dung giải trí đa dạng hơn.',
        'icon': '📡',
        'apply_when': 'Wifi hoặc Entertainment dưới 3.0/5'
      },
      'PA2': {
        'name': 'Cải thiện đặt vé trực tuyến',
        'desc': 'Tối ưu UX website/app đặt vé, giảm số bước đặt vé, hỗ trợ 24/7.',
        'icon': '💻',
        'apply_when': 'Ease of Online Booking thấp'
      },
      'PA3': {
        'name': 'Hỗ trợ kỹ thuật tại chỗ',
        'desc': 'Bố trí nhân viên hỗ trợ kỹ thuật wifi và thiết bị giải trí trên mỗi chuyến bay.',
        'icon': '🔧',
        'apply_when': 'Wifi liên tục gặp sự cố kỹ thuật'
      }
    },

    # Nhóm 4: Vận hành mặt đất
    'ground_operation': {
      'PA1': {
        'name': 'Tối ưu quy trình mặt đất',
        'desc': 'Cải thiện quy trình boarding, xử lý hành lý nhanh hơn, bố trí cổng thuận tiện hơn.',
        'icon': '🏛️',
        'apply_when': 'Gate Location hoặc Baggage Handling thấp'
      },
      'PA2': {
        'name': 'Tăng cường nhân sự mặt đất',
        'desc': 'Bổ sung nhân viên hỗ trợ tại cổng, quầy check-in và khu vực nhận hành lý.',
        'icon': '👥',
        'apply_when': 'Thời gian chờ đợi dài tại sân bay'
      },
      'PA3': {
        'name': 'Ứng dụng tự phục vụ',
        'desc': 'Mở rộng kiosk check-in tự động, tag hành lý điện tử, cổng tự động.',
        'icon': '🤖',
        'apply_when': 'Tắc nghẽn tại quầy check-in thủ công'
      }
    }
  },

  # ── SATISFIED ────────────────────────────────────────

  'satisfied': {
    'Business': {
      'PA1': {
        'name': 'Duy trì tiêu chuẩn dịch vụ hạng thương gia',
        'desc': 'Tiếp tục duy trì chất lượng phòng chờ VIP, bữa ăn cao cấp và dịch vụ cá nhân hóa.',
        'icon': '👑',
        'apply_when': 'Business class satisfied toàn diện'
      },
      'PA2': {
        'name': 'Chương trình thành viên hạng cao',
        'desc': 'Tặng thêm miles, ưu tiên nâng hạng tự động, quà tặng đặc biệt cho khách hàng thân thiết.',
        'icon': '💎',
        'apply_when': 'Giữ chân khách Business class'
      },
      'PA3': {
        'name': 'Dịch vụ cá nhân hóa VIP',
        'desc': 'Ghi nhớ sở thích cá nhân, chủ động chuẩn bị trải nghiệm riêng cho từng chuyến bay tiếp theo.',
        'icon': '🌟',
        'apply_when': 'Khách Business có điểm satisfaction rất cao'
      }
    },
    'Economy': {
      'PA1': {
        'name': 'Duy trì chất lượng dịch vụ phổ thông',
        'desc': 'Tiếp tục duy trì các tiêu chí đang tốt, đặc biệt là đúng giờ và vệ sinh cabin.',
        'icon': '✅',
        'apply_when': 'Economy satisfied, duy trì ổn định'
      },
      'PA2': {
        'name': 'Đề xuất nâng hạng lên Eco Plus',
        'desc': 'Gợi ý khách hàng trải nghiệm Eco Plus với không gian rộng hơn, giá hợp lý.',
        'icon': '⬆️',
        'apply_when': 'Khách Economy satisfied, có tiềm năng nâng hạng'
      },
      'PA3': {
        'name': 'Chương trình tích điểm phổ thông',
        'desc': 'Cung cấp điểm thưởng cho mỗi chuyến bay, đổi điểm lấy ghế nâng cấp hoặc hành lý thêm.',
        'icon': '🎯',
        'apply_when': 'Giữ chân khách Economy thường xuyên'
      }
    },
    'Eco Plus': {
      'PA1': {
        'name': 'Duy trì và phát huy Eco Plus',
        'desc': 'Tiếp tục duy trì ưu điểm không gian rộng, dịch vụ tốt hơn Economy cơ bản.',
        'icon': '✅',
        'apply_when': 'Eco Plus satisfied, ổn định'
      },
      'PA2': {
        'name': 'Đề xuất nâng hạng Business',
        'desc': 'Giới thiệu trải nghiệm Business class với ưu đãi thử nghiệm lần đầu.',
        'icon': '⬆️',
        'apply_when': 'Eco Plus satisfied cao, sẵn sàng nâng hạng'
      },
      'PA3': {
        'name': 'Ưu đãi khách hàng trung thành Eco Plus',
        'desc': 'Tặng miles, ưu tiên chọn ghế sớm, miễn phí hành lý thêm cho khách thường xuyên.',
        'icon': '🎁',
        'apply_when': 'Giữ chân khách Eco Plus thường xuyên'
      }
    }
  }
}

CRITERIA_GROUPS = {
  'delay': [
      'Departure Delay in Minutes',
      'Arrival Delay in Minutes'
  ],
  'onboard_service': [
      'Seat comfort',
      'Leg room service',
      'Cleanliness',
      'Food and drink',
      'Inflight service',
      'On-board service'
  ],
  'technology': [
      'Inflight wifi service',
      'Ease of Online booking',
      'Inflight entertainment'
  ],
  'ground_operation': [
      'Gate location',
      'Baggage handling',
      'Checkin service',
      'Departure/Arrival time convenient'
  ]
}

def get_criteria_group(criteria_name):
    """Xác định nhóm của 1 tiêu chí"""
    for group, criteria_list in CRITERIA_GROUPS.items():
        if any(criteria_name.lower() == c.lower() for c in criteria_list):
            return group
    return 'onboard_service'

def normalize_to_satisfaction(criteria_name: str, raw_value: float) -> float:
    if criteria_name in INVERSE_CRITERIA:
        max_delay = 120.0
        score = max(0.0, 1.0 - (float(raw_value) / max_delay))
        return round(score, 4)
    else:
        val = max(1.0, min(5.0, float(raw_value)))
        return round((val - 1) / 4, 4)

def is_weak_criteria(criteria_name: str, raw_value: float) -> bool:
    if criteria_name in INVERSE_CRITERIA:
        return float(raw_value) > DELAY_THRESHOLD_ACCEPTABLE
    else:
        return float(raw_value) < 3.5

def get_criteria_label(criteria_name: str, raw_value: float) -> tuple[str, str]:
    val = float(raw_value)
    if criteria_name in INVERSE_CRITERIA:
        if val <= DELAY_THRESHOLD_GOOD:
            return "✅ Tốt", "success"
        elif val <= DELAY_THRESHOLD_ACCEPTABLE:
            return "⚠️ Chấp nhận được", "warning"
        else:
            return "⚠️ Cần cải thiện", "danger"
    else:
        if val >= 4.0:
            return "✅ Tốt", "success"
        elif val >= 3.5:
            return "⚠️ Chấp nhận được", "warning"
        else:
            return "⚠️ Cần cải thiện", "danger"

def classify_satisfaction_level(score: float) -> str:
    return "satisfied" if float(score) >= 0.6 else "unsatisfied"

def rank_pa_by_ahp(pa_set, ahp_scores):
    """
    Xếp hạng PA trong bộ bằng AHP score.
    ahp_scores = {'PA1': 0.409, 'PA2': 0.325, 'PA3': 0.266}
    """
    pa_list = []
    if ahp_scores and len(ahp_scores) >= 3:
        # Sort based on AHP score (PA1, PA2, PA3 keys in AHP score mapping)
        sorted_keys = sorted(ahp_scores.keys(), key=lambda k: ahp_scores[k], reverse=True)
        # Map those sorted keys to the plans in pa_set (which always has PA1, PA2, PA3 keys)
        for k in sorted_keys:
            if k in pa_set:
                pa_list.append(pa_set[k])
    else:
        pa_list = list(pa_set.values())

    while len(pa_list) < 3:
        pa_list.append(pa_list[-1] if pa_list else {})
    return pa_list[:3]

def select_pa_combined(prediction, seat_class, impact_analysis, ahp_scores, prob_satisfied, risk_score):
    is_satisfied = (prediction == 'satisfied')
    
    if is_satisfied:
        seat_key = seat_class if seat_class in PA_BANK['satisfied'] else 'Economy'
        pa_set = PA_BANK['satisfied'][seat_key]
        ranked = rank_pa_by_ahp(pa_set, ahp_scores)
        
        return {
            'PA1': ranked[0],
            'PA2': ranked[1],
            'PA3': ranked[2],
            'recommended': 'PA1',
            'reasoning': f'Khách hàng {seat_class} hài lòng (score={prob_satisfied/100:.0%}). Tập trung giữ chân và phát triển.'
        }
    else:
        # impact_analysis is list of dicts with keys: Feature, Current_Value, Impact_Score, Impact_%
        weak = [c for c in impact_analysis if is_weak_criteria(c.get('Feature', ''), c.get('Current_Value', 5.0))]
        weak_sorted = sorted(weak, key=lambda x: x.get('Impact_Score', 0), reverse=True)

        if not weak_sorted:
            weak_sorted = sorted(impact_analysis, key=lambda x: x.get('Impact_Score', 0), reverse=True)

        top_criteria = weak_sorted[0]['Feature'] if weak_sorted else ''
        primary_group = get_criteria_group(top_criteria)
        pa_set = PA_BANK['unsatisfied'][primary_group]
        ranked = rank_pa_by_ahp(pa_set, ahp_scores)

        if risk_score >= 70 and primary_group != 'delay':
            comp_pa = {
                'name': 'Liên hệ chăm sóc khách hàng trực tiếp',
                'desc': 'Đội ngũ CSKH chủ động liên hệ khách trong 24h, lắng nghe phản hồi, giải thích rõ sự cố và cam kết cải thiện chuyến bay tiếp theo.',
                'icon': '📞',
                'apply_when': 'Risk score cao >= 70, khách hàng có trải nghiệm rất tiêu cực'
            }
            ranked = [comp_pa] + ranked[:2]

        top3_weak = [c['Feature'] for c in weak_sorted[:3]]
        reasoning = (
            f'Khách hàng không hài lòng (score={prob_satisfied/100:.0%}, risk={risk_score:.0f}). '
            f'Tiêu chí yếu nhất: {", ".join(top3_weak)}. Nhóm vấn đề: {primary_group}.'
        )

        return {
            'PA1': ranked[0],
            'PA2': ranked[1],
            'PA3': ranked[2],
            'recommended': 'PA1',
            'top_weak_criteria': top3_weak,
            'primary_group': primary_group,
            'reasoning': reasoning
        }


def get_recommendation_plans(prediction, seat_class='Economy',
                             impact_analysis=None, ahp_scores=None,
                             prob_satisfied=50.0, risk_score=5.0):
    """
    Wrapper gọi select_pa_combined, trả về bộ PA1/PA2/PA3.
    Hỗ trợ cả tham số cũ (chỉ nhận prediction) và tham số mới.
    """
    # Nếu prediction là list Recommendation từ bản cũ, trả về chính nó
    if isinstance(prediction, list):
        return prediction

    res = select_pa_combined(
        prediction=prediction,
        seat_class=seat_class,
        impact_analysis=impact_analysis or [],
        ahp_scores=ahp_scores or {},
        prob_satisfied=prob_satisfied,
        risk_score=risk_score
    )
    
    # Trả về list các Recommendation (name, desc) để tương thích bản cũ
    return [
        {"title": res['PA1']['name'], "description": res['PA1']['desc']},
        {"title": res['PA2']['name'], "description": res['PA2']['desc']},
        {"title": res['PA3']['name'], "description": res['PA3']['desc']}
    ]

def rank_recommendation_plans_by_impact(satisfaction_level, impact_rows=None):
    """
    Wrapper tương thích cho survey route.
    """
    return get_recommendation_plans(satisfaction_level, impact_analysis=impact_rows)
