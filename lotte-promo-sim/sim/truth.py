"""
'정답 세계(ground truth)'.

가상 데이터를 만들 때만 쓰고, AI 모델은 이 파일을 절대 보지 않습니다.
나중에 "모델이 이 정답을 얼마나 맞혔는가"를 채점하는 데 다시 씁니다.
→ 실제 데이터가 없다는 약점을, 정확성을 증명하는 장치로 바꾸는 부분.
"""
import numpy as np

# 카테고리마다 혜택에 반응하는 정도가 다르다 (예: 가전은 무이자·할인, 뷰티는 증정·구성)
SENSITIVITY = {
    #            할인   사은품  무이자  적립   당겨쓰기율
    "뷰티":     (0.6,  1.4,   0.1,   0.6,  0.35),
    "건강식품": (0.8,  0.9,   0.2,   0.9,  0.45),
    "패션":     (0.9,  0.6,   0.3,   0.5,  0.15),
    "생활가전": (1.3,  0.5,   1.6,   0.4,  0.20),
    "식품":     (0.5,  0.7,   0.0,   1.2,  0.50),
}
BASE_ORDERS = {"뷰티": 3000, "건강식품": 2000, "패션": 1500, "생활가전": 400, "식품": 4000}
SLOT_MULT = {"아침(06-09)": 0.8, "오전(09-12)": 0.9, "오후(12-18)": 0.85,
             "프라임(20-23)": 1.35, "심야(23-02)": 0.6}
ADDON_LIFT = {0: 0.0, 15: 0.06, 30: 0.10, 45: 0.05}   # 중반 투입이 가장 효과적


def base_mu(category, price, ref_price, popularity, slot, weekend, host_tier):
    """프로모션이 없을 때의 기대 주문 수."""
    mu = BASE_ORDERS[category] * SLOT_MULT[slot]
    mu *= 1.1 if weekend else 1.0
    mu *= 1.15 if host_tier == "A" else 1.0
    mu *= np.exp(0.3 * popularity)
    mu *= (price / ref_price) ** -1.2
    return mu


def promo_lift(category, disc_pct, gift_level, noint_months, points_pct, addon_min):
    """프로모션의 '진짜' 효과(배수 - 1). 혜택을 겹칠수록 효과가 체감된다."""
    s_card, s_gift, s_ni, s_pt, _ = SENSITIVITY[category]
    L = 0.0
    L += s_card * 0.40 * (1 - np.exp(-disc_pct / 8))
    L += s_gift * {0: 0.0, 1: 0.10, 2: 0.16}[int(gift_level)]
    L += s_ni * 0.20 * (1 - np.exp(-noint_months / 6))
    L += s_pt * 0.12 * (1 - np.exp(-points_pct / 1.5))
    L += ADDON_LIFT[int(addon_min)]
    return L / (1 + 0.6 * L)          # 체감(포화) 효과


def pull_rate(category):
    """늘어난 주문 중 '다음 주에 살 사람을 당겨온' 비율."""
    return SENSITIVITY[category][4]


from .minute_profile import minute_curve, share_after  # noqa: E402,F401
