"""
공통 설정: 카테고리, 방송 시간대, 프로모션 선택지, 비용 분담 기본값.

※ 숫자들은 '가정값'입니다. 실제 롯데홈쇼핑 데이터가 들어오면 이 파일만 바꾸면 됩니다.
"""

# ── 카테고리별 상품 특성 ─────────────────────────────────────────────
# price: 대표 판매가(원) / commission: 홈쇼핑 판매수수료율 / cost_ratio: 협력사 원가율
CATEGORIES = {
    "뷰티":     {"price": 59_000,  "commission": 0.35, "cost_ratio": 0.30},
    "건강식품": {"price": 79_000,  "commission": 0.33, "cost_ratio": 0.35},
    "패션":     {"price": 99_000,  "commission": 0.35, "cost_ratio": 0.35},
    "생활가전": {"price": 399_000, "commission": 0.20, "cost_ratio": 0.60},
    "식품":     {"price": 39_000,  "commission": 0.30, "cost_ratio": 0.45},
}

# ── 방송 시간대 ──────────────────────────────────────────────────────
SLOTS = ["아침(06-09)", "오전(09-12)", "오후(12-18)", "프라임(20-23)", "심야(23-02)"]
# 방송 1시간 고정비(송출수수료 배분 가정, 원). 롯데홈쇼핑은 방송매출의 약 70%가 송출수수료.
SLOT_COST = {
    "아침(06-09)": 30_000_000, "오전(09-12)": 25_000_000, "오후(12-18)": 25_000_000,
    "프라임(20-23)": 45_000_000, "심야(23-02)": 12_000_000,
}

# ── 프로모션 선택지 ──────────────────────────────────────────────────
CARD_OPTIONS = [0, 5, 7, 10]       # 카드 청구할인 %
GIFT_OPTIONS = [0, 1, 2]           # 사은품: 0 없음 / 1 실속형 / 2 고급형
GIFT_COST_RATIO = {0: 0.0, 1: 0.03, 2: 0.07}   # 사은품 원가 = 판매가 × 비율
NOINT_OPTIONS = [0, 1]             # 무이자 할부 여부
NOINT_COST_RATIO = 0.035           # 무이자 할부 수수료(판매가 대비)
POINT_OPTIONS = [0, 3, 5]          # 적립금 %
POINT_REDEEM = 0.7                 # 적립금 실제 사용률
ADDON_OPTIONS = [0, 15, 30, 45]    # 방송 중 추가혜택 투입 시점(분). 0 = 투입 안 함
ADDON_COST_RATIO = 0.02            # 추가혜택 원가 = 판매가 × 비율 (투입 후 주문에만 적용)

PROMO_COLS = ["card_pct", "gift_level", "no_interest", "points_pct", "addon_min"]
CONTEXT_COLS = ["category", "price", "popularity", "slot", "weekend", "host_tier"]

GIFT_LABEL = {0: "없음", 1: "실속형", 2: "고급형"}

# ── 비용 분담 기본값 (홈쇼핑 / 협력사 / 카드사) ──────────────────────
DEFAULT_SHARES = {
    "card":   {"hs": 0.2, "partner": 0.4, "cardco": 0.4},
    "gift":   {"hs": 0.0, "partner": 1.0},
    "noint":  {"hs": 0.5, "partner": 0.5},
    "points": {"hs": 1.0, "partner": 0.0},
    "addon":  {"hs": 0.5, "partner": 0.5},
}


def promo_label(p: dict) -> str:
    """프로모션 조합을 사람이 읽는 문장으로."""
    parts = []
    if p["card_pct"]:
        parts.append(f"카드 {p['card_pct']}%")
    if p["gift_level"]:
        parts.append(f"사은품 {GIFT_LABEL[p['gift_level']]}")
    if p["no_interest"]:
        parts.append("무이자")
    if p["points_pct"]:
        parts.append(f"적립 {p['points_pct']}%")
    if p["addon_min"]:
        parts.append(f"{p['addon_min']}분 추가혜택")
    return " + ".join(parts) if parts else "프로모션 없음"
