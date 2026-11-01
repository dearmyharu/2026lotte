"""
분당 주문 곡선 (방송 60분).

실제 운영에서는 과거 방송의 '분당 주문 로그'를 평균 내서 얻는 값이다.
여기서는 홈쇼핑의 전형적인 패턴(오프닝 집중 → 완만 → 마감 임박 반등)을 가정했다.
추가혜택 비용(투입 이후 주문에만 지급)과 타이밍 화면에 쓴다.
"""
import numpy as np

def minute_curve(addon_min=0, minutes=60):
    m = np.arange(minutes)
    base = 1.6 * np.exp(-m / 10) + 0.8 + 1.4 * np.exp((m - minutes) / 4)
    if addon_min:
        bump = np.where(m >= addon_min, 0.9 * np.exp(-(m - addon_min) / 6) + 0.12, 0)
        base = base * (1 + bump)
    return base / base.sum()


def share_after(addon_min, minutes=60):
    """추가혜택 투입 이후에 들어온 주문 비중 → 추가혜택 비용 계산에 사용."""
    if not addon_min:
        return 0.0
    c = minute_curve(addon_min, minutes)
    return float(c[addon_min:].sum())
