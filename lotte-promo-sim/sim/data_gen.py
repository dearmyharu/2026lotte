"""
가상 방송 데이터 생성기.

- 방송 1회 = 1행 (약 2년치 6,000회)
- 일부러 '현실의 함정'을 심어둔다:
  MD는 잘 팔릴 방송(프라임 시간대, 인기 상품, A급 쇼호스트)에 혜택을 더 많이 건다.
  → 그래서 단순 비교하면 혜택 효과가 실제보다 커 보인다 (교란 효과).
"""
import numpy as np
import pandas as pd

from . import truth
from .config import CATEGORIES, SLOTS, DISC_OPTIONS, GIFT_OPTIONS, NOINT_OPTIONS, POINT_OPTIONS, ADDON_OPTIONS


def _pick(rng, options, intensity):
    """혜택 강도 점수가 높을수록 더 센 옵션을 고르는 MD의 습관."""
    k = len(options)
    logits = np.arange(k) * intensity
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return options[rng.choice(k, p=p)]


def generate(n=6000, seed=42, explore_rate=0.0):
    """explore_rate: 혜택 조합을 무작위로 편성한 '탐색 방송' 비율 (A/B 실험 편성)."""
    rng = np.random.default_rng(seed)
    rows, truth_rows = [], []
    start = pd.Timestamp("2024-11-01")
    for i in range(n):
        cat = rng.choice(list(CATEGORIES))
        ref = CATEGORIES[cat]["price"]
        price = float(np.round(ref * rng.uniform(0.8, 1.2), -2))
        pop = float(rng.normal())
        slot = SLOTS[rng.choice(len(SLOTS), p=[0.18, 0.2, 0.27, 0.2, 0.15])]
        date = start + pd.Timedelta(days=int(rng.integers(0, 730)))
        weekend = int(date.dayofweek >= 5)
        host = "A" if rng.random() < 0.35 else "B"

        # MD의 혜택 결정 (교란): 잘 팔릴 방송일수록 혜택을 세게
        intensity = (0.9 * (slot == "프라임(20-23)") + 0.5 * pop
                     + 0.4 * (host == "A") + rng.normal(0, 0.5))
        # 관찰값에 맞춘 편성 습관: 즉시할인은 방송의 약 30%, 무이자는 약 93%
        card = _pick(rng, DISC_OPTIONS, intensity - 1.8)
        gift = _pick(rng, GIFT_OPTIONS, intensity - 1.0)
        noint = 0 if rng.random() < 0.07 else _pick(rng, NOINT_OPTIONS[1:],
                                                     intensity + (0.8 if cat == "생활가전" else 0.0))
        pts = _pick(rng, POINT_OPTIONS, intensity - 0.8)
        addon = _pick(rng, ADDON_OPTIONS, 0.3 * intensity)
        explore = int(rng.random() < explore_rate)
        if explore:   # 탐색 방송: 방송 조건과 무관하게 혜택을 무작위 편성
            card = DISC_OPTIONS[rng.integers(len(DISC_OPTIONS))]
            gift = GIFT_OPTIONS[rng.integers(len(GIFT_OPTIONS))]
            noint = NOINT_OPTIONS[rng.integers(len(NOINT_OPTIONS))]
            pts = POINT_OPTIONS[rng.integers(len(POINT_OPTIONS))]
            addon = ADDON_OPTIONS[rng.integers(len(ADDON_OPTIONS))]

        mu0 = truth.base_mu(cat, price, ref, pop, slot, weekend, host)
        lift = truth.promo_lift(cat, card, gift, noint, pts, addon)
        mu = mu0 * (1 + lift)
        orders = rng.poisson(rng.gamma(20, mu / 20))          # 음이항 잡음

        # 방송 후 7일 앱 주문: 기본 수요 - 당겨온 만큼 감소
        post_mu = max(0.45 * mu0 - truth.pull_rate(cat) * (mu - mu0), 1.0)
        post7 = rng.poisson(rng.gamma(15, post_mu / 15))

        rows.append(dict(broadcast_id=f"B{i:05d}", date=date.date(), category=cat, price=price,
                         popularity=round(pop, 3), slot=slot, weekend=weekend, host_tier=host,
                         disc_pct=card, gift_level=gift, noint_months=noint, points_pct=pts,
                         addon_min=addon, explore=explore, orders=orders, post7_orders=post7))
        truth_rows.append(dict(broadcast_id=f"B{i:05d}", true_base_mu=mu0, true_mu=mu,
                               true_post_mu=post_mu))
    return pd.DataFrame(rows), pd.DataFrame(truth_rows)


if __name__ == "__main__":
    import os
    df, tr = generate(explore_rate=0.2)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/broadcasts.csv", index=False)
    tr.to_csv("data/truth_hidden.csv", index=False)
    print(df.head(), "\n", df.describe().T)
