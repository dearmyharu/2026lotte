"""
손익 계산 + 최적안 탐색.

홈쇼핑 프로모션 비용은 홈쇼핑과 협력사가 나눠 낸다 (카드 청구할인은 상시 혜택이라 제외).
그래서 '매출이 가장 큰 조합'이 아니라
'홈쇼핑 이익이 가장 크면서 협력사도 받아들일 수 있는 조합'을 찾는다.
또 방송 후 7일 주문 감소(당겨쓰기)까지 빼서 '진짜 증분 이익'으로 판단한다.
"""
import itertools
import numpy as np
import pandas as pd

from .minute_profile import share_after
from .config import (CATEGORIES, SLOT_COST, DISC_OPTIONS, GIFT_OPTIONS, NOINT_OPTIONS,
                     POINT_OPTIONS, ADDON_OPTIONS, GIFT_COST_RATIO, NOINT_COST_PER_MONTH,
                     POINT_REDEEM, ADDON_COST_RATIO, DEFAULT_SHARES, PARTNER_SHARE_LIMIT,
                     promo_label)

NO_PROMO = dict(disc_pct=0, gift_level=0, noint_months=0, points_pct=0, addon_min=0)


def all_combos(include_addon=True):
    addons = ADDON_OPTIONS if include_addon else [0]
    for c, g, n, p, a in itertools.product(DISC_OPTIONS, GIFT_OPTIONS, NOINT_OPTIONS,
                                           POINT_OPTIONS, addons):
        yield dict(disc_pct=c, gift_level=g, noint_months=n, points_pct=p, addon_min=a)


def pnl(ctx, promo, orders, post7, base_post7, shares=None):
    """
    한 번의 방송에 대한 손익. orders/post7 는 앙상블 멤버별 배열이어도 된다.
    반환: 홈쇼핑 이익, 협력사 이익, 판촉비(주체별), 취급고
    """
    s = shares or DEFAULT_SHARES
    cat = CATEGORIES[ctx["category"]]
    P, c, cr = ctx["price"], cat["commission"], cat["cost_ratio"]
    gmv = orders * P                                   # 취급고

    disc = gmv * promo["disc_pct"] / 100
    gift = orders * P * GIFT_COST_RATIO[promo["gift_level"]]
    noint = gmv * NOINT_COST_PER_MONTH * promo["noint_months"]
    points = gmv * promo["points_pct"] / 100 * POINT_REDEEM
    addon = orders * share_after(promo["addon_min"]) * P * ADDON_COST_RATIO

    hs_cost = (disc * s["disc"]["hs"] + gift * s["gift"]["hs"] + noint * s["noint"]["hs"]
               + points * s["points"]["hs"] + addon * s["addon"]["hs"])
    pt_cost = (disc * s["disc"]["partner"] + gift * s["gift"]["partner"]
               + noint * s["noint"]["partner"] + points * s["points"]["partner"]
               + addon * s["addon"]["partner"])
    cardco_cost = disc * 0.0          # 카드사 분담 없음 (호환용)

    # 당겨쓰기: 방송 후 7일 주문이 기준보다 줄어든 만큼 미래 이익 손실
    lost_post = (base_post7 - post7) * P
    hs = gmv * c - hs_cost - SLOT_COST[ctx["slot"]] - lost_post * c
    partner = gmv * (1 - c - cr) - pt_cost - lost_post * (1 - c - cr)
    return dict(hs_profit=hs, partner_profit=partner, gmv=gmv, hs_cost=hs_cost,
                partner_cost=pt_cost, cardco_cost=cardco_cost, lost_post_gmv=lost_post)


def _ratio(pt, hs):
    return float(pt / (pt + hs)) if (pt + hs) > 0 else 0.0


def _ctx_frame(ctx, promos):
    return pd.DataFrame([{**ctx, **p} for p in promos])


def evaluate(model, ctx, promos, shares=None):
    """여러 조합을 한꺼번에 시뮬레이션. 각 조합마다 평균·90% 구간·성공확률 반환."""
    promos = list(promos)
    o, p = model.predict_all(_ctx_frame(ctx, [NO_PROMO] + promos))
    base_o, base_p = o[:, 0], p[:, 0]
    base = pnl(ctx, NO_PROMO, base_o, base_p, base_p, shares)
    rows = []
    for j, promo in enumerate(promos, start=1):
        r = pnl(ctx, promo, o[:, j], p[:, j], base_p, shares)
        d_hs = r["hs_profit"] - base["hs_profit"]
        d_pt = r["partner_profit"] - base["partner_profit"]
        rows.append({
            **promo, "label": promo_label(promo),
            "orders": o[:, j].mean(), "orders_lo": np.percentile(o[:, j], 5),
            "orders_hi": np.percentile(o[:, j], 95),
            "lift_pct": (o[:, j] / base_o - 1).mean() * 100,
            "pull_forward_orders": (base_p - p[:, j]).mean(),
            "gmv": r["gmv"].mean(),
            "hs_profit": r["hs_profit"].mean(),
            "hs_delta": d_hs.mean(), "hs_delta_lo": np.percentile(d_hs, 5),
            "hs_delta_hi": np.percentile(d_hs, 95),
            "p_better": (d_hs > 0).mean(),                # 이익이 늘어날 확률
            "partner_delta": d_pt.mean(), "partner_delta_lo": np.percentile(d_pt, 5),
            "hs_cost": r["hs_cost"].mean(), "partner_cost": r["partner_cost"].mean(),
            "cardco_cost": r["cardco_cost"].mean(),
            # 법규 점검: 홈쇼핑+협력사 판촉비 중 협력사 몫 (카드사 분담분 제외)
            "partner_share": _ratio(r["partner_cost"].mean(), r["hs_cost"].mean()),
        })
    out = pd.DataFrame(rows)
    out["legal_ok"] = out["partner_share"] <= PARTNER_SHARE_LIMIT + 1e-9
    out.attrs["baseline"] = {"orders": base_o.mean(), "hs_profit": base["hs_profit"].mean(),
                             "partner_profit": base["partner_profit"].mean(),
                             "post7": base_p.mean()}
    return out


def optimize(model, ctx, objective="hs", partner_floor=0.0, hs_budget=None,
             min_p_better=0.0, shares=None, top_k=5, include_addon=True, safe=False,
             allow_partner_initiated=False):
    """
    objective: "hs"(홈쇼핑 이익) 또는 "joint"(홈쇼핑+협력사 합계)
    partner_floor: 협력사 이익 변화 하한(원). 0이면 '협력사가 손해 보지 않는 조합만'.
    allow_partner_initiated: True면 협력사가 자발적으로 요청한 차별화 행사로 보고 50% 상한 예외 적용
    safe: True면 협력사 조건을 '평균'이 아니라 '비관적 시나리오(하위 5%)'에서도 지키는 조합만
    hs_budget: 홈쇼핑 부담 판촉비 상한(원)
    min_p_better: 이익 증가 확률 최소치 (불확실한 조합 거르기)
    """
    res = evaluate(model, ctx, all_combos(include_addon), shares)
    pcol = "partner_delta_lo" if safe else "partner_delta"
    feasible = res[res[pcol] >= partner_floor]
    if not allow_partner_initiated:
        feasible = feasible[feasible["legal_ok"]]       # 대규모유통업법 50% 분담 상한
    if hs_budget is not None:
        feasible = feasible[feasible["hs_cost"] <= hs_budget]
    # 보수 모드: 협력사 조건뿐 아니라 홈쇼핑 이익도 비관적 시나리오에서 손해가 아닌 조합만
    feasible = feasible[feasible["p_better"] >= (max(min_p_better, 0.9) if safe else min_p_better)]
    key = "hs_delta" if objective == "hs" else "_joint"
    feasible = feasible.assign(_joint=feasible["hs_delta"] + feasible["partner_delta"])
    ranked = feasible.sort_values(key, ascending=False)
    ranked.attrs = res.attrs
    return ranked.head(top_k), res
