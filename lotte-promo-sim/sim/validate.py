"""
검증: "이 시뮬레이터를 믿어도 되는가?"

1) 예측 정확도: 학습에 안 쓴 방송(20%)의 주문 수를 얼마나 맞히는가
2) 효과 추정 정확도: 혜택별 '진짜 효과'(정답 세계)를 얼마나 맞히는가
   - 비교 대상: 현업에서 흔한 '혜택 있었던 방송 vs 없었던 방송' 단순 평균 비교
3) 의사결정 정확도: 모델이 추천한 조합을 정답 세계에서 돌렸을 때,
   정답을 아는 '신(oracle)'이 고른 조합 대비 이익을 몇 % 확보하는가
"""
import json
import numpy as np
import pandas as pd

from . import truth
from .config import CATEGORIES
from .economics import NO_PROMO, all_combos, optimize, pnl
from .model import PromoModel

LEVERS = {  # (변수, 비교할 값)
    "즉시할인 10%": ("disc_pct", 10), "전고객 증정": ("gift_level", 2),
    "무이자 12개월": ("noint_months", 12), "추가적립 3%": ("points_pct", 3),
}


def true_orders(ctx, promo):
    ref = CATEGORIES[ctx["category"]]["price"]
    mu0 = truth.base_mu(ctx["category"], ctx["price"], ref, ctx["popularity"], ctx["slot"],
                        ctx["weekend"], ctx["host_tier"])
    mu = mu0 * (1 + truth.promo_lift(ctx["category"], **promo))
    post = max(0.45 * mu0 - truth.pull_rate(ctx["category"]) * (mu - mu0), 1.0)
    return mu, post, mu0


def true_hs_delta(ctx, promo):
    mu, post, mu0 = true_orders(ctx, promo)
    _, post0, _ = true_orders(ctx, NO_PROMO)
    r = pnl(ctx, promo, mu, post, post0)
    b = pnl(ctx, NO_PROMO, mu0, post0, post0)
    return r["hs_profit"] - b["hs_profit"], r["partner_profit"] - b["partner_profit"]


def run(df, n_boot=20, n_ctx=60, seed=1):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cut = int(len(df) * 0.8)
    train, test = df.iloc[idx[:cut]], df.iloc[idx[cut:]]
    model = PromoModel(n_boot=n_boot).fit(train)
    report = {}

    # 1) 예측 정확도
    pred, _ = model.predict(test)
    y = test["orders"].to_numpy()
    report["holdout"] = {
        "n_test": int(len(test)),
        "mape_pct": float(np.mean(np.abs(pred - y) / y) * 100),
        "r2": float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
    }

    # 2) 혜택별 효과 추정: 카테고리마다, 테스트 방송 조건에서 평균 증분 효과(%)
    effect_rows = []
    ctx_cols = ["category", "price", "popularity", "slot", "weekend", "host_tier"]
    for cat in CATEGORIES:
        sub = test[test["category"] == cat]
        allc = df[df["category"] == cat]
        for name, (col, val) in LEVERS.items():
            on, off = dict(NO_PROMO), dict(NO_PROMO)
            on[col] = val
            ctx_df = sub[ctx_cols]
            p_on, _ = model.predict(ctx_df.assign(**on))
            p_off, _ = model.predict(ctx_df.assign(**off))
            est = float(np.mean(p_on / p_off - 1) * 100)
            true = float(truth.promo_lift(cat, **on) * 100)
            naive = float((allc.loc[allc[col] == val, "orders"].mean()
                           / allc.loc[allc[col] == 0, "orders"].mean() - 1) * 100)
            effect_rows.append(dict(category=cat, lever=name, true=true, model=est, naive=naive))
    eff = pd.DataFrame(effect_rows)
    report["effects"] = eff.to_dict("records")
    report["effect_mae"] = {"model_pp": float((eff.model - eff.true).abs().mean()),
                            "naive_pp": float((eff.naive - eff.true).abs().mean())}

    # 3) 의사결정 정확도 (oracle 대비). 일반 모드 / 보수 모드(협력사 조건을 비관 시나리오에서도 준수)
    ctxs = test.sample(n_ctx, random_state=seed)[ctx_cols].to_dict("records")
    combos = list(all_combos())
    maxp = dict(disc_pct=15, gift_level=2, noint_months=12, points_pct=3, addon_min=30)
    oracle, max_promo, picks = [], [], {False: [], True: []}
    for ctx in ctxs:
        vals = [true_hs_delta(ctx, c) for c in combos]
        oracle.append(max([v[0] for v in vals if v[1] >= 0] + [0.0]))
        max_promo.append(true_hs_delta(ctx, maxp)[0])
        mu, post0, mu0 = true_orders(ctx, NO_PROMO)
        base_pt = pnl(ctx, NO_PROMO, mu0, post0, post0)["partner_profit"]
        for safe in (False, True):
            top, _ = optimize(model, ctx, top_k=1, safe=safe)
            pick = {k: int(top.iloc[0][k]) for k in NO_PROMO} if len(top) else NO_PROMO
            hs, pt = true_hs_delta(ctx, pick)
            picks[safe].append((hs, pt / base_pt * 100))
    O = np.sum(oracle)
    def summ(lst):
        hs = np.array([x[0] for x in lst]); pt = np.array([x[1] for x in lst])
        return {"capture_pct": float(hs.sum() / O * 100),
                "loss_cases_pct": float((hs < 0).mean() * 100),
                "partner_change_avg_pct": float(pt.mean()),
                "partner_ok_pct": float((pt >= -1).mean() * 100)}
    report["decision"] = {
        "n_contexts": n_ctx,
        "avg_oracle_gain_krw": float(np.mean(oracle)),
        "max_promo_capture_pct": float(np.sum(max_promo) / O * 100),
        "normal": summ(picks[False]), "conservative": summ(picks[True]),
    }
    return model, report


if __name__ == "__main__":
    import os, pickle, time
    t = time.time()
    df = pd.read_csv("data/broadcasts.csv")
    model, rep = run(df)
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/validation.json", "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    # 앱용 최종 모델은 전체 데이터로 다시 학습
    final = PromoModel(n_boot=20).fit(df)
    with open("artifacts/model.pkl", "wb") as f:
        pickle.dump(final, f)
    print(json.dumps({k: v for k, v in rep.items() if k != "effects"}, ensure_ascii=False, indent=2))
    print(pd.DataFrame(rep["effects"]).round(1).to_string())
    print(f"{time.time()-t:.0f}s")
