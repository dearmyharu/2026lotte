"""
에이전트가 호출하는 '도구'들.

LLM은 숫자를 직접 만들지 않는다. 숫자는 전부 이 도구(시뮬레이터)에서 나오고,
LLM은 MD의 질문을 해석해 어떤 도구를 부를지 정하고, 결과를 설명만 한다.
"""
import json
import numpy as np

from .config import CATEGORIES, SLOTS, GIFT_LABEL, ADDON_OPTIONS, promo_label
from .economics import evaluate, optimize, NO_PROMO


def krw(x: float) -> str:
    """원 단위를 '억/만원'으로."""
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e8:
        return f"{sign}{x / 1e8:.2f}억원"
    return f"{sign}{x / 1e4:,.0f}만원"


def make_ctx(category, slot="오후(12-18)", price=None, popularity=0.0, weekend=0, host_tier="B"):
    if category not in CATEGORIES:
        raise ValueError(f"카테고리는 {list(CATEGORIES)} 중 하나여야 합니다.")
    if slot not in SLOTS:
        match = [s for s in SLOTS if slot and slot[:2] in s]
        if not match:
            raise ValueError(f"시간대는 {SLOTS} 중 하나여야 합니다.")
        slot = match[0]
    return dict(category=category, slot=slot, price=float(price or CATEGORIES[category]["price"]),
                popularity=float(popularity), weekend=int(weekend), host_tier=host_tier)


def _promo(d):
    p = dict(NO_PROMO)
    for k in p:
        if k in d and d[k] is not None:
            p[k] = int(d[k])
    if p["addon_min"] not in ADDON_OPTIONS:
        p["addon_min"] = min(ADDON_OPTIONS, key=lambda a: abs(a - p["addon_min"]))
    return p


def _row(r, base):
    return {
        "조합": r["label"],
        "예상주문": f"{r['orders']:,.0f}건 (90% 구간 {r['orders_lo']:,.0f}~{r['orders_hi']:,.0f})",
        "주문증가율": f"{r['lift_pct']:+.1f}%",
        "홈쇼핑이익변화": f"{krw(r['hs_delta'])} (90% 구간 {krw(r['hs_delta_lo'])}~{krw(r['hs_delta_hi'])})",
        "이익증가확률": f"{r['p_better'] * 100:.0f}%",
        "협력사이익변화": krw(r["partner_delta"]),
        "당겨쓰기(방송후7일 감소주문)": f"{r['pull_forward_orders']:,.0f}건",
        "판촉비부담": f"홈쇼핑 {krw(r['hs_cost'])} / 협력사 {krw(r['partner_cost'])} / 카드사 {krw(r['cardco_cost'])}",
    }


def _base_info(res):
    b = res.attrs["baseline"]
    return {"혜택없을때_예상주문": f"{b['orders']:,.0f}건", "혜택없을때_홈쇼핑이익": krw(b["hs_profit"])}


# ── 도구 함수 ──────────────────────────────────────────────────────────
def simulate_promotion(model, context, promotion):
    ctx = make_ctx(**context)
    res = evaluate(model, ctx, [_promo(promotion)])
    return {"방송조건": ctx, **_base_info(res), "결과": _row(res.iloc[0], res.attrs)}


def compare_promotions(model, context, promotions):
    ctx = make_ctx(**context)
    res = evaluate(model, ctx, [_promo(p) for p in promotions])
    res = res.sort_values("hs_delta", ascending=False)
    return {"방송조건": ctx, **_base_info(res),
            "비교(홈쇼핑 이익 큰 순)": [_row(r, res.attrs) for _, r in res.iterrows()]}


def recommend_promotion(model, context, objective="hs", partner_floor_krw=0,
                        hs_budget_krw=None, conservative=False, top_k=3):
    ctx = make_ctx(**context)
    top, allres = optimize(model, ctx, objective=objective, partner_floor=partner_floor_krw,
                           hs_budget=hs_budget_krw, safe=conservative, top_k=top_k)
    maxp = allres[(allres.card_pct == 10) & (allres.gift_level == 2) & (allres.no_interest == 1)
                  & (allres.points_pct == 5) & (allres.addon_min == 30)].iloc[0]
    out = {"방송조건": ctx, **_base_info(allres),
           "탐색한 조합 수": f"{len(allres)}개",
           "제약": {"목표": "홈쇼핑 이익" if objective == "hs" else "홈쇼핑+협력사 합계",
                  "협력사 이익변화 하한": krw(partner_floor_krw),
                  "홈쇼핑 판촉비 상한": krw(hs_budget_krw) if hs_budget_krw else "없음",
                  "보수 모드": "켜짐(협력사 조건을 비관적 시나리오에서도 충족)" if conservative else "꺼짐"},
           "추천": [_row(r, allres.attrs) for _, r in top.iterrows()],
           "참고_혜택최대로다걸기": _row(maxp, allres.attrs)}
    if len(top) == 0:
        out["안내"] = "조건을 만족하는 조합이 없습니다. 협력사 하한이나 예산을 완화해 보세요."
    return out


def best_addon_timing(model, context, promotion=None):
    """방송 중 추가혜택을 몇 분에 꺼낼지."""
    ctx = make_ctx(**context)
    base = _promo(promotion or {})
    cands = [{**base, "addon_min": a} for a in ADDON_OPTIONS]
    res = evaluate(model, ctx, cands)
    best = res.sort_values("hs_delta", ascending=False).iloc[0]
    return {"방송조건": ctx, "기본혜택": promo_label({**base, "addon_min": 0}),
            "시점별결과": [{"투입시점": f"{int(r.addon_min)}분" if r.addon_min else "투입 안 함",
                         "예상주문": f"{r.orders:,.0f}건", "홈쇼핑이익변화": krw(r.hs_delta)}
                        for _, r in res.iterrows()],
            "추천시점": f"{int(best.addon_min)}분" if best.addon_min else "투입 안 함"}


def effect_table(model, category):
    t = model.glm.coefficients(category)
    return {"카테고리": category,
            "혜택별 주문 효과(다른 조건 동일할 때)": [
                f"{r['항목']}: {r['효과(%)']:+.1f}% (90% 구간 {r['하한']:+.1f}~{r['상한']:+.1f}%)"
                for _, r in t.iterrows()]}


def model_validation(report):
    d, e, h = report["decision"], report["effect_mae"], report["holdout"]
    return {
        "예측정확도": f"학습에 쓰지 않은 방송 {h['n_test']}회에서 R² {h['r2']:.2f}, 평균오차 {h['mape_pct']:.1f}%",
        "효과추정오차": f"모델 {e['model_pp']:.1f}%p vs 단순비교 {e['naive_pp']:.1f}%p",
        "추천품질": f"정답을 아는 최적안 대비 이익 {d['normal']['capture_pct']:.0f}% 확보, "
                 f"손실 발생 방송 {d['normal']['loss_cases_pct']:.0f}%",
        "혜택최대로다걸기": f"최적안 대비 {d['max_promo_capture_pct']:.0f}%",
        "주의": "가상 데이터 기준 검증. 실제 데이터 적용 시 재검증 필요.",
    }


# ── OpenAI 함수 호출 스키마 ────────────────────────────────────────────
_CTX = {
    "type": "object", "description": "방송 조건",
    "properties": {
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "slot": {"type": "string", "enum": SLOTS},
        "price": {"type": "number", "description": "판매가(원). 생략 시 카테고리 대표가"},
        "popularity": {"type": "number", "description": "상품 인기도 지수(-2~2, 평균 0)"},
        "weekend": {"type": "integer", "enum": [0, 1]},
        "host_tier": {"type": "string", "enum": ["A", "B"]},
    },
    "required": ["category"],
}
_PROMO = {
    "type": "object", "description": "프로모션 조합",
    "properties": {
        "card_pct": {"type": "integer", "enum": [0, 5, 7, 10], "description": "카드 청구할인 %"},
        "gift_level": {"type": "integer", "enum": [0, 1, 2], "description": "사은품 0없음 1실속형 2고급형"},
        "no_interest": {"type": "integer", "enum": [0, 1], "description": "무이자 할부"},
        "points_pct": {"type": "integer", "enum": [0, 3, 5], "description": "적립금 %"},
        "addon_min": {"type": "integer", "enum": [0, 15, 30, 45], "description": "방송 중 추가혜택 투입 시점(분)"},
    },
}


def _fn(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}}


TOOL_SCHEMAS = [
    _fn("simulate_promotion", "프로모션 조합 하나의 주문·이익·협력사 영향을 시뮬레이션",
        {"context": _CTX, "promotion": _PROMO}, ["context", "promotion"]),
    _fn("compare_promotions", "여러 프로모션 조합을 같은 방송 조건에서 비교",
        {"context": _CTX, "promotions": {"type": "array", "items": _PROMO}}, ["context", "promotions"]),
    _fn("recommend_promotion", "가능한 모든 조합(288개)을 탐색해 제약 조건 하 최적 조합 추천",
        {"context": _CTX,
         "objective": {"type": "string", "enum": ["hs", "joint"], "description": "hs=홈쇼핑 이익, joint=홈쇼핑+협력사"},
         "partner_floor_krw": {"type": "number", "description": "협력사 이익 변화 하한(원). 0=협력사 손해 없음"},
         "hs_budget_krw": {"type": "number", "description": "홈쇼핑 부담 판촉비 상한(원)"},
         "conservative": {"type": "boolean", "description": "협력사 조건을 비관적 시나리오에서도 지킬지"},
         "top_k": {"type": "integer"}}, ["context"]),
    _fn("best_addon_timing", "방송 중 추가혜택을 몇 분에 투입하는 게 좋은지",
        {"context": _CTX, "promotion": _PROMO}, ["context"]),
    _fn("effect_table", "카테고리별 혜택 반응표(혜택마다 주문이 몇 % 늘어나는지)",
        {"category": {"type": "string", "enum": list(CATEGORIES)}}, ["category"]),
    _fn("model_validation", "시뮬레이터의 정확도 검증 결과", {}, []),
]


def call_tool(name, args, model, report):
    fn = {"simulate_promotion": simulate_promotion, "compare_promotions": compare_promotions,
          "recommend_promotion": recommend_promotion, "best_addon_timing": best_addon_timing,
          "effect_table": effect_table}
    try:
        if name == "model_validation":
            out = model_validation(report)
        else:
            out = fn[name](model, **args)
    except Exception as e:  # 잘못된 입력은 오류 메시지로 돌려줘서 LLM이 고치게 한다
        out = {"오류": str(e)}
    return json.dumps(out, ensure_ascii=False, default=str)
