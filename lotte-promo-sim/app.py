"""
롯데홈쇼핑 프로모션 시뮬레이터 — Streamlit 앱
실행: streamlit run app.py
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sim.config import CATEGORIES, SLOTS, GIFT_LABEL, DEFAULT_SHARES, ADDON_OPTIONS, promo_label
from sim.economics import evaluate, optimize, all_combos, NO_PROMO
from sim.minute_profile import minute_curve
from sim.tools import krw

st.set_page_config(page_title="홈쇼핑 프로모션 시뮬레이터", page_icon="📺", layout="wide")
RED = "#E60012"


# ── 모델·검증 결과 로드 ───────────────────────────────────────────────
@st.cache_resource(show_spinner="모델을 불러오는 중…")
def load():
    path = "artifacts/model.pkl"
    if not os.path.exists(path):   # 배포 환경에서 모델 파일이 없으면 바로 학습
        from sim.data_gen import generate
        from sim.model import PromoModel
        df = pd.read_csv("data/broadcasts.csv") if os.path.exists("data/broadcasts.csv") \
            else generate(explore_rate=0.1)[0]
        m = PromoModel(n_boot=20).fit(df)   # 첫 실행 시 약 30초
    else:
        with open(path, "rb") as f:
            m = pickle.load(f)
    with open("artifacts/validation.json", encoding="utf-8") as f:
        rep = json.load(f)
    return m, rep


model, report = load()

# ── 사이드바: 방송 조건 + 비용 분담 ───────────────────────────────────
st.sidebar.title("📺 방송 조건")
cat = st.sidebar.selectbox("상품 카테고리", list(CATEGORIES), index=3)
slot = st.sidebar.selectbox("방송 시간대", SLOTS, index=3)
ref = CATEGORIES[cat]["price"]
price = st.sidebar.number_input("판매가(원)", value=ref, step=1000, min_value=int(ref * 0.5))
pop = st.sidebar.slider("상품 인기도 (평균=0)", -2.0, 2.0, 0.0, 0.1)
weekend = st.sidebar.toggle("주말 방송", False)
host = st.sidebar.radio("쇼호스트 등급", ["A", "B"], index=1, horizontal=True)
ctx = dict(category=cat, slot=slot, price=float(price), popularity=pop,
           weekend=int(weekend), host_tier=host)

with st.sidebar.expander("💳 판촉비 분담 구조 (홈쇼핑 몫)"):
    st.caption("나머지는 협력사(카드할인은 카드사와 나눔)가 부담")
    sh_card = st.slider("카드 청구할인 · 홈쇼핑", 0.0, 0.6, DEFAULT_SHARES["card"]["hs"], 0.05)
    sh_cardco = st.slider("카드 청구할인 · 카드사", 0.0, 0.8, DEFAULT_SHARES["card"]["cardco"], 0.05)
    sh_gift = st.slider("사은품 · 홈쇼핑", 0.0, 1.0, DEFAULT_SHARES["gift"]["hs"], 0.1)
    sh_noint = st.slider("무이자 · 홈쇼핑", 0.0, 1.0, DEFAULT_SHARES["noint"]["hs"], 0.1)
    sh_addon = st.slider("추가혜택 · 홈쇼핑", 0.0, 1.0, DEFAULT_SHARES["addon"]["hs"], 0.1)
shares = {
    "card": {"hs": sh_card, "cardco": sh_cardco, "partner": max(0.0, 1 - sh_card - sh_cardco)},
    "gift": {"hs": sh_gift, "partner": 1 - sh_gift},
    "noint": {"hs": sh_noint, "partner": 1 - sh_noint},
    "points": DEFAULT_SHARES["points"],
    "addon": {"hs": sh_addon, "partner": 1 - sh_addon},
}
st.sidebar.caption("⚠️ 가상 데이터로 학습한 프로토타입입니다.")

st.title("홈쇼핑 프로모션 시뮬레이터")
st.caption("어떤 혜택을, 얼마나, 언제 걸어야 '진짜' 이익이 늘어나는가 — "
           "증분 효과 · 당겨쓰기 · 3자 비용 분담 · 방송 중 타이밍까지")

tab_agent, tab_opt, tab_sim, tab_time, tab_eff, tab_val = st.tabs(
    ["🤖 AI 에이전트", "🎯 최적안 찾기", "🧪 직접 시뮬레이션", "⏱️ 추가혜택 타이밍", "📊 혜택 반응표", "✅ 검증"])


# ── 1. AI 에이전트 ───────────────────────────────────────────────────
with tab_agent:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        try:
            key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            key = None
    if not key:
        key = st.text_input("OpenAI API 키", type="password",
                            help="키는 저장되지 않고 이 세션에서만 쓰입니다.")
    st.markdown("**예시 질문**  \n"
                "· 금요일 밤 프라임 시간에 가전 방송인데, 협력사 손해 없이 이익 최대인 혜택 조합 추천해줘  \n"
                "· 뷰티 오전 방송에서 카드 10% vs 사은품 고급형, 뭐가 나아?  \n"
                "· 식품 방송에서 추가혜택은 몇 분에 꺼내는 게 좋아?")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    q = st.chat_input("MD처럼 물어보세요")
    if q:
        if not key:
            st.warning("API 키를 입력해야 에이전트를 쓸 수 있어요. 다른 탭은 키 없이 동작합니다.")
        else:
            from openai import OpenAI
            from sim.agent import run_agent
            st.session_state.chat.append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                with st.spinner("시뮬레이터를 돌리는 중…"):
                    hist = [{"role": m["role"], "content": m["content"]}
                            for m in st.session_state.chat[:-1]][-6:]
                    try:
                        ans, trace, sus = run_agent(q, model, report, history=hist,
                                                    client=OpenAI(api_key=key))
                    except Exception as e:
                        ans, trace, sus = f"오류가 발생했어요: {e}", [], []
                st.markdown(ans)
                if sus:
                    st.warning(f"근거 확인 필요: 답변의 숫자 {', '.join(sus[:5])}는 시뮬레이터 결과에서 찾지 못했어요.")
                else:
                    st.success("근거 검사 통과: 답변의 숫자가 모두 시뮬레이터 결과에서 나왔어요.", icon="✅")
                with st.expander(f"🔧 에이전트가 호출한 도구 {len(trace)}개"):
                    for t in trace:
                        st.markdown(f"**{t['tool']}**")
                        st.json(t["args"], expanded=False)
                        st.json(t["result"], expanded=False)
            st.session_state.chat.append({"role": "assistant", "content": ans})


# ── 2. 최적안 찾기 ───────────────────────────────────────────────────
with tab_opt:
    c1, c2, c3, c4 = st.columns(4)
    objective = c1.radio("목표", ["홈쇼핑 이익", "홈쇼핑+협력사 합계"], horizontal=False)
    floor = c2.number_input("협력사 이익 변화 하한(만원)", value=0, step=100) * 1e4
    budget = c3.number_input("홈쇼핑 판촉비 상한(만원, 0=제한 없음)", value=0, step=500) * 1e4
    safe = c4.toggle("보수 모드", help="협력사 조건을 평균이 아니라 비관적 시나리오(하위 5%)에서도 지키는 조합만")
    top, allres = optimize(model, ctx, objective="hs" if objective == "홈쇼핑 이익" else "joint",
                           partner_floor=floor, hs_budget=budget or None, safe=safe,
                           shares=shares, top_k=5)
    base = allres.attrs["baseline"]
    maxp = allres[(allres.card_pct == 10) & (allres.gift_level == 2) & (allres.no_interest == 1)
                  & (allres.points_pct == 5) & (allres.addon_min == 30)].iloc[0]

    if len(top) == 0:
        st.error("조건을 만족하는 조합이 없어요. 협력사 하한이나 예산을 완화해 보세요.")
    else:
        best = top.iloc[0]
        st.subheader(f"추천: {best['label']}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("예상 주문", f"{best['orders']:,.0f}건", f"{best['lift_pct']:+.1f}% vs 혜택 없음")
        m2.metric("홈쇼핑 이익 변화", krw(best["hs_delta"]),
                  f"90% 구간 {krw(best['hs_delta_lo'])} ~ {krw(best['hs_delta_hi'])}", delta_color="off")
        m3.metric("이익 증가 확률", f"{best['p_better'] * 100:.0f}%")
        m4.metric("협력사 이익 변화", krw(best["partner_delta"]))
        st.info(f"참고: 혜택을 최대로 다 걸면 (카드 10%+사은품 고급형+무이자+적립 5%+30분 추가혜택) "
                f"주문은 {maxp['lift_pct']:+.1f}% 늘지만 홈쇼핑 이익 변화는 **{krw(maxp['hs_delta'])}**, "
                f"당겨쓰기로 방송 후 7일 주문 {maxp['pull_forward_orders']:,.0f}건 감소.")

        show = top[["label", "orders", "lift_pct", "hs_delta", "p_better", "partner_delta",
                    "pull_forward_orders", "hs_cost"]].copy()
        show.columns = ["조합", "예상주문", "주문증가%", "홈쇼핑이익변화(원)", "이익증가확률",
                        "협력사이익변화(원)", "당겨쓰기(건)", "홈쇼핑판촉비(원)"]
        st.dataframe(show.style.format({"예상주문": "{:,.0f}", "주문증가%": "{:+.1f}",
                     "홈쇼핑이익변화(원)": "{:,.0f}", "이익증가확률": "{:.0%}",
                     "협력사이익변화(원)": "{:,.0f}", "당겨쓰기(건)": "{:,.0f}",
                     "홈쇼핑판촉비(원)": "{:,.0f}"}), hide_index=True, width="stretch")

    plot = allres.assign(
        협력사=np.where(allres["partner_delta" + ("_lo" if safe else "")] >= floor, "협력사 수용 가능", "협력사 손해"),
        이익변화_만원=allres["hs_delta"] / 1e4, 매출증가=allres["lift_pct"])
    fig = px.scatter(plot, x="매출증가", y="이익변화_만원", color="협력사", hover_name="label",
                     color_discrete_map={"협력사 수용 가능": RED, "협력사 손해": "#BBBBBB"},
                     labels={"매출증가": "주문 증가율(%)", "이익변화_만원": "홈쇼핑 이익 변화(만원)"},
                     title=f"전체 {len(allres)}개 조합: 주문이 많이 늘어도 이익은 줄 수 있다")
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    if len(top):
        fig.add_scatter(x=[top.iloc[0]["lift_pct"]], y=[top.iloc[0]["hs_delta"] / 1e4], mode="markers",
                        marker=dict(size=18, symbol="star", color="#111"), name="추천")
    st.plotly_chart(fig, width="stretch")


# ── 3. 직접 시뮬레이션 ───────────────────────────────────────────────
with tab_sim:
    st.markdown("두 조합을 같은 방송 조건에서 나란히 비교합니다.")
    cols = st.columns(2)
    promos = []
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"**조합 {'AB'[i]}**")
            p = dict(
                card_pct=st.select_slider("카드 청구할인 %", [0, 5, 7, 10], value=[10, 0][i], key=f"c{i}"),
                gift_level=st.select_slider("사은품", [0, 1, 2], value=[0, 2][i],
                                            format_func=GIFT_LABEL.get, key=f"g{i}"),
                no_interest=int(st.toggle("무이자 할부", value=[True, False][i], key=f"n{i}")),
                points_pct=st.select_slider("적립금 %", [0, 3, 5], value=0, key=f"p{i}"),
                addon_min=st.select_slider("추가혜택 투입(분, 0=없음)", ADDON_OPTIONS, value=0, key=f"a{i}"))
            promos.append(p)
    res = evaluate(model, ctx, promos, shares)
    for i, col in enumerate(cols):
        r = res.iloc[i]
        with col:
            st.metric("예상 주문", f"{r['orders']:,.0f}건", f"{r['lift_pct']:+.1f}%")
            st.metric("홈쇼핑 이익 변화", krw(r["hs_delta"]),
                      f"90% 구간 {krw(r['hs_delta_lo'])} ~ {krw(r['hs_delta_hi'])}", delta_color="off")
            st.caption(f"이익 증가 확률 {r['p_better'] * 100:.0f}% · 협력사 {krw(r['partner_delta'])} · "
                       f"당겨쓰기 {r['pull_forward_orders']:,.0f}건")
    cost = pd.DataFrame({
        "조합": [f"A: {res.iloc[0]['label']}"] * 3 + [f"B: {res.iloc[1]['label']}"] * 3,
        "부담 주체": ["홈쇼핑", "협력사", "카드사"] * 2,
        "판촉비(만원)": [res.iloc[0][k] / 1e4 for k in ["hs_cost", "partner_cost", "cardco_cost"]]
                     + [res.iloc[1][k] / 1e4 for k in ["hs_cost", "partner_cost", "cardco_cost"]]})
    st.plotly_chart(px.bar(cost, x="조합", y="판촉비(만원)", color="부담 주체", barmode="stack",
                           color_discrete_map={"홈쇼핑": RED, "협력사": "#555", "카드사": "#AAA"},
                           title="판촉비는 누가 내는가"), width="stretch")


# ── 4. 추가혜택 타이밍 ──────────────────────────────────────────────
with tab_time:
    st.markdown("방송 중 '지금부터 추가 사은품' 같은 혜택을 **언제 꺼낼지**. "
                "일찍 꺼내면 혜택을 받는 주문이 많아 비용이 커지고, 늦으면 볼 시청자가 줄어든다.")
    base_p = dict(NO_PROMO)
    if len(top):
        base_p = {k: int(top.iloc[0][k]) for k in NO_PROMO}
    base_p["addon_min"] = 0
    st.caption(f"기본 혜택: {promo_label(base_p)} (최적안 탭의 추천 조합 기준)")
    tr = evaluate(model, ctx, [{**base_p, "addon_min": a} for a in ADDON_OPTIONS], shares)
    fig = go.Figure()
    for a, (_, r) in zip(ADDON_OPTIONS, tr.iterrows()):
        curve = minute_curve(a) * r["orders"]
        fig.add_scatter(y=curve, mode="lines", name=f"{a}분 투입" if a else "투입 안 함",
                        line=dict(width=3 if a == tr.sort_values('hs_delta').iloc[-1].addon_min else 1.5))
    fig.update_layout(title="분당 주문 곡선 (60분 방송)", xaxis_title="방송 경과(분)", yaxis_title="분당 주문")
    st.plotly_chart(fig, width="stretch")
    tt = tr[["addon_min", "orders", "hs_delta", "p_better"]].copy()
    tt["addon_min"] = tt["addon_min"].map(lambda a: f"{a}분" if a else "투입 안 함")
    tt.columns = ["투입 시점", "예상 주문", "홈쇼핑 이익 변화(원)", "이익 증가 확률"]
    st.dataframe(tt.style.format({"예상 주문": "{:,.0f}", "홈쇼핑 이익 변화(원)": "{:,.0f}",
                                  "이익 증가 확률": "{:.0%}"}), hide_index=True)
    st.caption("분당 곡선 형태는 과거 방송 로그의 평균 패턴(가상)을 가정했습니다.")


# ── 5. 혜택 반응표 ──────────────────────────────────────────────────
with tab_eff:
    st.markdown(f"**{cat}** 방송에서 혜택 하나를 추가했을 때 주문이 몇 % 늘어나는가 (다른 조건 동일)")
    t = model.glm.coefficients(cat)
    fig = go.Figure(go.Bar(x=t["효과(%)"], y=t["항목"], orientation="h", marker_color=RED,
                           error_x=dict(type="data", symmetric=False, array=t["상한"] - t["효과(%)"],
                                        arrayminus=t["효과(%)"] - t["하한"])))
    fig.update_layout(xaxis_title="주문 효과(%)", yaxis=dict(autorange="reversed"), height=480)
    st.plotly_chart(fig, width="stretch")
    st.caption("'혜택겹침'이 음수 = 혜택을 겹칠수록 하나하나의 효과가 줄어든다는 뜻.")


# ── 6. 검증 ─────────────────────────────────────────────────────────
with tab_val:
    st.markdown("실제 데이터가 없으므로 **정답 효과를 심어둔 가상 데이터**를 만들고, "
                "모델이 그 정답을 얼마나 맞히는지 채점했습니다. 모델은 정답을 보지 못한 상태로 학습합니다.")
    h, e, d = report["holdout"], report["effect_mae"], report["decision"]
    a1, a2, a3 = st.columns(3)
    a1.metric("예측 정확도 (처음 보는 방송)", f"R² {h['r2']:.2f}", f"평균 오차 {h['mape_pct']:.1f}%", delta_color="off")
    a2.metric("혜택 효과 추정 오차", f"{e['model_pp']:.1f}%p",
              f"단순 비교는 {e['naive_pp']:.1f}%p", delta_color="off")
    a3.metric("추천 조합의 이익 확보율", f"{d['normal']['capture_pct']:.0f}%",
              f"혜택 다 걸기: {d['max_promo_capture_pct']:.0f}%", delta_color="off")
    eff = pd.DataFrame(report["effects"])
    long = eff.melt(id_vars=["category", "lever"], value_vars=["true", "model", "naive"],
                    var_name="구분", value_name="효과(%)")
    long["구분"] = long["구분"].map({"true": "정답", "model": "모델 추정", "naive": "단순 비교"})
    long["항목"] = long["category"] + " · " + long["lever"]
    fig = px.bar(long, x="효과(%)", y="항목", color="구분", barmode="group", height=900,
                 color_discrete_map={"정답": "#111", "모델 추정": RED, "단순 비교": "#BBBBBB"},
                 title="혜택 효과: 단순 비교는 '잘 팔릴 방송에 혜택을 몰아준' 효과까지 혜택 덕으로 착각한다")
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")
    st.markdown(f"""
**의사결정 검증** (처음 보는 방송 {d['n_contexts']}개, 정답을 아는 최적안과 비교)
- 일반 모드: 최적안 이익의 **{d['normal']['capture_pct']:.0f}%** 확보, 이익이 오히려 줄어든 방송 {d['normal']['loss_cases_pct']:.0f}%,
  협력사 이익 평균 {d['normal']['partner_change_avg_pct']:+.1f}% (협력사 조건 준수 {d['normal']['partner_ok_pct']:.0f}%)
- 보수 모드: 협력사 조건 준수 **{d['conservative']['partner_ok_pct']:.0f}%**, 대신 이익 확보 {d['conservative']['capture_pct']:.0f}%
- 혜택 최대로 다 걸기: 최적안 대비 **{d['max_promo_capture_pct']:.0f}%** (이익이 오히려 감소)

**한계**: 가상 데이터 검증이며, 실제 데이터에서는 관측되지 않는 요인(경쟁사 동시간대 방송 등)이 있을 수 있어
탐색 편성(일부 방송의 혜택 무작위 편성)으로 지속 검증이 필요합니다.
""")
