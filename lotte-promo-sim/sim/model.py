"""
반응 예측 모델 (두 모델의 앙상블).

예측 대상
1) 주문 수: 방송 조건 + 프로모션 → 주문 수
2) 방송 후 7일 앱 주문 수 (당겨쓰기 측정용)

두 모델
- GBM(그래디언트 부스팅): 시간대·인기도·가격 같은 복잡한 비선형 패턴을 잘 잡는다.
- GLM(카테고리별 포아송 회귀): 혜택별 효과를 '계수'로 보여줘 MD가 해석할 수 있다.
- 최종 예측 = 두 모델의 기하평균. 검증에서 각각 따로 쓸 때보다 정확했다.

핵심 아이디어
- '프로모션 효과' = 같은 방송 조건에서 [혜택 있음 예측] - [혜택 없음 예측]
  (시간대·상품 인기도 같은 조건을 모델이 함께 보기 때문에, 단순 비교의 교란을 걸러낸다)
- 단조성 제약: "혜택을 키웠더니 주문이 줄었다" 같은 비상식적 예측을 구조적으로 막는다.
- 부트스트랩 앙상블 20개: 숫자 하나가 아니라 '범위(신뢰구간)'로 답한다.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import CATEGORIES, SLOTS, ADDON_OPTIONS

CAT_CODE = {c: i for i, c in enumerate(CATEGORIES)}
SLOT_CODE = {s: i for i, s in enumerate(SLOTS)}
ADDON_CODE = {a: i for i, a in enumerate(ADDON_OPTIONS)}

FEATURES = ["category", "slot", "addon", "log_price_rel", "popularity", "weekend", "host_a",
            "disc_pct", "gift_level", "noint_months", "points_pct"]
IS_CATEGORICAL = [True, True, True] + [False] * 8


def featurize(df: pd.DataFrame) -> np.ndarray:
    ref = df["category"].map(lambda c: CATEGORIES[c]["price"])
    X = pd.DataFrame({
        "category": df["category"].map(CAT_CODE),
        "slot": df["slot"].map(SLOT_CODE),
        "addon": df["addon_min"].map(ADDON_CODE),
        "log_price_rel": np.log(df["price"] / ref),
        "popularity": df["popularity"],
        "weekend": df["weekend"],
        "host_a": (df["host_tier"] == "A").astype(int),
        "disc_pct": df["disc_pct"],
        "gift_level": df["gift_level"],
        "noint_months": df["noint_months"],
        "points_pct": df["points_pct"],
    })
    return X[FEATURES].to_numpy(dtype=float)


def _model(direction: int, seed: int):
    # 프로모션 4개 변수에 단조성 제약: 주문 모델은 +1(늘거나 같음), 사후 7일 모델은 -1
    mono = [0, 0, 0, 0, 0, 0, 0, direction, direction, direction, direction]
    return HistGradientBoostingRegressor(
        loss="poisson", max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=30, l2_regularization=1.0, categorical_features=IS_CATEGORICAL,
        monotonic_cst=mono, random_state=seed)


class GBMModel:
    def __init__(self, n_boot: int = 20, seed: int = 0):
        self.n_boot, self.seed = n_boot, seed
        self.orders_models, self.post_models = [], []

    def fit(self, df: pd.DataFrame):
        X = featurize(df)
        y, yp = df["orders"].to_numpy(), df["post7_orders"].to_numpy()
        rng = np.random.default_rng(self.seed)
        for b in range(self.n_boot):
            idx = rng.integers(0, len(df), len(df))       # 부트스트랩 재표본
            self.orders_models.append(_model(+1, b).fit(X[idx], y[idx]))
            self.post_models.append(_model(-1, b).fit(X[idx], yp[idx]))
        return self

    def predict_all(self, df: pd.DataFrame):
        """각 앙상블 멤버의 예측 (행 = 멤버, 열 = 입력 행)."""
        X = featurize(df)
        o = np.vstack([m.predict(X) for m in self.orders_models])
        p = np.vstack([m.predict(X) for m in self.post_models])
        return o, p

    def predict(self, df: pd.DataFrame):
        o, p = self.predict_all(df)
        return o.mean(0), p.mean(0)


# ── GLM: 카테고리별 포아송 회귀 (해석 가능한 효과 계수) ────────────────────
from sklearn.linear_model import PoissonRegressor
from .config import SLOTS as _SLOTS


def glm_features(d: pd.DataFrame) -> pd.DataFrame:
    ref = d["category"].map(lambda c: CATEGORIES[c]["price"]).to_numpy()
    o = pd.DataFrame({"인기도": d["popularity"].to_numpy(), "주말": d["weekend"].to_numpy() * 1.,
                      "A급호스트": (d["host_tier"].to_numpy() == "A") * 1.,
                      "가격(상대)": np.log(d["price"].to_numpy() / ref)})
    for s in _SLOTS[1:]:
        o[s] = (d["slot"].to_numpy() == s) * 1.
    for v in [5, 10, 15]:
        o[f"즉시할인{v}%"] = (d["disc_pct"].to_numpy() == v) * 1.
    for v, name in [(1, "조건부 증정"), (2, "전고객 증정·구성추가")]:
        o[name] = (d["gift_level"].to_numpy() == v) * 1.
    for v in [3, 10, 12]:
        o[f"무이자{v}개월"] = (d["noint_months"].to_numpy() == v) * 1.
    for v in [1, 3]:
        o[f"추가적립{v}%"] = (d["points_pct"].to_numpy() == v) * 1.
    for v in [15, 30, 45]:
        o[f"추가혜택{v}분"] = (d["addon_min"].to_numpy() == v) * 1.
    k = ((d["disc_pct"].to_numpy() > 0) * 1 + (d["gift_level"].to_numpy() > 0)
         + (d["noint_months"].to_numpy() > 0) + (d["points_pct"].to_numpy() > 0)
         + (d["addon_min"].to_numpy() > 0))
    o["혜택겹침"] = np.maximum(k - 1, 0) * 1.     # 혜택을 겹칠수록 효과가 줄어드는 정도
    return o


class GLMModel:
    def __init__(self, n_boot: int = 20, seed: int = 0):
        self.n_boot, self.seed, self.models = n_boot, seed, {}

    def fit(self, df: pd.DataFrame):
        rng = np.random.default_rng(self.seed)
        for cat, sub in df.groupby("category"):
            X = glm_features(sub)
            y, yp = sub["orders"].to_numpy(), sub["post7_orders"].to_numpy()
            ms = []
            for _ in range(self.n_boot):
                i = rng.integers(0, len(sub), len(sub))
                ms.append((PoissonRegressor(alpha=1e-5, max_iter=2000).fit(X.iloc[i], y[i]),
                           PoissonRegressor(alpha=1e-5, max_iter=2000).fit(X.iloc[i], yp[i])))
            self.models[cat] = ms
        return self

    def predict_all(self, df: pd.DataFrame):
        df = df.reset_index(drop=True)
        o = np.zeros((self.n_boot, len(df)))
        p = np.zeros_like(o)
        for cat, sub in df.groupby("category"):
            X = glm_features(sub)
            for b, (mo, mp) in enumerate(self.models[cat]):
                o[b, sub.index] = mo.predict(X)
                p[b, sub.index] = mp.predict(X)
        return o, p

    def coefficients(self, category: str) -> pd.DataFrame:
        """혜택별 효과(%)와 90% 구간 — MD에게 보여줄 '혜택 반응표'."""
        names = glm_features(pd.DataFrame([dict(category=category, price=1, popularity=0,
                    weekend=0, host_tier="B", slot=_SLOTS[0], disc_pct=0, gift_level=0,
                    noint_months=0, points_pct=0, addon_min=0)])).columns
        C = np.array([mo.coef_ for mo, _ in self.models[category]])
        eff = (np.exp(C) - 1) * 100
        out = pd.DataFrame({"항목": names, "효과(%)": eff.mean(0),
                            "하한": np.percentile(eff, 5, 0), "상한": np.percentile(eff, 95, 0)})
        promo_terms = [n for n in names if any(k in n for k in ["할인", "증정", "무이자", "적립", "추가혜택", "겹침"])]
        return out[out["항목"].isin(promo_terms)].reset_index(drop=True)


class PromoModel:
    """최종 모델: GBM과 GLM의 기하평균 앙상블."""

    def __init__(self, n_boot: int = 20, seed: int = 0):
        self.gbm, self.glm = GBMModel(n_boot, seed), GLMModel(n_boot, seed)

    def fit(self, df: pd.DataFrame):
        self.gbm.fit(df)
        self.glm.fit(df)
        return self

    def predict_all(self, df: pd.DataFrame):
        o1, p1 = self.gbm.predict_all(df)
        o2, p2 = self.glm.predict_all(df)
        return np.sqrt(o1 * o2), np.sqrt(np.maximum(p1, 1e-6) * np.maximum(p2, 1e-6))

    def predict(self, df: pd.DataFrame):
        o, p = self.predict_all(df)
        return o.mean(0), p.mean(0)
