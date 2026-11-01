"""
프로모션 기획 AI 에이전트 (OpenAI 함수 호출).

흐름: MD 질문 → LLM이 필요한 도구 선택 → 시뮬레이터 실행 → 결과를 근거와 함께 설명

안정성 장치 3가지
1) 숫자는 도구에서만: 시스템 프롬프트로 강제 + 답변 속 숫자가 도구 결과에 실제로 있는지 자동 검사
2) 입력 검증: 카테고리·시간대·혜택 값은 정해진 선택지(enum)만 허용, 잘못되면 오류를 돌려줘 LLM이 스스로 고침
3) 무한 반복 방지: 도구 호출은 최대 6회
"""
import json
import os
import re

from .tools import TOOL_SCHEMAS, call_tool

SYSTEM = """당신은 롯데홈쇼핑 MD를 돕는 '프로모션 기획 에이전트'입니다.
규칙:
- 숫자(주문 수, 금액, %)는 반드시 도구 결과에 있는 값만 그대로 인용합니다. 직접 계산하거나 추측하지 않습니다.
- 추천할 때는 (1) 추천 조합 (2) 근거: 주문 증가, 홈쇼핑 이익 변화와 90% 구간, 이익 증가 확률
  (3) 협력사 영향 (4) 리스크 순으로 짧게 답합니다.
- 방송 조건이 빠졌으면 합리적 기본값(오후 시간대, 대표 가격, B급 호스트)을 쓰고 그 사실을 밝힙니다.
- '혜택을 다 거는 것'과 비교해 차이를 설명하면 MD 설득에 도움이 됩니다.
- 이 시뮬레이터는 가상 데이터로 학습된 프로토타입이라는 점을 필요할 때 밝힙니다.
- 한국어로, 표 대신 짧은 문단과 목록으로 답합니다."""

MAX_STEPS = 6
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")


def _numbers(text):
    """'1,234', '12.5', '3.4억' 같은 숫자 토큰을 정규화해서 추출."""
    return {n.replace(",", "") for n in re.findall(r"\d[\d,]*\.?\d*", text)}


def grounding_check(answer, tool_outputs):
    """답변에 나온 숫자 중 도구 결과에 없는 것(= LLM이 지어냈을 수 있는 숫자)을 찾는다."""
    allowed = set()
    for t in tool_outputs:
        allowed |= _numbers(t)
    suspicious = [n for n in _numbers(answer)
                  if n not in allowed and len(n.replace(".", "")) >= 3]   # 1~2자리 숫자(순번 등)는 제외
    return suspicious


def run_agent(question, sim_model, report, history=None, client=None, model_name=None):
    """
    반환: (답변, 도구 호출 기록, 근거 없는 숫자 목록)
    """
    if client is None:
        from openai import OpenAI
        client = OpenAI()
    msgs = [{"role": "system", "content": SYSTEM}] + (history or []) + [
        {"role": "user", "content": question}]
    trace, outputs = [], []
    for _ in range(MAX_STEPS):
        resp = client.chat.completions.create(model=model_name or DEFAULT_MODEL, messages=msgs,
                                              tools=TOOL_SCHEMAS, temperature=0.2)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            answer = msg.content or ""
            return answer, trace, grounding_check(answer, outputs)
        msgs.append({"role": "assistant", "content": msg.content or "",
                     "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            out = call_tool(tc.function.name, args, sim_model, report)
            trace.append({"tool": tc.function.name, "args": args, "result": json.loads(out)})
            outputs.append(out)
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": out})
    return "도구 호출 한도(6회)에 도달했습니다. 질문을 나눠서 다시 물어봐 주세요.", trace, []
