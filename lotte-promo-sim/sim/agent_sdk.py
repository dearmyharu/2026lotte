"""
프로모션 기획 에이전트 — OpenAI Agents SDK 버전.

기존 agent.py(함수 호출 루프를 직접 구현)와 같은 도구·규칙을 쓰되,
대회 개발 환경인 OpenAI Agents SDK의 구성 요소로 옮겼다.

- Agent: 시스템 지침 + 도구 7개
- FunctionTool: 시뮬레이터 함수를 도구로 등록 (입력 스키마는 tools.TOOL_SCHEMAS 재사용)
- Output guardrail: 답변 속 숫자가 도구 결과에 실제로 있는지 검사(근거 검사)
- Runner: max_turns로 무한 반복 방지
"""
import asyncio
import json
import os
from dataclasses import dataclass, field

from agents import (Agent, FunctionTool, GuardrailFunctionOutput, ModelSettings, RunContextWrapper,
                    Runner, output_guardrail)

from .agent import SYSTEM, grounding_check
from .tools import TOOL_SCHEMAS, call_tool

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
MAX_TURNS = 8      # 도구 호출 6회 + 최종 답변 여유


@dataclass
class SimContext:
    """도구가 공유하는 실행 맥락: 학습된 시뮬레이터, 검증 결과, 도구 호출 기록."""
    sim_model: object
    report: dict
    trace: list = field(default_factory=list)


def _make_tool(schema):
    fn = schema["function"]
    name = fn["name"]

    async def on_invoke(ctx: RunContextWrapper[SimContext], args_json: str) -> str:
        args = json.loads(args_json or "{}")
        out = call_tool(name, args, ctx.context.sim_model, ctx.context.report)
        ctx.context.trace.append({"tool": name, "args": args, "result": json.loads(out)})
        return out

    return FunctionTool(name=name, description=fn["description"],
                        params_json_schema=fn["parameters"], on_invoke_tool=on_invoke,
                        strict_json_schema=False)


@output_guardrail
async def number_grounding(ctx: RunContextWrapper[SimContext], agent: Agent, output: str):
    """근거 없는 숫자를 찾아 기록한다. 답변을 막지는 않고 화면에 경고로 보여준다."""
    outputs = [json.dumps(t["result"], ensure_ascii=False) for t in ctx.context.trace]
    suspicious = grounding_check(str(output), outputs)
    return GuardrailFunctionOutput(output_info={"suspicious_numbers": suspicious},
                                   tripwire_triggered=False)


def build_agent(model=None):
    return Agent[SimContext](
        name="프로모션 기획 에이전트",
        instructions=SYSTEM,
        tools=[_make_tool(s) for s in TOOL_SCHEMAS],
        model=model or DEFAULT_MODEL,
        model_settings=ModelSettings(temperature=0.2),
        output_guardrails=[number_grounding],
    )


async def run_agent_sdk_async(question, sim_model, report, history=None, model=None):
    ctx = SimContext(sim_model=sim_model, report=report)
    items = [{"role": m["role"], "content": m["content"]} for m in (history or [])]
    items.append({"role": "user", "content": question})
    result = await Runner.run(build_agent(model), items, context=ctx, max_turns=MAX_TURNS)
    suspicious = []
    for g in result.output_guardrail_results:
        suspicious += g.output.output_info.get("suspicious_numbers", [])
    return str(result.final_output), ctx.trace, suspicious


def run_agent_sdk(question, sim_model, report, history=None, model=None):
    """Streamlit 등 동기 코드에서 쓰는 진입점. 반환: (답변, 도구 호출 기록, 근거 없는 숫자)"""
    return asyncio.run(run_agent_sdk_async(question, sim_model, report, history, model))
