"""
LLM 프롬프트 빌더 + OpenAI 호출.

v2.md §3 공정성 원칙:
  OFF와 ON의 유일한 차이는 description + state 두 줄.
  instruction · 출력형식 · 모델 · temperature 는 양쪽 100% 동일.
"""

import json
import os
from collections import defaultdict

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_SYSTEM_INSTRUCTION = """너는 실내 정리 컨설턴트다. 아래 탐지된 물건 각각에 대해
discard / relocate / keep 중 하나를 판정하라.
판정 기준:
- discard: 재사용 의도 없는 빈 용기·포장지·소모품
- relocate: 유효하지만 현재 위치가 부적절한 물건
- keep: 현재 위치가 적절한 물건
확신이 낮으면 keep으로 둔다."""

_OUTPUT_SCHEMA = """
반드시 아래 JSON만 출력하라 (다른 텍스트 없이):
{
  "decisions": [
    {"object_id": <id(int)>, "label": "<label>", "decision": "<discard|relocate|keep>"}
  ],
  "guide": {
    "priority_top3": ["<action1>", "<action2>", "<action3>"],
    "five_min_routine": ["<step1>", "<step2>", "<step3>"],
    "by_category": {"<category_ko>": "<guidance>"}
  }
}"""


def _format_object_line(obj: dict, include_describe: bool) -> str:
    """객체 한 줄 포맷. include_describe=True면 description+state 추가."""
    line = f"  - {obj['label']} (conf {obj['confidence']:.2f}) [id={obj['id']}]"
    if include_describe and obj.get("description"):
        line += f"\n      · description: \"{obj['description']}\""
        if obj.get("state"):
            line += f"\n      · state: {obj['state']}"
    return line


def build_prompt(objects: list[dict], describe: bool) -> str:
    """
    v2.md §3 포맷으로 LLM 프롬프트 생성.
    describe=False → OFF(control), describe=True → ON(treatment).
    """
    # 카테고리별 그룹핑
    groups: dict[str, list[dict]] = defaultdict(list)
    for obj in objects:
        groups[obj.get("category_ko", "미분류")].append(obj)

    lines = [_SYSTEM_INSTRUCTION, "\n탐지된 물건 목록:\n"]
    for cat_ko, objs in groups.items():
        lines.append(f"[{cat_ko}]")
        for obj in objs:
            lines.append(_format_object_line(obj, include_describe=describe))

    lines.append(_OUTPUT_SCHEMA)
    return "\n".join(lines)


def ask_llm(prompt: str) -> dict | None:
    """
    OpenAI API 호출 → {"decisions": [...], "guide": {...}} 반환.
    OPENAI_API_KEY 없으면 None 반환 (파이프라인은 계속 동작).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None
