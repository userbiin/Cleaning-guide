"""
describe 문자열 → state_signal 파싱.
describe=OFF 경로에서는 호출되지 않는다 (None 반환).
describe=ON 경로(P3)에서 Florence-2 output을 이 함수에 통과시킨다.
"""

STATE_KEYWORDS: dict[str, list[str]] = {
    "trash_signal":  ["empty", "crumpled", "crushed", "used", "torn",
                      "wrapper", "trash", "garbage", "waste", "discarded",
                      "disposable", "expired"],
    "needs_wash":    ["dirty", "stained", "half-full", "leftover",
                      "greasy", "soiled"],
    "tidy_target":   ["stacked", "folded", "closed", "arranged", "organized"],
    "in_use":        ["charging", "open", "running", "active"],
}


def parse_state_signal(description: str | None) -> str | None:
    """
    describe 문자열에서 첫 번째로 매칭된 state_signal 키를 반환.
    매칭 없거나 description이 None이면 None.
    """
    if not description:
        return None
    desc_lower = description.lower()
    for signal, keywords in STATE_KEYWORDS.items():
        if any(kw in desc_lower for kw in keywords):
            return signal
    return None
