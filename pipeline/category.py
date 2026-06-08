"""객체 라벨 → 카테고리 매핑 (baseline.py에서 이식, 시각화 분리)."""

CATEGORY_MAP: dict[str, dict] = {
    "study_supplies": {
        "items": [
            "book", "notebook", "paper", "document", "pen", "pencil",
            "marker", "scissors", "ruler", "eraser",
        ],
        "korean_name": "학습/문구류",
        "recommendation": "책은 책꽂이에 세우고, 종이는 파일철에 모으고, 필기구는 필통이나 문구함에 넣으세요.",
    },
    "electronics": {
        "items": [
            "laptop", "keyboard", "mouse", "monitor", "phone",
            "tablet", "charger", "cable", "earphones", "remote control",
        ],
        "korean_name": "전자기기/케이블",
        "recommendation": "케이블과 충전기는 한 박스에 모으고, 자주 쓰는 전자기기는 책상 한쪽에 고정 위치를 정하세요.",
    },
    "kitchen_items": {
        "items": [
            "cup", "bottle", "mug", "plate", "bowl", "spoon",
            "fork", "snack", "food wrapper",
        ],
        "korean_name": "식기/음식 관련",
        "recommendation": "컵과 식기는 주방으로 옮기고, 음식 포장지는 바로 버리세요.",
    },
    "clothing": {
        "items": [
            "clothes", "shirt", "hoodie", "pants", "socks",
            "bag", "backpack", "hat", "shoe",
        ],
        "korean_name": "의류/잡화",
        "recommendation": "입을 옷과 세탁할 옷을 분리하고, 가방과 모자는 옷장이나 행거에 걸어두세요.",
    },
    "daily_supplies": {
        "items": [
            "cosmetics", "tissue", "medicine", "toothbrush", "comb", "mirror",
        ],
        "korean_name": "생활용품",
        "recommendation": "화장품과 생활용품은 작은 바구니나 서랍에 모아 보관하세요.",
    },
    "trash_or_misc": {
        "items": [
            "trash", "box", "plastic bag", "toy",
        ],
        "korean_name": "쓰레기/기타",
        "recommendation": "버릴 것은 먼저 제거하고, 박스나 비닐봉지는 접어서 따로 모으세요.",
    },
}


def classify_item(label: str) -> tuple[str, str, str]:
    """label → (category_key, category_ko, recommendation)."""
    for key, info in CATEGORY_MAP.items():
        if label in info["items"]:
            return key, info["korean_name"], info["recommendation"]
    return "unknown", "미분류", "정리 위치를 사용자가 직접 지정해야 합니다."
