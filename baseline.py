import os
import cv2
import pandas as pd
from ultralytics import YOLO


DEFAULT_LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")


OBJECT_CLASSES = [
    # 문구 / 학습
    "book", "notebook", "paper", "document", "pen", "pencil",
    "marker", "scissors", "ruler", "eraser",

    # 전자기기
    "laptop", "keyboard", "mouse", "monitor", "phone",
    "tablet", "charger", "cable", "earphones", "remote control",

    # 식기 / 음식
    "cup", "bottle", "mug", "plate", "bowl", "spoon",
    "fork", "snack", "food wrapper", 

    # 의류 / 잡화
    "clothes", "shirt", "hoodie", "pants", "socks",
    "bag", "backpack", "hat", "shoe",

    # 화장품 / 생활용품
    "cosmetics", "tissue", "medicine", "toothbrush",
    "comb", "mirror",

    # 쓰레기 / 기타
    "trash", "box", "plastic bag", "toy"
]



CATEGORY_MAP = {
    "study_supplies": {
        "items": [
            "book", "notebook", "paper", "document", "pen", "pencil",
            "marker", "scissors", "ruler", "eraser"
        ],
        "korean_name": "학습/문구류",
        "recommendation": "책은 책꽂이에 세우고, 종이는 파일철에 모으고, 필기구는 필통이나 문구함에 넣으세요.",
    },
    "electronics": {
        "items": [
            "laptop", "keyboard", "mouse", "monitor", "phone",
            "tablet", "charger", "cable", "earphones", "remote control"
        ],
        "korean_name": "전자기기/케이블",
        "recommendation": "케이블과 충전기는 한 박스에 모으고, 자주 쓰는 전자기기는 책상 한쪽에 고정 위치를 정하세요.",
    },
    "kitchen_items": {
        "items": [
            "cup", "bottle", "mug", "plate", "bowl", "spoon",
            "fork", "snack", "food wrapper"
        ],
        "korean_name": "식기/음식 관련",
        "recommendation": "컵과 식기는 주방으로 옮기고, 음식 포장지는 바로 버리세요.",
    },
    "clothing": {
        "items": [
            "clothes", "shirt", "hoodie", "pants", "socks",
            "bag", "backpack", "hat", "shoe"
        ],
        "korean_name": "의류/잡화",
        "recommendation": "입을 옷과 세탁할 옷을 분리하고, 가방과 모자는 옷장이나 행거에 걸어두세요.",
    },
    "daily_supplies": {
        "items": [
            "cosmetics", "tissue", "medicine", "toothbrush",
            "comb", "mirror"
        ],
        "korean_name": "생활용품",
        "recommendation": "화장품과 생활용품은 작은 바구니나 서랍에 모아 보관하세요.",
    },
    "trash_or_misc": {
        "items": [
            "trash", "box", "plastic bag", "toy"
        ],
        "korean_name": "쓰레기/기타",
        "recommendation": "버릴 것은 먼저 제거하고, 박스나 비닐봉지는 접어서 따로 모으세요.",
    }
}


def classify_item(item_name: str):
    for category_key, info in CATEGORY_MAP.items():
        if item_name in info["items"]:
            return category_key, info["korean_name"], info["recommendation"]

    return "unknown", "미분류", "정리 위치를 사용자가 직접 지정해야 합니다."


def detect_objects(image_path: str, output_dir: str = "outputs", conf: float = 0.08):
    os.makedirs(output_dir, exist_ok=True)

    # YOLO-World 모델 로드
    # 더 빠름: yolov8s-world.pt
    # 더 정확함: yolov8m-world.pt 또는 yolov8l-world.pt
    model = YOLO("yolov8s-world.pt")

    # 생활 물건 후보 class 지정
    # 배경 class ""를 추가하면 일부 상황에서 탐지가 안정될 수 있다고 문서에 언급되어 있음
    model.set_classes(OBJECT_CLASSES + [""])

    # 추론
    results = model.predict(
        source=image_path,
        conf=conf,
        imgsz=960,
        verbose=False
    )

    result = results[0]
    image = cv2.imread(image_path)

    detections = []

    if result.boxes is None:
        return pd.DataFrame(), None, None

    for box in result.boxes:
        cls_id = int(box.cls[0])
        score = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        item_name = result.names[cls_id]

        # background class 제거
        if item_name == "":
            continue

        category_key, category_ko, recommendation = classify_item(item_name)

        detections.append({
            "object": item_name,
            "confidence": round(score, 3),
            "category": category_key,
            "category_ko": category_ko,
            "recommendation": recommendation,
            "bbox": [x1, y1, x2, y2]
        })

        # 이미지에 bbox 표시
        label = f"{item_name} {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

    df = pd.DataFrame(detections)

    # 결과 저장
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_image_path = os.path.join(output_dir, f"{base_name}_detected.jpg")
    output_csv_path = os.path.join(output_dir, f"{base_name}_objects.csv")
    output_prompt_path = os.path.join(output_dir, f"{base_name}_llm_prompt.txt")
    output_recommendation_path = os.path.join(output_dir, f"{base_name}_recommendation.txt")

    cv2.imwrite(output_image_path, image)

    if not df.empty:
        df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
        llm_prompt = build_llm_prompt(df)
        with open(output_prompt_path, "w", encoding="utf-8") as f:
            f.write(llm_prompt)
    else:
        output_csv_path = None
        output_prompt_path = None
        output_recommendation_path = None

    return df, output_image_path, output_prompt_path, output_recommendation_path



# 분류 결과 LLM에 입력
def build_llm_prompt(df: pd.DataFrame) -> str:

    grouped = df.groupby("category_ko")["object"].apply(list).to_dict()

    prompt = """
너는 정리 컨설턴트 역할을 한다.
사용자가 어질러진 책상/방/책꽂이/옷장 사진을 찍었고, 객체 탐지 모델이 아래 물건들을 감지했다.

목표:
1. 감지된 물건들을 정리 카테고리별로 묶어라.
2. 사용자가 바로 실행할 수 있는 정리 순서를 제안하라.
3. 5분 안에 할 수 있는 최소 정리 루틴을 제안하라.
4. 버릴 것, 옮길 것, 제자리에 둘 것을 나누어라.

탐지된 객체 목록:
"""

    for category, items in grouped.items():
        unique_items = sorted(set(items))
        prompt += f"\n[{category}]\n"
        for item in unique_items:
            count = items.count(item)
            prompt += f"- {item}: {count}개\n"

    prompt += """

출력 형식:
- 전체 상태 요약
- 카테고리별 정리 방법
- 우선순위 TOP 3
- 5분 정리 루틴
- 다음부터 어지럽히지 않기 위한 수납 규칙
"""

    return prompt.strip()


def ask_openai_for_recommendation(prompt: str, model: str = DEFAULT_LLM_MODEL) -> str:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY 환경변수가 없습니다. "
            "예: export OPENAI_API_KEY='your_api_key_here'"
        )

    from openai import OpenAI

    client = OpenAI()
    response = client.responses.create(
        model=model,
        instructions=(
            "너는 한국어로 답하는 실내 정리 컨설턴트다. "
            "객체 탐지 결과에는 오탐이 섞일 수 있으므로, 확신하지 못하는 물건은 단정하지 말고 "
            "사용자가 바로 실행할 수 있는 짧고 구체적인 순서로 제안한다."
        ),
        input=prompt,
    )

    return response.output_text.strip()


def save_llm_recommendation(
    prompt_path: str | None,
    output_path: str | None,
    model: str = DEFAULT_LLM_MODEL
):
    if prompt_path is None or output_path is None:
        return None

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()

    recommendation = ask_openai_for_recommendation(prompt, model=model)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(recommendation)

    return output_path


def summarize_by_category(df: pd.DataFrame):
    """터미널에 간단 요약 출력"""
    if df.empty:
        print("탐지된 객체가 없습니다.")
        return

    print("\n탐지된 객체 목록")
    print(df[["object", "confidence", "category_ko"]])

    print("\n카테고리별 요약")
    summary = df.groupby("category_ko")["object"].apply(list)

    for category, items in summary.items():
        print(f"\n[{category}]")
        for item in sorted(set(items)):
            print(f"- {item}: {items.count(item)}개")


if __name__ == "__main__":
    image_path = "test_images/test1.jpeg"

    df, output_image, output_prompt, output_recommendation = detect_objects(
        image_path=image_path,
        output_dir="outputs",
        conf=0.08
    )

    summarize_by_category(df)

    try:
        output_recommendation = save_llm_recommendation(
            prompt_path=output_prompt,
            output_path=output_recommendation,
        )
    except RuntimeError as exc:
        print(f"\nLLM 추천 생성 건너뜀: {exc}")
        output_recommendation = None

    print("\n저장 완료")
    print(f"- 탐지 결과 이미지: {output_image}")
    print(f"- LLM 프롬프트 파일: {output_prompt}")
    print(f"- LLM 정리 추천 파일: {output_recommendation}")
