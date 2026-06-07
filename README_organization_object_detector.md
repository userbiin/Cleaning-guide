# Organization Object Detector

> 어질러진 테이블, 책상, 책꽂이, 옷장 등을 사진으로 찍으면 객체를 탐지하고, 탐지된 물건들을 정리 카테고리별로 분류한 뒤, LLM을 통해 정리 방법을 제안하는 시스템입니다.

## 1. Project Summary

이 프로젝트의 목표는 **정리가 어려운 사용자를 위한 객체 탐지 기반 정리 보조 모델**을 만드는 것입니다.

사용자가 어질러진 공간의 사진을 입력하면 시스템은 다음 순서로 동작합니다.

```text
[사용자 사진 입력]
        ↓
OpenCV 전처리
        ↓
Object Detection
        ↓
객체 리스트 생성
        ↓
정리 카테고리 분류
        ↓
LLM 정리 추천
        ↓
카테고리별 정리 방법 출력
```

예시 입력 장면:

- 어질러진 책상
- 테이블 위 잡동사니
- 책꽂이
- 옷장
- 화장대
- 침대 주변
- 주방 조리대
- 현관 신발장

예시 탐지 객체:

```text
book, notebook, pen, cup, bottle, laptop, charger, cable, hoodie, socks, tissue, snack bag
```

예시 정리 카테고리:

| 탐지 객체 | 정리 카테고리 | 추천 위치 |
|---|---|---|
| book, notebook | 학습/서류 | 책꽂이, 책상 서랍 |
| pen, pencil | 문구류 | 필통, 문구함 |
| charger, cable | 전자기기 액세서리 | 케이블 박스 |
| cup, bottle | 식기/음료 | 주방, 세척 공간 |
| hoodie, socks | 의류 | 옷장, 세탁 바구니 |
| snack bag, tissue | 쓰레기/소모품 | 쓰레기통 |

---

## 2. Current Baseline Direction

현재 baseline은 **학습 없이 zero-shot/open-vocabulary detection을 먼저 적용**하는 방식으로 설계합니다.

초기 baseline 구조:

```text
사진 1장 입력
    ↓
YOLO-World로 생활 물건 탐지
    ↓
객체 리스트 생성
    ↓
rule-based category mapping
    ↓
LLM prompt 생성
    ↓
정리 추천 출력
```

초기 목표는 높은 정확도보다 다음을 빠르게 확인하는 것입니다.

1. 실제 어질러진 공간 사진에서 어떤 물체가 탐지되는가
2. 어떤 물체가 자주 누락되는가
3. 어떤 class를 fine-tuning해야 하는가
4. LLM에 전달할 객체 리스트 포맷이 적절한가

---

## 3. Recommended Models

### 3.1 YOLO-World

초기 baseline에 가장 적합한 모델입니다.

장점:

- custom class prompt를 지정할 수 있음
- 일반 YOLO처럼 사용하기 쉬움
- OpenCV와 실시간 데모 연결이 쉬움
- fine-tuning 이전 baseline으로 적합

사용 방식:

```python
from ultralytics import YOLO

model = YOLO("yolov8s-world.pt")
model.set_classes([
    "book", "notebook", "paper", "pen", "cup", "bottle",
    "laptop", "charger", "cable", "phone", "clothes", "socks",
    "bag", "tissue", "trash"
])
```

### 3.2 YOLOE

프로젝트 고도화 단계에서 추천하는 모델입니다.

장점:

- real-time open-vocabulary detection/segmentation
- 텍스트/이미지 프롬프트 기반 탐지 가능
- custom dataset fine-tuning에 적합
- 캡스톤/논문형 프로젝트로 최신성이 좋음

### 3.3 Grounding DINO

텍스트 기반 탐지에 강한 모델입니다.

장점:

- open-set object detection에 적합
- “person wearing helmet”, “plastic bottle”, “messy cable”처럼 문장형 prompt 가능
- SAM 2와 결합하기 좋음

단점:

- 실시간성은 YOLO 계열보다 떨어질 수 있음

### 3.4 SAM 2

탐지된 객체를 mask 단위로 정교하게 분리할 때 사용합니다.

역할:

- YOLO/ Grounding DINO가 찾은 bbox를 SAM 2에 prompt로 전달
- 겹쳐 있는 물건의 segmentation mask 생성
- 책상 위 cluttered object 분리

### 3.5 Depth Anything V2

선택적 확장 모델입니다.

역할:

- 사진/영상에서 depth map 생성
- 물건이 앞쪽/뒤쪽/바닥/책상 위에 있는지 공간 감각 보조
- “먼저 치워야 하는 앞쪽 물건” 같은 우선순위 추론에 활용 가능

---

## 4. Dataset Candidates

이 프로젝트에 딱 맞는 단일 공개 데이터셋은 거의 없습니다. 따라서 다음처럼 조합하는 것이 현실적입니다.

```text
Open Images V7 / Objects365 / LVIS
        ↓
일반 생활 물체 인식 기반 확보

OCID / ESD / HOTS
        ↓
cluttered indoor scene 적응

직접 수집한 책상/책꽂이/옷장 데이터
        ↓
fine-tuning 및 실제 사용 환경 최적화
```

### 4.1 Open Images V7

용도:

- 일반 생활 물체 탐지 기반 데이터셋
- 약 600개 object class
- bounding box, segmentation mask, visual relationship annotation 포함

프로젝트 활용:

- 컵, 병, 책, 노트북, 의자, 가방, 옷, 신발 등 생활 물건 탐지 기반 확보

한계:

- “정리되어 있음/어질러져 있음” 상태 라벨은 부족
- 책상/옷장/책꽂이 내부 clutter 상황에 특화되지는 않음

### 4.2 Objects365

용도:

- 대규모 object detection pretraining/fine-tuning
- 365개 category
- 약 200만 장 이미지 규모

프로젝트 활용:

- 생활 물체 탐지 모델의 기본 성능 확보

한계:

- 정리 상황 특화 데이터는 아님

### 4.3 LVIS

용도:

- long-tail object detection
- 1,200개 이상의 category

프로젝트 활용:

- 가위, 리모컨, 병, 박스, 가방, 옷, 신발 등 세밀한 생활 물건 탐지에 유리

한계:

- “이 물건을 어디에 정리해야 하는가” 라벨은 없음

### 4.4 HomeObjects-3K

용도:

- household object detection
- 약 3,000장 이미지
- 12개 household object class

프로젝트 활용:

- 가정 환경 기반 detector baseline
- YOLO 학습 파이프라인과 연결하기 쉬움

한계:

- class 수가 적음
- 작은 잡동사니 탐지에는 부족

### 4.5 OCID: Object Clutter Indoor Dataset

용도:

- cluttered indoor scene object segmentation
- RGB-D / point cloud / pixel-wise annotation

프로젝트 활용:

- 어질러진 테이블 위 객체 segmentation 실험
- SAM 2, YOLO-seg, Mask R-CNN 비교 가능

한계:

- 로봇 비전용 데이터라 일반 스마트폰 사진과 domain gap이 있을 수 있음

### 4.6 ESD: Event-based Segmentation Dataset for Indoor Cluttered Environment

용도:

- cluttered indoor tabletop segmentation
- RGB frame, instance mask, event camera data 포함

프로젝트 활용:

- 어질러진 테이블 위 객체 분할 연구에 적합

한계:

- event camera 기반 데이터라 일반 RGB 앱과 차이가 있음

### 4.7 HOTS: Household Objects in Tabletop Scenarios

용도:

- tabletop household object recognition
- RGB-D 기반 household object dataset

프로젝트 활용:

- 책상/테이블 위 생활 물체 인식 실험

한계:

- 옷장/책꽂이까지 커버하지 않음

### 4.8 SUN RGB-D / NYU Depth V2 / ScanNet / ARKitScenes

용도:

- indoor scene understanding
- 가구/공간 인식

프로젝트 활용:

- 책상, 책꽂이, 옷장, 침대, 소파 등 큰 가구 및 공간 맥락 이해
- “책상 위”, “옷장 안”, “책장 근처” 같은 context 추론

한계:

- 작은 생활 물체 detection에는 부족

### 4.9 Messy Rooms Dataset / Messy vs Clean Room

용도:

- clean/messy room classification
- scene-level messiness classification

프로젝트 활용:

- 전체 공간이 얼마나 어질러졌는지 점수화하는 보조 task

한계:

- 객체별 bounding box/mask가 없는 경우가 많음
- “무엇이 어질러져 있는지”를 알려주기에는 부족

### 4.10 Roboflow Universe Household/Messy/Clutter Datasets

용도:

- 빠른 YOLO prototype fine-tuning

프로젝트 활용:

- household objects, messy room, clutter 관련 소규모 데이터셋을 빠르게 실험

한계:

- 데이터 품질과 라벨 일관성이 제각각일 수 있음
- 연구용 메인 데이터셋보다는 prototype용에 가까움

---

## 5. Baseline Implementation

### 5.1 Install

```bash
pip install ultralytics opencv-python pandas
```

### 5.2 Folder Structure

```text
organize-detector/
├── baseline.py
├── test_images/
│   └── messy_desk.jpg
└── outputs/
```

### 5.3 baseline.py

```python
import os
import cv2
import pandas as pd
from ultralytics import YOLO


OBJECT_CLASSES = [
    # study / stationery
    "book", "notebook", "paper", "document", "pen", "pencil",
    "marker", "scissors", "ruler", "eraser",

    # electronics
    "laptop", "keyboard", "mouse", "monitor", "phone",
    "tablet", "charger", "cable", "earphones", "remote control",

    # kitchen / food
    "cup", "bottle", "mug", "plate", "bowl", "spoon",
    "fork", "snack", "food wrapper",

    # clothing / accessories
    "clothes", "shirt", "hoodie", "pants", "socks",
    "bag", "backpack", "hat", "shoe",

    # daily supplies
    "cosmetics", "tissue", "medicine", "toothbrush",
    "comb", "mirror",

    # trash / misc
    "trash", "box", "plastic bag", "toy"
]


CATEGORY_MAP = {
    "study_supplies": {
        "items": [
            "book", "notebook", "paper", "document", "pen", "pencil",
            "marker", "scissors", "ruler", "eraser"
        ],
        "korean_name": "학습/문구류",
        "recommendation": "책은 책꽂이에 세우고, 종이는 파일철에 모으고, 필기구는 필통이나 문구함에 넣으세요."
    },
    "electronics": {
        "items": [
            "laptop", "keyboard", "mouse", "monitor", "phone",
            "tablet", "charger", "cable", "earphones", "remote control"
        ],
        "korean_name": "전자기기/케이블",
        "recommendation": "케이블과 충전기는 한 박스에 모으고, 자주 쓰는 전자기기는 책상 한쪽에 고정 위치를 정하세요."
    },
    "kitchen_items": {
        "items": [
            "cup", "bottle", "mug", "plate", "bowl", "spoon",
            "fork", "snack", "food wrapper"
        ],
        "korean_name": "식기/음식 관련",
        "recommendation": "컵과 식기는 주방으로 옮기고, 음식 포장지는 바로 버리세요."
    },
    "clothing": {
        "items": [
            "clothes", "shirt", "hoodie", "pants", "socks",
            "bag", "backpack", "hat", "shoe"
        ],
        "korean_name": "의류/잡화",
        "recommendation": "입을 옷과 세탁할 옷을 분리하고, 가방과 모자는 옷장이나 행거에 걸어두세요."
    },
    "daily_supplies": {
        "items": [
            "cosmetics", "tissue", "medicine", "toothbrush",
            "comb", "mirror"
        ],
        "korean_name": "생활용품",
        "recommendation": "화장품과 생활용품은 작은 바구니나 서랍에 모아 보관하세요."
    },
    "trash_or_misc": {
        "items": [
            "trash", "box", "plastic bag", "toy"
        ],
        "korean_name": "쓰레기/기타",
        "recommendation": "버릴 것은 먼저 제거하고, 박스나 비닐봉지는 접어서 따로 모으세요."
    }
}


def classify_item(item_name: str):
    for category_key, info in CATEGORY_MAP.items():
        if item_name in info["items"]:
            return category_key, info["korean_name"], info["recommendation"]

    return "unknown", "미분류", "정리 위치를 사용자가 직접 지정해야 합니다."


def detect_objects(image_path: str, output_dir: str = "outputs", conf: float = 0.08):
    os.makedirs(output_dir, exist_ok=True)

    model = YOLO("yolov8s-world.pt")
    model.set_classes(OBJECT_CLASSES + [""])

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

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_image_path = os.path.join(output_dir, f"{base_name}_detected.jpg")
    output_csv_path = os.path.join(output_dir, f"{base_name}_objects.csv")
    output_prompt_path = os.path.join(output_dir, f"{base_name}_llm_prompt.txt")

    cv2.imwrite(output_image_path, image)

    if not df.empty:
        df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
        llm_prompt = build_llm_prompt(df)
        with open(output_prompt_path, "w", encoding="utf-8") as f:
            f.write(llm_prompt)
    else:
        output_csv_path = None
        output_prompt_path = None

    return df, output_image_path, output_prompt_path


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


def summarize_by_category(df: pd.DataFrame):
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
    image_path = "test_images/messy_desk.jpg"

    df, output_image, output_prompt = detect_objects(
        image_path=image_path,
        output_dir="outputs",
        conf=0.08
    )

    summarize_by_category(df)

    print("\n저장 완료")
    print(f"- 탐지 결과 이미지: {output_image}")
    print(f"- LLM 프롬프트 파일: {output_prompt}")
```

---

## 6. Run Baseline

```bash
python baseline.py
```

Expected outputs:

```text
outputs/
├── messy_desk_detected.jpg
├── messy_desk_objects.csv
└── messy_desk_llm_prompt.txt
```

---

## 7. Suggested Object Classes for MVP

처음부터 너무 많은 class를 잡지 말고, MVP에서는 20~30개 정도로 시작하는 것이 좋습니다.

```text
book
notebook
paper
pen
cup
bottle
laptop
phone
charger
cable
earphones
keyboard
mouse
tissue
cosmetics
medicine
clothes
socks
bag
box
trash
snack
food_wrapper
```

---

## 8. Category Mapping Design

LLM에 객체 리스트를 그대로 넘기기보다, 먼저 정리 카테고리로 묶는 것이 좋습니다.

```text
object_name: charger
storage_category: electronics_accessory
recommended_place: cable_box
messiness_role: misplaced
priority: medium
```

추천 카테고리:

| Category Key | Korean Name | Example Objects | Recommended Place |
|---|---|---|---|
| study_supplies | 학습/문구류 | book, notebook, paper, pen | 책꽂이, 파일철, 필통 |
| electronics | 전자기기/케이블 | laptop, phone, charger, cable | 책상 한쪽, 케이블 박스 |
| kitchen_items | 식기/음식 관련 | cup, bottle, plate, snack | 주방, 세척 공간, 쓰레기통 |
| clothing | 의류/잡화 | clothes, socks, bag, hat | 옷장, 세탁 바구니, 행거 |
| daily_supplies | 생활용품 | cosmetics, tissue, medicine | 서랍, 바구니 |
| trash_or_misc | 쓰레기/기타 | trash, food wrapper, plastic bag | 쓰레기통, 분리수거함 |

---

## 9. Evaluation Plan

초기에는 mAP보다 **Recall**이 중요합니다.

정리 앱에서는 “물건을 조금 잘못 탐지하는 것”보다 “정리해야 할 물건을 놓치는 것”이 더 큰 문제입니다.

평가 항목:

| Evaluation Metric | Meaning |
|---|---|
| Detection Recall | 실제 물건 중 몇 개를 탐지했는가 |
| False Positive Rate | 없는 물건을 얼마나 잘못 탐지했는가 |
| Category Accuracy | 탐지된 객체를 올바른 정리 카테고리로 묶었는가 |
| Recommendation Usefulness | LLM 정리 추천이 실제로 쓸 만한가 |
| Small Object Recall | 케이블, 립밤, 포스트잇 등 작은 물건을 잡는가 |
| Occlusion Robustness | 겹쳐 있거나 일부 가려진 물건을 잡는가 |

---

## 10. Fine-tuning Strategy

Baseline 이후 fine-tuning은 다음 순서로 진행합니다.

```text
1. 직접 찍은 책상/방/옷장 사진 50장 테스트
2. 모델 탐지 성공/실패 객체 기록
3. 자주 실패하는 객체 class 선정
4. MVP class 20~30개 확정
5. Roboflow / CVAT / Label Studio로 라벨링
6. YOLOv10, YOLOE, YOLO-World 중 하나로 fine-tuning
7. SAM 2로 segmentation refinement
8. LLM 정리 추천과 연결
```

실패 케이스 기록 예시:

```text
원본 이미지: messy_desk_001.jpg

모델 탐지 성공:
- book
- cup
- laptop

모델 탐지 실패:
- charger
- tangled cable
- sticky note
- receipt
- lip balm
- hair tie
```

---

## 11. Recommended Final System Architecture

```text
[User Image]
    ↓
OpenCV Preprocessing
- resize
- brightness correction
- blur/sharpness check
- image tiling for small objects
    ↓
Object Detection
- YOLO-World baseline
- YOLOE or Grounding DINO for advanced version
    ↓
Segmentation Refinement
- SAM 2
    ↓
Object List Generation
- object name
- confidence
- bbox
- optional mask
- approximate location
    ↓
Category Mapping
- study supplies
- electronics
- kitchen/food
- clothing
- daily supplies
- trash/misc
    ↓
LLM Recommendation
- category summary
- priority ranking
- 5-minute cleanup routine
- storage rules
    ↓
User Output
```

---

## 12. Possible Research / Capstone Titles

Korean:

```text
Open-Vocabulary Object Detection과 LLM을 활용한 실내 정리 보조 시스템
```

```text
YOLOE/SAM 2 기반 생활 공간 객체 탐지 및 LLM 기반 정리 추천 시스템
```

English:

```text
Cluttered Indoor Scene Understanding for Personalized Room Organization
```

```text
LLM-Guided Room Organization via Open-Vocabulary Object Detection
```

---

## 13. Current Best Recommendation

가장 현실적인 개발 방향은 다음입니다.

```text
YOLO-World baseline
    ↓
직접 수집 데이터로 실패 케이스 분석
    ↓
YOLOE 또는 YOLOv10 fine-tuning
    ↓
SAM 2 segmentation 추가
    ↓
LLM 정리 추천 연결
```

이유:

- 사용자가 어떤 물건을 찍을지 예측하기 어렵기 때문에 open-vocabulary detector가 유리함
- 어질러진 책상/방에서는 물건이 겹치므로 segmentation이 필요함
- 직접 수집 데이터가 거의 필수지만, 처음부터 많은 데이터를 만들 필요는 없음
- baseline을 먼저 돌려본 뒤 자주 실패하는 물건만 라벨링하는 것이 효율적임

---

## 14. Immediate Next Tasks for Codex

Codex에서 이어서 구현할 작업:

1. `baseline.py`를 실제 파일로 생성
2. `test_images/`와 `outputs/` 폴더 생성
3. CLI argument 추가
   - `--image`
   - `--conf`
   - `--model`
   - `--output-dir`
4. 결과 CSV 저장
5. 결과 JSON 저장
6. LLM prompt `.txt` 저장
7. bbox 표시 이미지 저장
8. 실패 케이스 기록용 `failure_log.csv` 템플릿 생성
9. 추후 Streamlit 또는 Gradio UI 추가

CLI 예시:

```bash
python baseline.py --image test_images/messy_desk.jpg --conf 0.08 --model yolov8s-world.pt
```

---

## 15. Notes

- 초기 baseline은 fine-tuning 없이 시작한다.
- 핵심은 성능 최적화가 아니라 탐지 실패 목록을 모으는 것이다.
- OpenCV는 전처리, 시각화, 결과 저장, 실시간 영상 처리에 사용한다.
- 학습/fine-tuning은 Ultralytics YOLO, YOLOE, Grounding DINO, SAM 2 등 외부 모델 파이프라인을 사용한다.
- LLM은 최종 정리 추천, 우선순위화, 사용자 친화적 설명 생성에 사용한다.
