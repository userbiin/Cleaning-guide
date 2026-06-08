# Labeling Guide — Organization Object Detector

이 가이드는 `eval/ground_truth/` 의 GT 라벨 파일을 작성하기 위한 기준이다.
**LLM 출력 형식과 사람 라벨링 기준이 완전히 동일**해야 A/B 실험이 유효하다.

---

## 1. Decision Taxonomy (네 값)

| 값 | 의미 | 판정 기준 |
|---|---|---|
| `discard` | 버릴 것 | 재사용 의도 없는 빈 용기·포장지·다 쓴 소모품 |
| `relocate` | 옮길 것 | 유효하지만 현재 위치가 부적절한 물건 |
| `keep` | 제자리 | 현재 위치가 적절한 / 상시 사용 물건 |
| `invalid` | 오탐 | 탐지는 됐으나 실제 물건이 아님 → decision 정확도에서 제외 |

### 1.1 경계 규칙 (모호할 때 이 순서로 적용)

| 상황 | 판정 | 이유 |
|---|---|---|
| 빈 캔 / 빈 페트병 | `discard` | 재사용 의도 없음 |
| 반쯤 남은 음료 | `relocate` | 유효하나 책상 위는 부적절 (주방/냉장고) |
| 다 쓴 물티슈 / 휴지 | `discard` | 소모품 소진 |
| 케이블 — 바닥에 엉킴 | `relocate` | 유효하나 위치 부적절 |
| 케이블 — 제자리 박스에 있음 | `keep` | 정위치 |
| 책 — 책꽂이에 있음 | `keep` | 정위치 |
| 책 — 바닥/책상에 쌓임 | `relocate` | 유효하나 위치 부적절 |
| 음식 포장지 / 영수증 | `discard` | 재사용 없음 |
| 충전 중인 기기 | `keep` | 현재 사용 중 |
| 분류 불가 (잘 안 보임) | `keep` | 확신 낮으면 keep (과잉 discard 방지) |

### 1.2 False Positive 처리 (`invalid`)
탐지된 객체가 실제로 존재하지 않거나 완전히 잘못된 경우:
- `.todo.json`에서 `"_fp": true` 로 표시 (빠른 체크박스)
- finalize 시 자동으로 `decision: "invalid"` 로 변환되어 GT에 보존됨
- `compare_detectors.py` 는 `invalid` GT를 탐지 지표 계산에서 제외하고 FP로 처리

### 1.3 Missed Object (FN 복구)
탐지기가 놓친 실제 물건:
- `.todo.json`의 `"missed_objects"` 배열에 직접 추가
- bbox는 이미지 뷰어에서 픽셀 좌표 확인 후 기입

---

## 2. GT JSON 포맷

### 2.1 파일명 규칙
- 이미지: `eval/data/dev/img001.jpg`
- GT 파일: `eval/ground_truth/img001.json`
- **파일 stem이 정확히 일치해야** 비교 스크립트가 매칭한다.

### 2.2 완성된 GT 파일 (`.json`)

```json
{
  "image_id": "img001",
  "objects": [
    {
      "id": 0,
      "label": "bottle",
      "bbox": [120, 80, 240, 360],
      "decision": "discard"
    },
    {
      "id": 1,
      "label": "book",
      "bbox": [300, 50, 520, 200],
      "decision": "relocate"
    },
    {
      "id": 2,
      "label": "cable",
      "bbox": [10, 400, 80, 430],
      "decision": "relocate"
    }
  ]
}
```

- `bbox`: `[x1, y1, x2, y2]` 픽셀 좌표 (좌상단, 우하단)
- `decision`: `"discard"` | `"relocate"` | `"keep"` 중 하나
- `id`: 0부터 순번 (객체 추적용)

### 2.3 라벨링 작업 파일 (`.todo.json`) — 자동 생성

`make_todo.py` 가 탐지 결과로 초안을 생성한다. 사람이 채워야 할 필드만 `null`.

```json
{
  "_todo": true,
  "_labeled": false,
  "image_id": "img001",
  "objects": [
    {
      "id": 0,
      "label": "bottle",
      "bbox": [120, 80, 240, 360],
      "decision": null,
      "_conf": 0.71,
      "_fp": false
    }
  ],
  "missed_objects": []
}
```

| 필드 | 작업 내용 |
|---|---|
| `"decision": null` | **필수** — `"discard"` / `"relocate"` / `"keep"` 중 하나로 채울 것 |
| `"label"` | 잘못된 경우 수정 |
| `"bbox"` | 크게 어긋난 경우 수정 |
| `"_fp": false` | 오탐이면 `true`로 변경 → finalize 시 제외됨 |
| `"missed_objects"` | 탐지 못한 실제 물건 추가 (GT label 형식으로) |

---

## 3. 작업 흐름

```
1. python eval/make_todo.py --images eval/data/dev --output eval/ground_truth
   → eval/ground_truth/img001.todo.json 자동 생성

2. 각 .todo.json 파일을 열어 "decision" 필드 채우기
   (+ label 보정, _fp 표시, missed_objects 추가)

3. python eval/make_todo.py --finalize --gt eval/ground_truth
   → .todo.json 검증 후 .json 변환
   (decision=null 남아있으면 에러 출력)

4. python eval/compare_detectors.py \
       --images eval/data/dev \
       --gt eval/ground_truth \
       --detectors yolo-world yoloe
```

---

## 4. 라벨러 일치도 (Cohen's kappa)

- **2인 이상**이 동일 이미지 서브셋(≥20장)을 독립 라벨링
- `eval/stats.py`의 `compute_kappa()` 로 일치도 산출
- 목표: κ ≥ 0.7 (substantial agreement). κ < 0.6이면 경계 규칙 보완 후 재라벨링.

---

## 4b. 평가 지표 우선순위 (v3 CV 수업 기준)

| 순위 | 지표 | 용도 |
|---|---|---|
| **1 (CV 코어)** | **mAP@0.5, mAP@0.5:0.95**, Recall, Precision, Small-object AP | 탐지 성능, GT bbox 대비 |
| **1 (CV 핵심 기여)** | fine-tuning 전후 delta | zero-shot → fine-tuned 향상 |
| 2 | detector 비교 (YOLO-World vs YOLOE vs fine-tuned) | `compare_detectors.py` |
| 3 (보조) | cost-weighted decision acc, false-discard rate | describe ON/OFF 시연 |

**cost matrix (v3, 비가역 오류에 최대 페널티):**
```
              pred:discard  pred:relocate  pred:keep
gt:discard         0             2            3       ← 쓰레기 방치
gt:relocate        3             0            1
gt:keep            4             1            0       ← 멀쩡한 것 버림 (최악)
```
→ `eval/metrics.py`의 `COST_MATRIX` 에 구현됨. **false-discard rate**(gt=keep, pred=discard)도 별도 보고.

---

## 5. 수집 다양성 체크리스트

아래 환경을 고르게 포함할 것 (dev 50장 기준):

- [ ] 책상 위 (학습 공간) — 15장 이상
- [ ] 옷장 / 행거 주변 — 10장
- [ ] 주방 / 식탁 — 10장
- [ ] 화장대 / 욕실 — 5장
- [ ] 기타 (바닥, 현관 등) — 10장
- [ ] 소형 객체 포함 (케이블, 립밤, 이어폰) — 전체의 30% 이상
- [ ] `discard` 라벨 포함 이미지 — 전체의 40% 이상

---

## 6. 공개 데이터 활용 (fine-tuning / 탐지 지표 물량 확보)

| 데이터셋 | 용도 | 접근 |
|---|---|---|
| **LVIS** (1,203 카테고리) | 소형/다양 객체 fine-tune — **1순위** | `fiftyone` |
| **Roboflow "Detecting working desk"** (~3,124장) | 책상 도메인 — fine-tune 보완 | Roboflow SDK |
| **Roboflow "Office Desk"** (~2,245장) | 동일 도메인 추가 | Roboflow SDK |
| ~~HomeObjects-3K~~ | **비추천** — 가구 12종 위주, 소형 잡동사니와 도메인 불일치 | — |

**SAHI (Slicing Aided Hyper Inference)**: 소형 객체 Recall 부족 시 이미지 슬라이싱 추론 적용 검토.
→ `pip install sahi` 후 `SahiDetector` 래퍼 추가 (detector.py 슬롯에 플러그인).
