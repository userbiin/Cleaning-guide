"""
Roboflow Student Desk 데이터셋 다운로드 — 한 번만 실행.
다운로드 후 Student-Desk-1/ 에 COCO 포맷으로 저장됨.

실행:
  python eval/download.py

다운로드 이후 순서:
  1. GT 변환 (COCO xywh → xyxy, "none" 카테고리 제거)
     python eval/convert_roboflow_gt.py --split-dir Student-Desk-1/test

  2. 탐지 + 정량 비교
     python eval/compare_detectors.py \\
         --images Student-Desk-1/test \\
         --gt eval/ground_truth \\
         --detectors yolo-world yoloe \\
         --output outputs/detector_comparison
"""

from roboflow import Roboflow

rf = Roboflow(api_key="VNcRZqzR7TsOrkoZOIMm")
project = rf.workspace("subins-workspace-ylfzq").project("Student Desk")
dataset = project.version(1).download("coco")
print(f"다운로드 완료: {dataset.location}")
print("\n다음 단계:")
print(f"  python eval/convert_roboflow_gt.py --split-dir {dataset.location}/test")
