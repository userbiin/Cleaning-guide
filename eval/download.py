import roboflow
# 가장 간단: dataset_url 직접 지정 (YOLO 형식으로 다운로드)
dataset = roboflow.download_dataset(
    dataset_url="https://universe.roboflow.com/1-cvbtv/office-desk/dataset/1",
    model_format="yolov8")
print(dataset.location)   # 다운로드된 경로

