from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(model="yolov8n.yaml")
    model.train(
        data="NGVOC.yaml",
        epochs=1000,
        imgsz=1024,
        device=0,
        name="./logs",
        project="./glass-detection",
    )
