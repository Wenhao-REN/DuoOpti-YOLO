from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()

    model = YOLO('models/yolov8n.yaml')
    model.train(
        data='data/erosion.yaml',
        epochs=100,
        imgsz=640,
        batch=8,
        name='duoopti-yolo',
        device=0
    )

if __name__ == "__main__":
    main()
