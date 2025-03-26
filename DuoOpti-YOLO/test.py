import torch
from ultralytics import YOLO
from utils.metrics import evaluate_model

model = YOLO('runs/train/duoopti-yolo/weights/best.pt')
results = model('data/images/val', save=True)
evaluate_model(results)
