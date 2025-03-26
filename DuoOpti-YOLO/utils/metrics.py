import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(results):
    print("Precision:", results[0].boxes.conf.mean().item())
    print("Detected:", len(results[0].boxes))
    # Add mAP, IoU calculation if needed
