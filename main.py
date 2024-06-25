from ultralytics import YOLO
import torch

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print (f"Using device {device}")
    model.to(device)

    results = model.train(data="data.yaml", epochs=100)

    model.save("model-v2.pt")
