# train_yolo.py
from ultralytics import YOLO
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.yaml", help="path to data.yaml")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model", default="yolov8n.pt", help="base model to fine-tune (yolov8n.pt recommended)")
    args = parser.parse_args()

    print("Training YOLO detector")
    print(f"data: {args.data}")
    print(f"base model: {args.model}")
    print(f"epochs: {args.epochs}")
    print(f"imgsz: {args.imgsz}")

    # Create model from pretrained weights
    model = YOLO(args.model)

    # Train (this will save best weights to runs/train/...)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, name="cat_detector")

    print("Training finished. Check runs/train/cat_detector for results.")

if __name__ == "__main__":
    main()
