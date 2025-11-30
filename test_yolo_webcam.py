# test_yolo_webcam.py
import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import time

CLASS_NAMES = ["tabby", "black", "calico"]  # must match data.yaml order

def draw_boxes(frame, boxes, scores, classes):
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{CLASS_NAMES[int(cls)]} {score:.2f}"
        # box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (16, 185, 129), 2)
        # label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (16, 185, 129), -1)
        cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

def run_webcam(model_path="runs/train/cat_detector/weights/best.pt", source=0, conf=0.35):
    print("Loading model:", model_path)
    model = YOLO(model_path)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    fps_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference (small resizing inside ultralytics)
        results = model.predict(source=[frame], conf=conf, stream=False)  # single-frame prediction

        # ultralytics returns a list of Result objects (one per image)
        try:
            res = results[0]
            boxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else []
            scores = res.boxes.conf.cpu().numpy() if len(res.boxes) else []
            classes = res.boxes.cls.cpu().numpy() if len(res.boxes) else []
        except Exception:
            boxes, scores, classes = [], [], []

        # draw
        if len(boxes):
            draw_boxes(frame, boxes, scores, classes)

        # FPS
        fps = 1.0 / (time.time() - fps_time)
        fps_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("YOLO Cat Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runs/train/cat_detector/weights/best.pt", help="path to trained .pt file")
    parser.add_argument("--source", default=0, help="webcam device id or video file")
    parser.add_argument("--conf", type=float, default=0.4)
    args = parser.parse_args()
    run_webcam(model_path=args.model, source=int(args.source) if str(args.source).isdigit() else args.source, conf=args.conf)
