# feeder_yolo.py
import cv2, time, numpy as np
from ultralytics import YOLO

# CONFIG
MODEL_PATH = "runs/train/cat_detector/weights/best.pt"
CLASS_NAMES = ["tabby", "black", "calico"]
AUTHORIZED_CLASS = "tabby"   # change per-feeder
CONF_THRESH = 0.4
SERVO_PIN = 18               # Raspberry Pi GPIO pin

# ROI rectangle (x1,y1,x2,y2) in pixels - set after you open camera and see frame size
# We'll auto-calc ROI as bottom center area, but you can customize
def open_feeder_action():
    # On Raspberry Pi you'd use gpiozero Servo control.
    try:
        from gpiozero import Servo
        servo = Servo(SERVO_PIN)
        print("Opening feeder (servo)...")
        servo.max()
        time.sleep(3)
        servo.min()
        print("Feeder closed.")
    except Exception as e:
        # On non-Pi (windows), just print
        print("(SIMULATION) Would open feeder (servo). Error/Note:", e)

def feeder_loop(model_path=MODEL_PATH, source=0, conf=CONF_THRESH):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    ret, sample = cap.read()
    if not ret:
        raise RuntimeError("Failed to read from camera")
    H, W = sample.shape[:2]

    # Define ROI: bottom center box (customize as needed)
    roi_w = int(W * 0.6)
    roi_h = int(H * 0.35)
    roi_x1 = (W - roi_w) // 2
    roi_y1 = H - roi_h
    roi_x2 = roi_x1 + roi_w
    roi_y2 = roi_y1 + roi_h

    cooldown_seconds = 5
    last_open_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        res = model.predict(source=[frame], conf=conf, stream=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else []
        scores = res.boxes.conf.cpu().numpy() if len(res.boxes) else []
        classes = res.boxes.cls.cpu().numpy() if len(res.boxes) else []

        # draw ROI
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (200, 100, 50), 2)
        cv2.putText(frame, "Feeder ROI", (roi_x1 + 5, roi_y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,100,50), 2)

        authorized_detected = False

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            label = f"{CLASS_NAMES[int(cls)]} {float(score):.2f}"
            cv2.rectangle(frame, (x1,y1), (x2,y2), (16,185,129), 2)
            cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # check center in ROI and class match
            if (roi_x1 <= cx <= roi_x2) and (roi_y1 <= cy <= roi_y2):
                if CLASS_NAMES[int(cls)] == AUTHORIZED_CLASS and (time.time() - last_open_time) > cooldown_seconds:
                    authorized_detected = True

        if authorized_detected:
            open_feeder_action()
            last_open_time = time.time()

        cv2.imshow("Feeder Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    feeder_loop()
