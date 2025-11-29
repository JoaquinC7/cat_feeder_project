#feeder.py
import cv2
import tensorflow as tf
import numpy as np
from time import sleep
from gpiozero import Servo

IMG_SIZE = (160, 160)
AUTHORIZED_CAT = "tabby"   # Change for each feeder
SERVO_PIN = 18             # GPIO pin controlling the feeder's servo

# Labels must match order in training generator
LABELS = ["tabby", "black", "calico"]

# Load model
model = tf.keras.models.load_model("cat_classifier.h5")

# Setup servo
servo = Servo(SERVO_PIN)

def open_feeder():
    print("Opening feeder...")
    servo.max()   # fully open
    sleep(4)
    print("Closing feeder...")
    servo.min()   # fully closed
    sleep(1)

def preprocess(frame):
    img = cv2.resize(frame, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def predict_cat(frame):
    x = preprocess(frame)
    preds = model.predict(x)
    return LABELS[np.argmax(preds)]

def main():
    cap = cv2.VideoCapture(0)
    print("Feeder system active. Press 'q' to quit.")

    cooldown = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cat = predict_cat(frame)

        cv2.putText(frame, f"Detected: {cat}", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Feeder Camera", frame)

        if cat == AUTHORIZED_CAT and cooldown == 0:
            open_feeder()
            cooldown = 60  # 60 frames cooldown (≈2 seconds)

        if cooldown > 0:
            cooldown -= 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
