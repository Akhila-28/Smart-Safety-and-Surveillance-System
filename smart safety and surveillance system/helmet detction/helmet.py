import sys
import os
from ultralytics import YOLO
import cv2

def detect_helmet_image(image_path):
    print(f"[IMAGE] Detecting helmets in image: {image_path}")
    model = YOLO("best.pt")
    results = model(image_path, save=True, conf=0.5)
    print("[IMAGE] Detection completed.")

def detect_helmet_video(video_path):
    print(f"[VIDEO] Detecting helmets in video: {video_path}")
    model = YOLO("best.pt")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("[VIDEO] Error: Cannot open video file")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, save=False, conf=0.5)
        annotated_frame = results[0].plot()

        cv2.imshow("Helmet Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[VIDEO] Detection finished.")

def detect_helmet_webcam():
    print("[WEBCAM] Starting webcam helmet detection")
    model = YOLO("best.pt")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, save=False, conf=0.5)
        annotated_frame = results[0].plot()

        cv2.imshow("Helmet Detection - Webcam", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[WEBCAM] Detection finished.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python helmet.py <input>")
        sys.exit(1)

    input_arg = sys.argv[1]
    print(f"[MAIN] Running helmet detection on: {input_arg}")

    if input_arg == "webcam":
        detect_helmet_webcam()
    elif input_arg.endswith(('.jpg', '.jpeg', '.png')):
        detect_helmet_image(input_arg)
    elif input_arg.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        detect_helmet_video(input_arg)
    else:
        print("[ERROR] Invalid input format")
