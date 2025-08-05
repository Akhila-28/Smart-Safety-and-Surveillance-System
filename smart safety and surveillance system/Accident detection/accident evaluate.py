import os
import sys
import torch
import requests
from collections import deque
from PIL import Image, ImageFile
from torchvision import models, transforms
import cv2

# ========== Setup ==========
ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path='tensorboardexp.pt'):
    model = models.densenet161()
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(2208, 1000),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(1000, 2),
        torch.nn.LogSoftmax(dim=1)
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

model = load_model()
classes = ['accident', 'noaccident']

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ========== Location Fetch ==========
def get_location():
    try:
        response = requests.get('https://ipinfo.io/json')
        if response.status_code == 200:
            data = response.json()
            loc = data.get("loc")
            if loc:
                lat, lon = loc.split(",")
                return {
                    "latitude": lat,
                    "longitude": lon,
                    "city": data.get("city"),
                    "region": data.get("region")
                }
    except Exception as e:
        print("Error fetching geolocation:", e)
    return None

# ========== Alert Logic ==========
def alert_cli(enable_location=False):
    loc = get_location() if enable_location else None
    print("Accident Detected!")
    if loc:
        print(f"Google Maps: https://www.google.com/maps?q={loc['latitude']},{loc['longitude']}")
    else:
        print("Unable to fetch location.")

# ========== Utility ==========
def compute_probs(logits):
    exps = torch.exp(logits)
    return exps / exps.sum()

# ========== Process Image ==========
def process_image(path, thresh=0.3):
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        return
    img = Image.open(path).convert('RGB')
    orig = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)[0]
        probs = compute_probs(logits)
        acc_p = probs[0].item()
        label = 'accident' if acc_p >= thresh else 'noaccident'
    print(f"[Image] {os.path.basename(path)} -> {label} (acc_p={acc_p:.2f})")
    if label == 'accident':
        alert_cli(enable_location=False)
    title = f"{label.upper()} ({acc_p:.2f})"
    cv2.imshow(title, cv2.cvtColor(orig, cv2.COLOR_RGB2BGR))
    print("Press any key to close image window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ========== Process Video ==========
def process_video(path, thresh=0.3, window_size=5):
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        return
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Cannot open video: {path}")
        return

    frame_no = 0
    prob_window = deque(maxlen=window_size)
    alerted = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)[0]
            probs = compute_probs(logits)
            acc_p = probs[0].item()

        prob_window.append(acc_p)
        avg_p = sum(prob_window) / len(prob_window)
        label = 'accident' if avg_p >= thresh else 'noaccident'

        display = frame.copy()
        color = (0, 0, 255) if label == 'accident' else (0, 255, 0)
        text = f"{label.upper()} avg_p={avg_p:.2f}"
        cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow('Detection', display)
        print(f"[Frame {frame_no}] acc_p={acc_p:.2f}, avg_p={avg_p:.2f} -> {label}")

        if label == 'accident' and not alerted:
            alerted = True
            alert_cli(enable_location=False)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            print('Interrupted by user.')
            break
        frame_no += 1

    cap.release()
    cv2.destroyAllWindows()

# ========== Process Webcam ==========
def process_webcam(thresh=0.3, window_size=5):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not detected.")
        return

    print("Press 'q' to quit webcam.")
    prob_window = deque(maxlen=window_size)
    alerted = False
    frame_no = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)[0]
            probs = compute_probs(logits)
            acc_p = probs[0].item()

        prob_window.append(acc_p)
        avg_p = sum(prob_window) / len(prob_window)
        label = 'accident' if avg_p >= thresh else 'noaccident'

        color = (0, 0, 255) if label == 'accident' else (0, 255, 0)
        text = f"{label.upper()} avg_p={avg_p:.2f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Webcam Detection", frame)

        print(f"[Frame {frame_no}] acc_p={acc_p:.2f}, avg_p={avg_p:.2f} -> {label}")

        if label == 'accident' and not alerted:
            alerted = True
            alert_cli(enable_location=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopped by user.")
            break
        frame_no += 1

    cap.release()
    cv2.destroyAllWindows()

# ========== CLI ==========
if __name__ == '__main__':
    if len(sys.argv) > 1:
        input_arg = sys.argv[1]
        if input_arg == 'webcam':
            process_webcam()
        else:
            process_video(input_arg)
    else:
        print('=== Accident Detection CLI ===')
        print('1) Photo   2) Video   3) Webcam')
        choice = input('Select mode [1-3]: ').strip()

        if choice == '1':
            path = input('Enter image file path: ').strip()
            process_image(path)
        elif choice == '2':
            path = input('Enter video file path: ').strip()
            process_video(path)
        elif choice == '3':
            process_webcam()
        else:
            print('Invalid choice.')
