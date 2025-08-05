import os
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from torch import nn

# -----------------------
# CONFIG
# -----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", DEVICE)

MODEL_PATH = "tensorboardexp.pt"
CLASSES = ["accident", "noaccident"]

# -----------------------
# LOAD MODEL
# -----------------------
model = models.densenet161(weights=None)
model.classifier = nn.Sequential(
    nn.Linear(2208, 1000),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(1000, 2),
    nn.LogSoftmax(dim=1)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -----------------------
# TRANSFORM
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------
# PREDICT IMAGE
# -----------------------
def predict_image(img_path):
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
        return CLASSES[pred.item()]

# -----------------------
# PREDICT VIDEO
# -----------------------
def predict_video(video_path, frame_skip=30):
    cap = cv2.VideoCapture(video_path)
    frame_preds = []

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return "unknown"

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame)
            pil_img = transform(pil_img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(pil_img)
                _, pred = torch.max(output, 1)
                frame_preds.append(pred.item())

        frame_idx += 1

    cap.release()

    if frame_preds:
        pred_class = max(set(frame_preds), key=frame_preds.count)
        return CLASSES[pred_class]
    else:
        return "unknown"

# -----------------------
# USER INPUT
# -----------------------
def main():
    file_path = input("Enter the full path to your file (image/video): ").strip()

    if not os.path.exists(file_path):
        print("❌ File does not exist. Please check the path.")
        return

    file_type = input("Is it an image or video? (i/v): ").strip().lower()

    if file_type == 'i':
        result = predict_image(file_path)
        print(f"✅ Predicted (Image): {result}")
    elif file_type == 'v':
        result = predict_video(file_path)
        print(f"✅ Predicted (Video): {result}")
    else:
        print("❌ Invalid choice. Please type 'i' for image or 'v' for video.")

if __name__ == "__main__":
    main()
