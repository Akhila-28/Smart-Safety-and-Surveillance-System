import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageFile
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from torch.optim import lr_scheduler

ImageFile.LOAD_TRUNCATED_IMAGES = True
train_on_gpu = torch.cuda.is_available()
print('Training on GPU' if train_on_gpu else 'Training on CPU')

# Custom dataset that handles both images and videos
class AccidentDataset(Dataset):
    def __init__(self, folder_path, transform=None, frames_per_video=5):
        self.folder_path = folder_path
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.samples = []

        image_exts = ['.jpg', '.jpeg', '.png']
        video_exts = ['.mp4', '.avi', '.mov']

        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            ext = os.path.splitext(file)[1].lower()

            if ext in image_exts:
                self.samples.append((file_path, self._get_label(file)))

            elif ext in video_exts:
                cap = cv2.VideoCapture(file_path)
                if not cap.isOpened():
                    continue

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_idxs = list(range(0, total_frames, max(1, total_frames // self.frames_per_video)))
                for idx in frame_idxs[:self.frames_per_video]:
                    self.samples.append((file_path, self._get_label(file), idx))

                cap.release()

    def _get_label(self, filename):
        return 1 if "accident" in filename.lower() else 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        label = torch.tensor(sample[1], dtype=torch.long)

        if len(sample) == 2:
            # It's an image
            img = Image.open(sample[0]).convert("RGB")
        else:
            # It's a video frame
            video_path, _, frame_idx = sample
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()
            cap.release()

            if not success:
                raise RuntimeError(f"Failed to load frame {frame_idx} from {video_path}")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)

        if self.transform:
            img = self.transform(img)

        return img, label

# Transforms
train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = 'accidents'
full_dataset = AccidentDataset(data_dir, transform=train_transforms)

# Split dataset
train_size = int(0.7 * len(full_dataset))
valid_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - valid_size
train_data, valid_data, test_data = random_split(full_dataset, [train_size, valid_size, test_size])

trainloader = DataLoader(train_data, batch_size=8, shuffle=True)
validloader = DataLoader(valid_data, batch_size=8, shuffle=True)
testloader = DataLoader(test_data, batch_size=8, shuffle=True)

# Load pretrained model
model = models.densenet161(pretrained=True)
model.classifier = nn.Sequential(
    nn.Linear(2208, 1000),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(1000, 2),
    nn.LogSoftmax(dim=1)
)

if train_on_gpu:
    model.cuda()

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training loop
n_epochs = 10
valid_loss_min = np.Inf

for epoch in range(1, n_epochs + 1):
    train_loss = 0.0
    valid_loss = 0.0
    model.train()

    for data, target in trainloader:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    scheduler.step()
    model.eval()

    with torch.no_grad():
        for data, target in validloader:
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item() * data.size(0)

    train_loss /= len(trainloader.dataset)
    valid_loss /= len(validloader.dataset)

    print(f'Epoch {epoch}: Training Loss: {train_loss:.6f}, Validation Loss: {valid_loss:.6f}')

    if valid_loss <= valid_loss_min:
        print(f'Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), 'accident_model.pt')
        valid_loss_min = valid_loss
