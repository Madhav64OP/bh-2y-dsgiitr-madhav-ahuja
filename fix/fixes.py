# the fixes i made after
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import json
import os
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
train_videos = [1]
test_videos = [110]  # Example: Train on video 1
IMG_SIZE = 112
BATCH_SIZE = 64
NUM_EPOCHS = 3
NUM_WORKERS = 4 if torch.cuda.is_available() else 0
ROOT_DIR = "/content/CholecT50/"  # ***REPLACE WITH YOUR ACTUAL PATH***
MODEL_SAVE_PATH = 'trained_model.pth'

# --- Dataset class ---
class CholecT50Dataset(Dataset):
    def __init__(self, root_dir, video_ids, transform=None):
        self.root_dir = root_dir
        self.video_ids = video_ids
        self.transform = transform
        self.frame_paths = []
        self.labels = {}

        for vid in video_ids:
            video_dir = os.path.join(root_dir, 'videos', f'VID{vid:02d}')
            label_path = os.path.join(root_dir, 'labels', f'VID{vid:02d}.json')

            frame_files = sorted(os.listdir(video_dir))
            frame_paths = [os.path.join(video_dir, f) for f in frame_files]
            self.frame_paths.extend(frame_paths)

            with open(label_path, 'r') as f:
                self.labels.update(json.load(f))

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        img_path = self.frame_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image file not found at {img_path}")
            return None, None, None, None
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            return None, None, None, None

        if self.transform:
            image = self.transform(image)

        frame_id = os.path.basename(img_path).split('.')[0]
        video_id = os.path.basename(os.path.dirname(img_path))
        label_key = f"{video_id}/{frame_id}"

        labels = self.labels.get(label_key, {})
        triplet_labels = torch.zeros(100)
        bbox_labels = []

        if 'triplets' in labels:
            for t in labels:
                triplet_labels[t[0]] = 1
                if 'bbox' in t:
                    bbox_labels.append({
                        'triplet_id': t['triplet_id'],
                        'bbox': torch.tensor(t['bbox']).tolist() # Store bbox as list directly
                    })

        return image, triplet_labels, bbox_labels, img_path

# --- Model architecture ---
class TripletDetectionModel(nn.Module):
    # ... (Model class code - EXACTLY as before)
        def __init__(self, num_triplets=100):
            super(TripletDetectionModel, self).__init__()
            self.backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            self.triplet_head = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_triplets),
                nn.Sigmoid()
            )

        def forward(self, x):
            features = self.backbone(x)
            features = features.view(features.size(0), -1)
            triplet_out = self.triplet_head(features)
            return triplet_out

# --- Training function ---
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, triplet_labels, _, _ in tqdm(train_loader):
            images = images.to(device)
            triplet_labels = triplet_labels.to(device)

            optimizer.zero_grad()
            triplet_out = model(images)
            triplet_loss = criterion(triplet_out, triplet_labels)
            triplet_loss.backward()
            optimizer.step()

            running_loss += triplet_loss.item()
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

def generate_predictions(model, test_loader, device):
    model.eval()
    predictions = {}

    with torch.no_grad():
        for images, triplet_labels, bbox_labels, img_paths in tqdm(test_loader):
            images = images.to(device)

            triplet_out, detection_out = model(images)

            for idx, triplet_probs in enumerate(triplet_out):
                prediction = {
                    'triplets': triplet_probs.cpu().numpy().tolist(),  # Convert to list for JSON
                    'detection': detection_out[idx].cpu().numpy().tolist()  # Include bounding box predictions
                }
                predictions[f'image_{idx}'] = prediction

    return predictions

# --- Main training execution ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CholecT50Dataset(ROOT_DIR, train_videos, transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    test_dataset = CholecT50Dataset(ROOT_DIR, test_videos, transform)  
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True) 

    model = TripletDetectionModel().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, criterion, optimizer, device, NUM_EPOCHS)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    predictions = generate_predictions(model, test_loader, device)

    # Save predictions
    with open('predictions.json', 'w') as f:
        json.dump(predictions, f)

if __name__ == "__main__":
    main()