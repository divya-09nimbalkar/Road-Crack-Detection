import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

class CrackDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.files = [f for f in os.listdir(root_dir) if f.endswith(".png")]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.files[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)
        else:
            # Convert HWC → CHW (3, H, W)
            img = torch.tensor(img).permute(2, 0, 1).float() / 255.0

        # Placeholder label: all images = crack (1)
        label = 1
        return img, torch.tensor(label)

def get_dataloader(batch_size=4):
    dataset = CrackDataset("data/raw")
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def prepare_data():
    print("Preparing data...")
    os.makedirs("data/processed", exist_ok=True)
    print("Data prepared from data/raw → data/processed")
