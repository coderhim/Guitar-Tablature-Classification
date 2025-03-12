import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
from PIL import Image
import numpy as np

class GuitarTabDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.annotation_files = sorted([f for f in os.listdir(annotation_dir) if f.endswith('.npy')])

        assert len(self.image_files) == len(self.annotation_files), "Mismatch in image and annotation file counts."

        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        annotation_path = os.path.join(self.annotation_dir, self.annotation_files[idx])

        # Load spectrogram image and resize to (128, 128)
        image = Image.open(image_path).resize((128, 128))
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0

        # Load annotation (e.g., tab heads)
        annotation = np.load(annotation_path, mmap_mode='r').astype(np.float32)
        heads = [torch.tensor(np.ascontiguousarray(annotation[i])) for i in range(6)]

        return image, heads

def create_dataloaders(image_dir, annotation_dir, batch_size=64, train_ratio=0.8, val_ratio=0.1):
    dataset = GuitarTabDataset(image_dir, annotation_dir)

    # Split dataset into training, validation, and testing
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Optimize DataLoader for performance
    loader_args = {
        'batch_size': batch_size,
        'num_workers': 4,  # Adjust based on your system
        'pin_memory': True,  # For faster GPU transfer
        'prefetch_factor': 4,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    return train_loader, val_loader, test_loader
