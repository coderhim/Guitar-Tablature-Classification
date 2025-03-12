import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
from PIL import Image
import numpy as np
from torchvision import transforms

class GuitarTabDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.annotation_files = sorted([f for f in os.listdir(annotation_dir) if f.endswith('.npy')])

        # Ensure filenames are matched correctly
        assert len(self.image_files) == len(self.annotation_files), "Mismatch in image and annotation file counts."
        for img, ann in zip(self.image_files, self.annotation_files):
            assert os.path.splitext(img)[0] == os.path.splitext(ann)[0], "Mismatch in corresponding files."

        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),  # Converts to [0, 1] range and CxHxW format
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # Load annotation (using memory-map for efficiency)
        annotation_path = os.path.join(self.annotation_dir, self.annotation_files[idx])
        annotation = np.load(annotation_path, mmap_mode='r').astype(np.float32)
        heads = [torch.tensor(np.ascontiguousarray(annotation[i])) for i in range(6)]

        return image, heads

def create_dataloaders(image_dir, annotation_dir, batch_size=64, train_ratio=0.8, val_ratio=0.1, num_workers=4):
    dataset = GuitarTabDataset(image_dir, annotation_dir)

    # Split dataset into training, validation, and testing
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Optimize DataLoader for performance
    loader_args = {
        'batch_size': batch_size,
        'num_workers': num_workers,       # Increase based on CPU cores
        'pin_memory': torch.cuda.is_available(),  # Speed up transfers to GPU
        'prefetch_factor': 4,             # Preload batches
        'persistent_workers': True,       # Avoid restarting workers every epoch
    }

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    return train_loader, val_loader, test_loader
