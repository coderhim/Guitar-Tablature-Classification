import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os

class GuitarTabDataset(Dataset):
    def __init__(self, audio_dir, annotation_dir, img_size=(224, 224)):
        self.audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.npy')])
        self.annotation_files = sorted([f for f in os.listdir(annotation_dir) if f.endswith('.npy')])

        assert len(self.audio_files) == len(self.annotation_files), "Mismatch in audio and annotation file counts."

        self.audio_dir = audio_dir
        self.annotation_dir = annotation_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
        annotation_path = os.path.join(self.annotation_dir, self.annotation_files[idx])

        # Load audio (e.g., spectrogram) and annotations
        audio = np.load(audio_path, mmap_mode='r').astype(np.float32)
        annotation = np.load(annotation_path, mmap_mode='r').astype(np.float32)

        # Ensure tensors are contiguous for efficient GPU use
        audio = torch.tensor(np.ascontiguousarray(audio))

        # Resize to ViT input size (224x224 by default)
        audio = torch.nn.functional.interpolate(audio.unsqueeze(0), size=self.img_size, mode='bilinear', align_corners=False).squeeze(0)

        # Split annotations into six output heads
        heads = [torch.tensor(np.ascontiguousarray(annotation[i])) for i in range(6)]

        return audio, heads

def create_dataloaders(audio_dir, annotation_dir, batch_size=32, train_ratio=0.8, val_ratio=0.1, img_size=(224, 224)):
    dataset = GuitarTabDataset(audio_dir, annotation_dir, img_size)

    # Split dataset into train, validation, and test
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # DataLoader configuration
    loader_args = {
        'batch_size': batch_size,
        'num_workers': min(4, os.cpu_count() // 2),  # Efficient CPU usage
        'pin_memory': torch.cuda.is_available(),     # Optimize GPU transfers
        'prefetch_factor': 2,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    return train_loader, val_loader, test_loader
