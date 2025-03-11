import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
from transformers import ViTImageProcessor


class GuitarTabDataset(Dataset):
    def __init__(self, audio_dir, annotation_dir, img_size=(128, 128)):
        self.audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.npy')])
        self.annotation_files = sorted([f for f in os.listdir(annotation_dir) if f.endswith('.npy')])

        assert len(self.audio_files) == len(self.annotation_files), "Mismatch in audio and annotation file counts."

        self.audio_dir = audio_dir
        self.annotation_dir = annotation_dir
        self.img_size = img_size
        self.processor = ViTImageProcessor.from_pretrained('facebook/dino-vits8')

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
        annotation_path = os.path.join(self.annotation_dir, self.annotation_files[idx])

        # Load CQT spectrogram and annotations
        audio = np.load(audio_path, mmap_mode='r').astype(np.float32)  # Already dB normalized
        annotation = np.load(annotation_path, mmap_mode='r').astype(np.float32)

        # Ensure tensors are contiguous for better performance
        audio = torch.tensor(np.ascontiguousarray(audio))

        # Ensure 3-channel input for ViT (ViT expects RGB-like inputs)
        audio = audio.unsqueeze(0)  # (H, W) → (1, H, W)
        # audio = audio.repeat(3, 1, 1)  # (1, H, W) → (3, H, W)

        # Resize using bicubic interpolation
        audio = torch.nn.functional.interpolate(audio.unsqueeze(0), size=self.img_size, mode='bicubic', align_corners=False).squeeze(0)

        # Use ViTImageProcessor for final formatting (normalization, channel order)
        audio = self.processor(images=audio.permute(1, 2, 0).numpy(), return_tensors="pt")["pixel_values"].squeeze(0)

        # Ensure annotation tensors are also contiguous (6 heads for guitar strings)
        heads = [torch.tensor(np.ascontiguousarray(annotation[i])).unsqueeze(0) for i in range(6)]

        return audio, heads


def create_dataloaders(audio_dir, annotation_dir, batch_size=32, train_ratio=0.8, val_ratio=0.1, img_size=(128, 128)):
    dataset = GuitarTabDataset(audio_dir, annotation_dir, img_size)

    # Split dataset into train, validation, and test
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # DataLoader configuration
    loader_args = {
        'batch_size': batch_size,
        'num_workers': min(8, os.cpu_count() or 1),
        'pin_memory': torch.cuda.is_available(),
        'prefetch_factor': 2,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    return train_loader, val_loader, test_loader
