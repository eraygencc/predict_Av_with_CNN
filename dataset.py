# src/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from simulate_galaxy import simulate_galaxy
from tqdm import tqdm

class GalaxyDataset(Dataset):
    """
    PyTorch Dataset wrapper for simulated galaxy images.
    Converts images and labels to torch tensors.
    """
    def __init__(self, images, labels):
        # Adding channel dimension for grayscale images
        self.images = torch.tensor(images, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def generate_dataset(n_samples=5000, image_size=64, pixel_scale=0.234):
    """
    Generate a dataset of simulated galaxies with random dust extinction (tau).
    
    Returns:
        images: np.array of shape (n_samples, image_size, image_size)
        labels: np.array of shape (n_samples,)
    """
    images = []
    labels = []
    
    for _ in tqdm(range(n_samples), desc="Generating dataset"):
        tau = np.random.uniform(0.0, 1.5)          # random tau
        img = simulate_galaxy(tau, image_size, pixel_scale)
        images.append(img)
        labels.append(tau)
    
    return np.array(images), np.array(labels)

def get_train_val_loaders(n_samples=5000, batch_size=64, val_split=0.2):
    """
    Generate dataset and return PyTorch DataLoaders for training and validation.
    """
    images, labels = generate_dataset(n_samples)
    
    split = int((1 - val_split) * len(labels))
    train_ds = GalaxyDataset(images[:split], labels[:split])
    val_ds   = GalaxyDataset(images[split:], labels[split:])
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)
    
    return train_loader, val_loader
