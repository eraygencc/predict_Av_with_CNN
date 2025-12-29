# src/train.py

import torch
from torch import nn, optim
from cnn_model import DustCNN
from dataset import get_train_val_loaders

# Device setup

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# Hyperparameters
BATCH_SIZE = 64
N_SAMPLES = 5000
EPOCHS = 10
LEARNING_RATE = 1e-3


# Prepare data
train_loader, val_loader = get_train_val_loaders(n_samples=N_SAMPLES, batch_size=BATCH_SIZE)
print(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}")


# Model, loss, optimizer
model = DustCNN().to(device)
criterion = nn.MSELoss()                # Regression: predict dust tau
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training and validation loops
def train_epoch(loader):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(loader):
    """
    Evaluate the model on validation data.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item()
    return total_loss / len(loader)


# Full training loop
for epoch in range(EPOCHS):
    train_loss = train_epoch(train_loader)
    val_loss = eval_epoch(val_loader)
    print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
