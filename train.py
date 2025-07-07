#!/usr/bin/env python3
"""
Q-Network Training Script
Trains a deep neural network to predict Q-values using Monte Carlo targets
"""

import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import argparse


class QNetwork(nn.Module):
    """Deep Q-Network with tanh activation"""
    def __init__(self, input_dim: int, hidden_sizes=(256, 128, 64, 32, 16, 8)):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.Tanh())
            last_dim = h
        # Final output layer → tanh bounded in [-1,1]
        layers.append(nn.Linear(last_dim, 1))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_data(h5_path):
    """Load features and Q-values from HDF5 file"""
    print(f"Loading data from: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        print("Top-level keys in HDF5:", list(f.keys()))
        X = f["features"][:]
        y_MC = f["q_values"][:]
    print(f"Loaded X.shape = {X.shape}, y_MC.shape = {y_MC.shape}")
    return X, y_MC


def prepare_data(X, y_MC, test_size=0.2, batch_size=512, device='cpu'):
    """Split, scale, and create DataLoaders"""
    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_MC, test_size=test_size, random_state=42, shuffle=True
    )
    
    print("\nAfter split:")
    print(f"  X_train.shape = {X_train.shape}, y_train.shape = {y_train.shape}")
    print(f"  X_val.shape   = {X_val.shape},   y_val.shape   = {y_val.shape}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    print("\nFeature stats (first 5 dims) after scaling:")
    print("  Means:", X_train_scaled[:, :5].mean(axis=0))
    print("  Stds:", X_train_scaled[:, :5].std(axis=0))
    
    # Convert to PyTorch tensors
    X_train_t = torch.from_numpy(X_train_scaled).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    X_val_t = torch.from_numpy(X_val_scaled).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True if device == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device == "cuda" else False
    )
    
    return train_loader, val_loader, scaler, X_train_t.shape[1]


def train_epoch(model, train_loader, optimizer, criterion):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        preds = model(batch_X)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0)
    
    return running_loss / len(train_loader.dataset)


def validate(model, val_loader, criterion):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            running_loss += loss.item() * batch_X.size(0)
    
    return running_loss / len(val_loader.dataset)


def plot_predictions(model, val_loader, save_path=None):
    """Plot predicted vs true Q-values"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            pred_vals = model(batch_X).cpu().numpy()
            all_preds.append(pred_vals)
            all_targets.append(batch_y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Plot first 1000 points
    n_plot = min(1000, len(all_preds))
    plt.figure(figsize=(6, 6))
    plt.scatter(
        all_targets[:n_plot],
        all_preds[:n_plot],
        s=3, alpha=0.4, color="tab:blue"
    )
    
    # Add diagonal line
    min_val = min(all_targets.min(), all_preds.min())
    max_val = max(all_targets.max(), all_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val],
             color="black", linestyle="--", linewidth=1)
    
    plt.xlabel("MC Target Q")
    plt.ylabel("Predicted Q̂")
    plt.title("MC Pretraining: Validation Set (first 1,000 samples)")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Train Q-Network with Monte Carlo targets")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to HDF5 file containing features and Q-values")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size for training (default: 512)")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate (default: 5e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="Weight decay for Adam optimizer (default: 1e-5)")
    parser.add_argument("--hidden-sizes", type=int, nargs="+",
                        default=[256, 128, 64, 32, 16, 8],
                        help="Hidden layer sizes (default: 256 128 64 32 16 8)")
    parser.add_argument("--save-model", type=str, default="qnet_mc_pretrained.pth",
                        help="Path to save best model (default: qnet_mc_pretrained.pth)")
    parser.add_argument("--save-plot", type=str, default=None,
                        help="Path to save prediction plot (optional)")
    parser.add_argument("--no-cuda", action="store_true",
                        help="Disable CUDA even if available")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    X, y_MC = load_data(args.data_path)
    
    # Prepare data
    train_loader, val_loader, scaler, input_dim = prepare_data(
        X, y_MC, batch_size=args.batch_size, device=device
    )
    
    # Initialize model
    model = QNetwork(input_dim=input_dim, hidden_sizes=tuple(args.hidden_sizes)).to(device)
    print(f"\nModel architecture:\n{model}")
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    
    print(f"\nTraining parameters:")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Weight decay: {args.weight_decay}")
    
    # Training loop
    best_val_loss = float("inf")
    print("\nStarting training...")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        
        # Validate
        val_loss = validate(model, val_loader, criterion)
        
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'scaler': scaler,
                'args': args
            }, args.save_model)
            print(f"  → Saved new best model (val_loss: {val_loss:.6f})")
    
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.6f}")
    
    # Load best model and plot predictions
    checkpoint = torch.load(args.save_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    # Generate prediction plot
    plot_predictions(model, val_loader, save_path=args.save_plot)


if __name__ == "__main__":
    main()
