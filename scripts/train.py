#!/usr/bin/env python3
"""
Training script for the chess neural network model.
"""
import os
import sys
import argparse
import time
import yaml
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torch.nn.functional as F

# Add the parent directory to the Python path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import ChessModel

class ChessDataset(Dataset):
    """Dataset for chess training data from HDF5 file."""
    
    def __init__(self, h5_file, dataset_type='train'):
        """
        Initialize the dataset.
        
        Args:
            h5_file: Path to the HDF5 file containing processed data.
            dataset_type: One of 'train' or 'val' to specify which dataset to use.
        """
        self.h5_file = h5_file
        self.dataset_type = dataset_type
        self.x_key = f'{dataset_type}_x'
        self.policy_key = f'{dataset_type}_policy'
        self.value_key = f'{dataset_type}_value'
        
        # Open the file just to get metadata
        with h5py.File(h5_file, 'r') as h5:
            self.length = h5[self.x_key].shape[0]
            self.num_planes = h5[self.x_key].shape[1]
        
    def __len__(self):
        """Return the length of the dataset."""
        return self.length
    
    def __getitem__(self, idx):
        """Get the item at the specified index."""
        # Open file for this specific read operation
        with h5py.File(self.h5_file, 'r') as h5:
            # Load the data from HDF5
            x = h5[self.x_key][idx]
            policy = h5[self.policy_key][idx]
            value = h5[self.value_key][idx]
        
        # Convert to PyTorch tensors
        x = torch.from_numpy(x).float()
        policy = torch.from_numpy(policy).float()
        value = torch.tensor(value).float().unsqueeze(0)  # Add dimension to match model output
        
        return x, policy, value
    
    def close(self):
        """Close the HDF5 file (no-op in this implementation)."""
        pass
            
    def __del__(self):
        """Cleanup (no-op in this implementation)."""
        pass

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train the chess neural network")
    parser.add_argument("--data", type=str, default="data/processed_training_data.h5",
                        help="Path to the HDF5 data file")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the configuration file")
    parser.add_argument("--output-dir", type=str, default="models",
                        help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs to train for")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Initial learning rate")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a checkpoint to resume training from")
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        print(f"Warning: Configuration file {config_path} not found. Using defaults.")
        return {}
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def validate(model, val_loader, criterion_policy, criterion_value, device):
    """Evaluate the model on the validation set."""
    model.eval()
    total_loss = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    
    with torch.no_grad():
        for x, policy_target, value_target in val_loader:
            # Move data to device
            x = x.to(device)
            policy_target = policy_target.to(device)
            value_target = value_target.to(device)
            
            # Forward pass
            policy_logits, value_pred = model(x)
            
            # Apply log_softmax to policy logits
            log_policy = F.log_softmax(policy_logits, dim=1)
            
            # Calculate loss
            policy_loss = criterion_policy(log_policy, policy_target)
            value_loss = criterion_value(value_pred, value_target)
            loss = policy_loss + value_loss
            
            # Update accumulators
            total_loss += loss.item() * x.size(0)
            policy_loss_sum += policy_loss.item() * x.size(0)
            value_loss_sum += value_loss.item() * x.size(0)
    
    # Calculate average loss
    avg_loss = total_loss / len(val_loader.dataset)
    avg_policy_loss = policy_loss_sum / len(val_loader.dataset)
    avg_value_loss = value_loss_sum / len(val_loader.dataset)
    
    model.train()
    return avg_loss, avg_policy_loss, avg_value_loss

def save_checkpoint(model, optimizer, scheduler, epoch, loss, output_dir, is_best=False):
    """Save model checkpoint."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save best model if this is the best so far
    if is_best:
        best_path = os.path.join(output_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
        print(f"Saved best model to {best_path}")

def train_model(
    model, 
    train_loader, 
    val_loader, 
    criterion_policy, 
    criterion_value, 
    optimizer, 
    scheduler, 
    device, 
    epochs, 
    output_dir, 
    start_epoch=0
):
    """Train the model for the specified number of epochs."""
    model.train()
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_policy_loss = 0.0
        train_value_loss = 0.0
        
        # Track time
        start_time = time.time()
        
        # Progress bar
        progress_bar = tqdm(train_loader, desc=f"Training")
        
        for x, policy_target, value_target in progress_bar:
            # Move data to device
            x = x.to(device)
            policy_target = policy_target.to(device)
            value_target = value_target.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            policy_logits, value_pred = model(x)
            
            # Apply log_softmax to policy logits
            log_policy = F.log_softmax(policy_logits, dim=1)
            
            # Calculate loss
            policy_loss = criterion_policy(log_policy, policy_target)
            value_loss = criterion_value(value_pred, value_target)
            loss = policy_loss + value_loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update accumulators
            train_loss += loss.item() * x.size(0)
            train_policy_loss += policy_loss.item() * x.size(0)
            train_value_loss += value_loss.item() * x.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'p_loss': policy_loss.item(),
                'v_loss': value_loss.item()
            })
        
        # Calculate average training loss for the epoch
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_policy_loss = train_policy_loss / len(train_loader.dataset)
        avg_train_value_loss = train_value_loss / len(train_loader.dataset)
        
        # Validation phase
        val_loss, val_policy_loss, val_value_loss = validate(
            model, val_loader, criterion_policy, criterion_value, device
        )
        
        # Adjust learning rate based on validation loss
        if scheduler:
            scheduler.step(val_loss)
        
        # Print epoch stats
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s")
        print(f"  Train Loss: {avg_train_loss:.6f} (Policy: {avg_train_policy_loss:.6f}, Value: {avg_train_value_loss:.6f})")
        print(f"  Val Loss: {val_loss:.6f} (Policy: {val_policy_loss:.6f}, Value: {val_value_loss:.6f})")
        
        # Check if this is the best model so far
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        # Save checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch + 1, val_loss, output_dir, is_best
        )
    
    return model

def main():
    """Main function."""
    # Parse arguments and load configuration
    args = parse_args()
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"Error: Data file {args.data} does not exist.")
        return 1
    
    # Open the HDF5 file to get metadata
    with h5py.File(args.data, 'r') as f:
        num_input_planes = f.attrs.get('num_input_planes', 19)
        policy_vector_size = f.attrs.get('policy_vector_size', 4672)
        # Get actual dimensions of the input data
        input_shape = f['train_x'].shape
        
        print(f"Data shape: {input_shape}")
        print(f"Number of input planes: {num_input_planes}")
        
        # Check if the file has both training and validation data
        has_val_data = 'val_x' in f
        
        if not has_val_data:
            print("Warning: No validation data found in the HDF5 file. Using a portion of training data for validation.")
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = ChessDataset(args.data, 'train')
    val_dataset = ChessDataset(args.data, 'val') if has_val_data else None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
    else:
        # Create a validation set from a portion of the training data
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    # Create or load model
    model = ChessModel(
        num_input_planes=input_shape[1],  # Use actual number of planes from data shape
        policy_output_size=policy_vector_size
    ).to(device)
    
    # Loss functions
    # For policy, use KL divergence between the target distribution and predicted distribution
    criterion_policy = nn.KLDivLoss(reduction='batchmean')
    # For value, use MSE
    criterion_value = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict') and scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resuming from epoch {start_epoch}")
    
    # Train the model
    try:
        trained_model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion_policy=criterion_policy,
            criterion_value=criterion_value,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=args.epochs,
            output_dir=args.output_dir,
            start_epoch=start_epoch
        )
        
        # Save the final model
        final_path = os.path.join(args.output_dir, 'final_model.pt')
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'config': {
                'num_input_planes': num_input_planes,
                'input_planes_shape': input_shape[1],
                'policy_vector_size': policy_vector_size
            }
        }, final_path)
        print(f"Saved final model to {final_path}")
        
        return 0
        
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving checkpoint...")
        save_checkpoint(model, optimizer, scheduler, start_epoch, 0.0, args.output_dir)
        return 1
    
    except Exception as e:
        print(f"Error during training: {e}")
        return 1
    
    finally:
        # Ensure datasets are closed
        if isinstance(train_dataset, ChessDataset):
            train_dataset.close()
        if isinstance(val_dataset, ChessDataset):
            val_dataset.close()

if __name__ == "__main__":
    exit(main()) 