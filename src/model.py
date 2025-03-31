"""
Chess neural network model - policy and value network inspired by AlphaZero.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual block with batch normalization."""
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class ChessModel(nn.Module):
    """
    Neural network for chess, consisting of:
    - Input: board representation as planes
    - ResNet-style convolutional layers
    - Two heads: policy head, value head
    """
    
    def __init__(self, 
                 num_input_planes=19*8, # Default: 19 feature planes × 8 historical positions
                 num_filters=256,       # Number of filters in convolutional layers
                 num_residual_blocks=19, # Number of residual blocks
                 policy_output_size=4672): # Number of possible moves to predict
        super(ChessModel, self).__init__()
        
        # Initial convolutional layer
        self.conv_input = nn.Conv2d(num_input_planes, num_filters, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_filters)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_residual_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, policy_output_size)
        
        # Value head
        self.value_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        """
        Forward pass through the neural network.
        
        Args:
            x: Input tensor with board planes, shape (batch_size, num_input_planes, 8, 8)
            
        Returns:
            tuple: (policy_logits, value) where:
                - policy_logits has shape (batch_size, policy_output_size)
                - value has shape (batch_size, 1) and is in range [-1, 1]
        """
        # Ensure input dimensions match what the model expects
        expected_channels = self.conv_input.in_channels
        if x.shape[1] != expected_channels:
            print(f"info string Input shape mismatch: got {x.shape[1]} channels but expected {expected_channels}")
            
            # Attempt to fix dimension mismatch
            if x.shape[1] < expected_channels:
                # Pad with zeros if we have fewer channels than expected
                padding = torch.zeros(x.shape[0], expected_channels - x.shape[1], 8, 8, device=x.device)
                x = torch.cat([x, padding], dim=1)
            else:
                # Truncate if we have more channels than expected
                x = x[:, :expected_channels, :, :]
        
        # Initial layers
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)  # Flatten
        policy_logits = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)  # Flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # Output in [-1, 1]
        
        return policy_logits, value
        
    def save(self, filepath):
        """Save model weights."""
        torch.save(self.state_dict(), filepath)
        
    def load(self, filepath, map_location=None):
        """Load model weights."""
        checkpoint = torch.load(filepath, map_location=map_location)
        
        # If the loaded object is a dictionary with 'model_state_dict' key, use that
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'])
        # Otherwise try to load it directly as a state dict
        else:
            self.load_state_dict(checkpoint) 