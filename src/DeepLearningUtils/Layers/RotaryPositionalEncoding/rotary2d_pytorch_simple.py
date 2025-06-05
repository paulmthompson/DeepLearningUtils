"""
Simplified PyTorch implementation of 2D Rotary Positional Encoding.

This implementation serves as ground truth for comparison with the Keras version.
It focuses specifically on 2D spatial rotary embeddings without external dependencies.

Based on rotary embedding principles but simplified for 2D spatial use case.
"""

from typing import Tuple
import torch
import torch.nn as nn
import math


class RotaryPositionalEncoding2DPyTorch(nn.Module):
    """
    Simplified PyTorch implementation of 2D Rotary Positional Encoding.
    
    This serves as ground truth for comparison with the Keras implementation.
    Focuses specifically on 2D spatial embeddings without external dependencies.
    """
    
    def __init__(
        self,
        dim: int,
        height: int,
        width: int,
        theta: float = 10.0,
        rotate: bool = True,
        max_freq: int = 64
    ) -> None:
        """
        Initialize PyTorch 2D Rotary Positional Encoding.
        
        Args:
            dim: Feature dimension (must be even).
            height: Height of 2D spatial grid.
            width: Width of 2D spatial grid.
            theta: Theta parameter for frequency computation.
            rotate: Whether to apply random rotations.
            max_freq: Maximum frequency (currently unused, for API compatibility).
        """
        super().__init__()
        
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        
        self.dim = dim
        self.height = height
        self.width = width
        self.theta = theta
        self.rotate = rotate
        self.max_freq = max_freq
        
        # Initialize frequency components
        self.register_buffer('freqs', self._init_2d_freqs())

    def _init_2d_freqs(self) -> torch.Tensor:
        """
        Initialize 2D frequency components.
        
        Returns:
            Frequency tensor with shape (2, dim//2).
        """
        freq_dim = self.dim // 2
        
        # Compute base magnitudes using theta scaling
        mag = 1.0 / (self.theta ** (torch.arange(0, freq_dim, dtype=torch.float32) * 2.0 / self.dim))
        
        # Apply random rotation if enabled
        if self.rotate:
            angles = torch.rand(1) * 2 * math.pi
        else:
            angles = torch.zeros(1)
        
        # Compute frequency components for x and y directions
        fx = mag * torch.cos(angles)
        fy = mag * torch.sin(angles)
        
        freqs = torch.stack([fx, fy], dim=0)
        return freqs

    def _init_spatial_coordinates(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize spatial coordinate arrays for the 2D grid.
        
        Returns:
            Tuple of (t_x, t_y) coordinate tensors.
        """
        # Create linear sequence for flattened spatial positions
        spatial_size = self.height * self.width
        t = torch.arange(spatial_size, dtype=torch.float32)
        
        # Convert to 2D coordinates
        t_x = t % self.width  # x coordinate
        t_y = torch.floor(t / self.width)  # y coordinate
        
        return t_x, t_y

    def _compute_rotation_matrix(self, t_x: torch.Tensor, t_y: torch.Tensor) -> torch.Tensor:
        """
        Compute complex rotation matrix for rotary embeddings.
        
        Args:
            t_x: X-coordinates tensor.
            t_y: Y-coordinates tensor.
            
        Returns:
            Complex rotation matrix.
        """
        # Compute frequency products for x and y directions
        freqs_x = t_x.unsqueeze(-1) @ self.freqs[0:1]  # Shape: (spatial_size, freq_dim)
        freqs_y = t_y.unsqueeze(-1) @ self.freqs[1:2]  # Shape: (spatial_size, freq_dim)
        
        # Combine frequency components
        freq_sum = freqs_x + freqs_y
        
        # Convert to complex exponentials
        rotation_matrix = torch.complex(torch.cos(freq_sum), torch.sin(freq_sum))
        
        return rotation_matrix

    def _apply_rotary_embedding(
        self, 
        q: torch.Tensor, 
        rotation_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply rotary embeddings to query tensor using rotation matrix.
        
        Args:
            q: Query tensor with shape (batch_size, seq_len, height*width, dim).
            rotation_matrix: Complex rotation matrix.
            
        Returns:
            Rotated query tensor with same shape as input.
        """
        batch_size, seq_len, hw, dim = q.shape
        
        # Ensure float32 for complex operations
        q = q.float()
        
        # Reshape to separate real/imaginary components
        q_reshaped = q.view(batch_size, seq_len, hw, dim // 2, 2)
        
        # Create complex representation
        q_complex = torch.complex(q_reshaped[..., 0], q_reshaped[..., 1])
        
        # Apply rotation - broadcasting rotation_matrix across batch and sequence dimensions
        q_rotated = q_complex * rotation_matrix.unsqueeze(0).unsqueeze(0)
        
        # Extract real and imaginary parts
        q_rotated_real = torch.real(q_rotated)
        q_rotated_imag = torch.imag(q_rotated)
        
        # Recombine into original format
        q_rotated = torch.stack([q_rotated_real, q_rotated_imag], dim=-1)
        q_rotated = q_rotated.view(batch_size, seq_len, hw, dim)
        
        return q_rotated

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """
        Apply 2D rotary positional encoding to query tensor.
        
        Args:
            q: Query tensor with shape (batch_size, seq_len, height*width, dim).
            
        Returns:
            Query tensor with rotary positional encoding applied.
        """
        if q.dim() != 4:
            raise ValueError(
                f"Expected 4D query tensor (batch_size, seq_len, height*width, dim), "
                f"got {q.dim()}D tensor with shape {q.shape}"
            )
        
        # Initialize spatial coordinates
        t_x, t_y = self._init_spatial_coordinates()
        t_x = t_x.to(q.device)
        t_y = t_y.to(q.device)
        
        # Compute rotation matrix
        rotation_matrix = self._compute_rotation_matrix(t_x, t_y)
        
        # Apply rotary embeddings
        q_rotated = self._apply_rotary_embedding(q, rotation_matrix)
        
        return q_rotated 