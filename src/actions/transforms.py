import torch
from torchvision import transforms
import torch.nn.functional as F
from typing import Tuple, List

# Based on SlowFast R50 settings from PyTorchVideo
# and common practices for video action recognition.

DEFAULT_MEAN = [0.45, 0.45, 0.45] # PyTorchVideo default for SlowFast
DEFAULT_STD = [0.225, 0.225, 0.225] # PyTorchVideo default for SlowFast
DEFAULT_SIDE_SIZE = 256 # Short side of frame before crop for SlowFast
DEFAULT_CROP_SIZE = 224 # Crop size for SlowFast

class UniformTemporalSubsample:
    """
    Uniformly subsample frames from the video.
    """
    def __init__(self, num_frames: int):
        self.num_frames = num_frames
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Video tensor with shape (C, T, H, W)
        Returns:
            Subsampled video tensor with shape (C, num_frames, H, W)
        """
        if x.shape[1] == self.num_frames:
            return x
        
        # Uniformly sample indices
        indices = torch.linspace(0, x.shape[1] - 1, self.num_frames, dtype=torch.long)
        return x[:, indices, :, :]

class ShortSideScale:
    """
    Scale the short side of the video to a specific size.
    """
    def __init__(self, size: int):
        self.size = size
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Video tensor with shape (C, T, H, W)
        Returns:
            Scaled video tensor
        """
        _, _, h, w = x.shape
        if h < w:
            scale = self.size / h
            new_h = self.size
            new_w = int(w * scale)
        else:
            scale = self.size / w
            new_w = self.size
            new_h = int(h * scale)
        
        # Reshape for interpolation: (C*T, 1, H, W)
        c, t = x.shape[0], x.shape[1]
        x_reshaped = x.reshape(c * t, 1, h, w)
        
        # Interpolate
        x_scaled = F.interpolate(x_reshaped, size=(new_h, new_w), mode='bilinear', align_corners=False)
        
        # Reshape back: (C, T, H, W)
        return x_scaled.reshape(c, t, new_h, new_w)

class ActionTransforms:
    def __init__(
        self,
        num_frames: int = 32, # Corresponds to T in (B, C, T, H, W)
        side_size: int = DEFAULT_SIDE_SIZE,
        crop_size: int = DEFAULT_CROP_SIZE,
        mean: List[float] = DEFAULT_MEAN,
        std: List[float] = DEFAULT_STD,
    ):
        self.num_frames = num_frames
        self.side_size = side_size
        self.crop_size = crop_size
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)
        
        self.temporal_subsample = UniformTemporalSubsample(num_frames)
        self.short_side_scale = ShortSideScale(side_size)

    def __call__(self, video_clip: torch.Tensor) -> torch.Tensor:
        """
        Apply transformations to a video clip.
        Args:
            video_clip (torch.Tensor): A tensor of shape (T, H, W, C) or (C, T, H, W).
                                       Expected to be in uint8 format [0, 255].
                                       If (T, H, W, C), will be permuted.
        Returns:
            torch.Tensor: Transformed clip, typically (C, T, H, W).
        """
        if video_clip.shape[-1] == 3: # If (T, H, W, C)
            video_clip = video_clip.permute(3, 0, 1, 2) # To (C, T, H, W)
        
        # Ensure it's float for division and normalization
        if video_clip.dtype != torch.float32:
            video_clip = video_clip.float()
        
        # Apply transforms
        video_clip = self.temporal_subsample(video_clip)
        video_clip = video_clip / 255.0  # Normalize to [0, 1]
        
        # Normalize with mean and std
        video_clip = (video_clip - self.mean.to(video_clip.device)) / self.std.to(video_clip.device)
        
        # Scale
        video_clip = self.short_side_scale(video_clip)
        
        # Center crop
        _, _, h, w = video_clip.shape
        start_h = (h - self.crop_size) // 2
        start_w = (w - self.crop_size) // 2
        video_clip = video_clip[:, :, start_h:start_h+self.crop_size, start_w:start_w+self.crop_size]
        
        return video_clip

# Example usage:
if __name__ == "__main__":
    # Create a dummy video clip (T, H, W, C)
    # e.g., 64 frames, 240 height, 320 width, 3 channels (RGB)
    dummy_frames_count = 64
    dummy_clip = torch.randint(0, 256, (dummy_frames_count, 240, 320, 3), dtype=torch.uint8)

    # Initialize transforms
    # Requesting 32 frames from the 64 available
    action_tfms = ActionTransforms(num_frames=32, side_size=256, crop_size=224)

    # Apply transforms
    transformed_clip = action_tfms(dummy_clip)

    print(f"Original clip shape: {dummy_clip.shape}")
    print(f"Transformed clip shape: {transformed_clip.shape}") # Expected: (3, 32, 224, 224)

    # Test with C, T, H, W input
    dummy_clip_cthw = dummy_clip.permute(3,0,1,2)
    transformed_clip_cthw = action_tfms(dummy_clip_cthw)
    print(f"Original CTHW clip shape: {dummy_clip_cthw.shape}")
    print(f"Transformed CTHW clip shape: {transformed_clip_cthw.shape}") 