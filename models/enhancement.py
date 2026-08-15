import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import logging

# ==========================================
# 1. FUnIE-GAN Generator Architecture
# ==========================================
class UNetDown(nn.Module):
    """Downsampling block for the Generator"""
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size, 0.8))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    """Upsampling block for the Generator"""
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_size, 0.8),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        # Concatenate the skip connection from the downsampling path
        x = torch.cat((x, skip_input), 1)
        return x

class Generator(nn.Module):
    """FUnIE-GAN Generator Network"""
    def __init__(self, in_channels=3, out_channels=3):
        super(Generator, self).__init__()

        self.down1 = UNetDown(in_channels, 32, normalize=False)
        self.down2 = UNetDown(32, 64)
        self.down3 = UNetDown(64, 128)
        self.down4 = UNetDown(128, 256, dropout=0.5)
        self.down5 = UNetDown(256, 256, dropout=0.5)

        self.up1 = UNetUp(256, 256, dropout=0.5)
        self.up2 = UNetUp(512, 128)
        self.up3 = UNetUp(256, 64)
        self.up4 = UNetUp(128, 32)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, out_channels, kernel_size=4, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)

        return self.final(u4)

# ==========================================
# 2. Inference Wrapper Class
# ==========================================
class ImageEnhancer:
    def __init__(self, model_path: str):
        """
        Initializes the FUnIE-GAN model for real-time inference.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Initializing FUnIE-GAN Enhancement module on: {self.device}")
        
        # Initialize architecture
        self.model = Generator().to(self.device)
        
        # Load weights if the file exists
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            logging.info(f"Successfully loaded FUnIE-GAN weights from {model_path}")
        else:
            logging.warning(f"Weights not found at {model_path}. Model will output noise. Please download the .pth file.")

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """
        Converts an OpenCV BGR frame to a normalized PyTorch tensor.
        FUnIE-GAN expects 256x256 inputs normalized between [-1, 1].
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to standard GAN input size
        img_resized = cv2.resize(img_rgb, (256, 256))
        
        # Normalize to [0, 1] then to [-1, 1]
        img_normalized = (img_resized / 255.0 - 0.5) / 0.5
        
        # Convert to tensor: HWC -> CHW format
        img_tensor = torch.from_numpy(img_normalized).float().permute(2, 0, 1)
        
        # Add batch dimension [1, C, H, W]
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        return img_tensor

    def _postprocess(self, tensor: torch.Tensor, original_shape: tuple) -> np.ndarray:
        """
        Converts the GAN output tensor back to an OpenCV BGR frame.
        """
        # Remove batch dimension and move to CPU
        img_tensor = tensor.squeeze(0).cpu().detach()
        
        # Denormalize from [-1, 1] to [0, 1]
        img_denormalized = (img_tensor + 1.0) / 2.0
        
        # Convert CHW -> HWC and scale to [0, 255]
        img_np = img_denormalized.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
        
        # Convert RGB back to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Restore original dimensions for the detection pipeline
        img_restored = cv2.resize(img_bgr, (original_shape[1], original_shape[0]))
        return img_restored

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Public method to process a single frame through the GAN.
        """
        original_shape = frame.shape
        
        # 1. Preprocess
        input_tensor = self._preprocess(frame)
        
        # 2. Inference (no gradients needed for speed)
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
            
        # 3. Postprocess
        enhanced_frame = self._postprocess(output_tensor, original_shape)
        
        return enhanced_frame
