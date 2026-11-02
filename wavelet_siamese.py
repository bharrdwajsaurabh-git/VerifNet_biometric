"""
Wavelet-based Siamese Neural Network for VerifNet.
Implements:
1. 2D Scattering Wavelet Transform (SWT) front-end using Morlet wavelets (J=2, L=8).
2. Lightweight Convolutional Neural Network (Conv2D filters: 4, 8, 8 with 8x8 kernels).
3. 16-dimensional embedding head.
4. Siamese network matching using Euclidean distance.
5. Contrastive Loss training framework (Eq 2 in the paper).
"""

# Monkey-patch scipy.special for kymatio compatibility with recent SciPy versions
import scipy.special
if not hasattr(scipy.special, 'sph_harm'):
    scipy.special.sph_harm = lambda *args, **kwargs: None

import torch
import torch.nn as nn
import torch.nn.functional as F
from kymatio.torch import Scattering2D


class VerifNetSiamese(nn.Module):
    def __init__(self, img_size=(128, 128), J=2, L=8, embedding_dim=16):
        super().__init__()
        self.img_size = img_size
        self.J = J
        self.L = L
        self.embedding_dim = embedding_dim

        # 1. 2D Scattering Wavelet Transform Layer
        self.scattering = Scattering2D(J=self.J, shape=self.img_size, L=self.L)
        in_channels = 1 + self.J * self.L + (self.J * (self.J - 1) // 2) * (self.L ** 2)

        # 2. Convolutional Layers (filters: 4, 8, 8 with kernel size 8x8 as per paper Table VI)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=8, padding=4),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=8, padding=4),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=8, padding=4),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # Determine flatten dimension dynamically
        dummy_in = torch.zeros(1, in_channels, self.img_size[0] // (2 ** self.J), self.img_size[1] // (2 ** self.J))
        with torch.no_grad():
            dummy_feat = self.conv3(self.conv2(self.conv1(dummy_in)))
            flatten_dim = dummy_feat.numel()

        # 3. Dense & Dropout Regularization Layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.embedding_dim)
        )

    def extract_scattering(self, x):
        """
        Passes input through Scattering Wavelet Transform.
        Input x: (B, 1, H, W) or (1, H, W)
        Output: (B, C, H', W')
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        scat = self.scattering(x)
        if scat.dim() == 5:
            scat = scat.squeeze(1)
        return scat

    def forward_embedding_from_scattering(self, scat):
        """Extracts 16-D normalized embedding directly from scattering features."""
        feat = self.conv1(scat)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        emb = self.fc(feat)
        return F.normalize(emb, p=2, dim=1)

    def forward_one(self, x):
        """Extracts 16-D normalized embedding for a single image."""
        scat = self.extract_scattering(x)
        return self.forward_embedding_from_scattering(scat)

    def forward(self, x1, x2):
        """
        Forward pass for a pair of inputs.
        Accepts either image tensors (B, 1, H, W) or precomputed scattering tensors (B, C, H', W').
        """
        if x1.dim() == 4 and x1.size(1) == 1:
            emb1 = self.forward_one(x1)
            emb2 = self.forward_one(x2)
        else:
            emb1 = self.forward_embedding_from_scattering(x1)
            emb2 = self.forward_embedding_from_scattering(x2)
            
        dw = torch.norm(emb1 - emb2, p=2, dim=1)
        return emb1, emb2, dw

    def compute_similarity(self, x1, x2, margin=1.5):
        """
        Calculates similarity score in range [0, 1].
        S_DL = max(0, 1 - Dw / margin)
        """
        self.eval()
        with torch.no_grad():
            _, _, dw = self.forward(x1, x2)
            sim = torch.clamp(1.0 - (dw / margin), min=0.0, max=1.0)
            return sim.cpu().numpy()


class ContrastiveLoss(nn.Module):
    """
    Distance-aware contrastive loss as defined in Eq (2) of the paper:
    L = 0.5 * Y * (Dw)^2 + 0.5 * (1 - Y) * (max(0, m - Dw))^2
    where Y=1 represents genuine pairs (same person), Y=0 represents imposter pairs.
    """
    def __init__(self, margin=1.5):
        super().__init__()
        self.margin = margin

    def forward(self, dw, target):
        loss_genuine = target * 0.5 * torch.pow(dw, 2)
        loss_imposter = (1.0 - target) * 0.5 * torch.pow(torch.clamp(self.margin - dw, min=0.0), 2)
        return torch.mean(loss_genuine + loss_imposter)
