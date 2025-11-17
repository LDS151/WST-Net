import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DynamicResidualFusion(nn.Module):
    """
    Dynamic Residual Fusion Module

    Args:
        clinical_dim: int = 7,
        image_dim: int = 128,
        fusion_dim: int = 256,
        conv_kernel_size: int = 3,
        dropout: float = 0.1
    """
    
    def __init__(
        self,
        clinical_dim: int = 7,
        image_dim: int = 128,
        fusion_dim: int = 256,
        conv_kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.clinical_dim = clinical_dim
        self.image_dim = image_dim
        self.fusion_dim = fusion_dim
        
        # Dimensionality alignment projection layer
        self.clinical_proj = nn.Linear(clinical_dim, fusion_dim)
        self.image_proj = nn.Linear(image_dim, fusion_dim)
        
        # 1D convolution processing after Cross Concat
        self.conv1d = nn.Conv1d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size // 2
        )
        
        # Dynamic weighting network
        self.weight_net = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 4, 2),
            nn.Softmax(dim=1)  # Ensure ω1 + ω2 = 1
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
    def cross_concat(self, clinical_feat: torch.Tensor, image_feat: torch.Tensor) -> torch.Tensor:
        """
        Cross Concat operation: Alternately fill two modal features

        Args:
            clinical_feat: [B, fusion_dim] clinical features
            image_feat: [B, fusion_dim] image features

        Returns:
            cross_concat_feat: [B, fusion_dim * 2] Alternating concatenated features
        """
        B = clinical_feat.size(0)
        output = torch.zeros(B, self.fusion_dim * 2, device=clinical_feat.device)
        
        # Fill image features with even indices, clinical features with odd indices
        output[:, 0::2] = image_feat      # O[0::2] = f_I(I)
        output[:, 1::2] = clinical_feat   # O[1::2] = f_t(T)
        
        return output
    
    def forward(self, clinical_feat: torch.Tensor, image_feat: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation

        Args:
            clinical_feat: [B, clinical_dim] original clinical features
            image_feat: [B, image_dim] original image features

        Returns:
            fused_feat: [B, fusion_dim] fused features
        """
        # 1. Dimensionality alignment projection
        clinical_proj = self.clinical_proj(clinical_feat)  # [B, fusion_dim]
        image_proj = self.image_proj(image_feat)           # [B, fusion_dim]
        
        # 2. Cross Concat operation
        cross_feat = self.cross_concat(clinical_proj, image_proj)  # [B, fusion_dim * 2]
        
        # 3. 1D convolution processing + residual connection
        # Reshape features to [B, 1, fusion_dim * 2] for 1D convolution
        conv_input = cross_feat.unsqueeze(1)  # [B, 1, fusion_dim * 2]
        conv_output = self.conv1d(conv_input).squeeze(1)  # [B, fusion_dim * 2]
        residual_feat = conv_output + cross_feat  # Residual connection
        
        # 4. Dynamic weight calculation
        weight_input = torch.cat([clinical_proj, image_proj], dim=1)  # [B, fusion_dim * 2]
        weights = self.weight_net(weight_input)  # [B, 2], [ω1, ω2]
        w1, w2 = weights[:, 0:1], weights[:, 1:2]  # [B, 1] each
        
        # 5. Weighted fusion: m = ω1 · (Conv(O) + O) + ω2 · f_I(I)
        # Here we pool residual_feat to get the fusion_dim dimension
        pooled_residual = F.adaptive_avg_pool1d(
            residual_feat.unsqueeze(1), self.fusion_dim
        ).squeeze(1)  # [B, fusion_dim]
        
        fused_feat = w1 * pooled_residual + w2 * image_proj  # [B, fusion_dim]
        
        # 6. Output projection
        output = self.output_proj(fused_feat)
        
        return output


class DRFWrapper(nn.Module):
    """
    Wrapper for DRF module to replace existing simple concatenation
    Maintains compatibility with existing ClinicalMLP interface
    """
    
    def __init__(
        self,
        clinical_dim: int = 7,
        image_dim: int = 128,
        fusion_dim: int = 256,
        clinical_mlp_aux_dim: int = 0
    ):
        super().__init__()
        
        self.drf = DynamicResidualFusion(
            clinical_dim=clinical_dim,
            image_dim=image_dim,
            fusion_dim=fusion_dim
        )
        
        # Update the input dimension of ClinicalMLP
        from .clinical_model import ClinicalMLP
        self.clinical_head = ClinicalMLP(aux_features_dim=fusion_dim)
        
    def forward(self, clinical_feat: torch.Tensor, image_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clinical_feat: [B, clinical_dim] clinical features
            image_feat: [B, image_dim] image features

        Returns:
            logits: [B] classification logits
        """
        # DRF fusion
        fused_feat = self.drf(clinical_feat, image_feat)  # [B, fusion_dim]
        
        # Construct the input format for ClinicalMLP [basic clinical features; fused features]
        # Here we use the first 2 dimensions as basic clinical features, and the rest as aux features
        clinical_base = clinical_feat[:, :2]  # [B, 2] - Keep the original NIHSS features
        clinical_input = torch.cat([clinical_base, fused_feat], dim=1)  # [B, 2 + fusion_dim]
        
        logits = self.clinical_head(clinical_input)
        return logits
