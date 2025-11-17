import torch
from torch import nn
import torch.nn.functional as F


class ClinicalMLP(nn.Module):
    """
    Advanced clinical/multimodal binary classification head (compatible with old interface)
    ------------------------------------------------
    Input: [B, D], usually a concatenation of [clinical vector; slice global vector]
    Branch:
      1) MLP branch: keep the original implementation
      2) Transformer branch: map the global vector to M "pseudo tokens" for self-attention modeling
    Fusion:
      Adaptive gating, dynamically weighting the two logit paths by sample
    Remarks:
      - Constructor parameters and forward shape are consistent with the old version; external training code does not need to be modified
      - Works robustly even with ultra-small samples/dimensions
    """

    def __init__(
        self,
        branch_hidden_dim: int = 256,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.1,
        aux_features_dim: int = 0,
    ) -> None:
        super().__init__()
        # Still use the old input dimension definition: 2 (e.g., two NIHSS items) + aux_features_dim
        self.aux_features_dim = aux_features_dim
        self.input_dim = 2 + aux_features_dim


        self.encoder = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, branch_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(branch_hidden_dim, branch_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.mlp_head = nn.Sequential(
            nn.Linear(branch_hidden_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

 
        # Token dimension and quantity (can be fine-tuned as needed, lightweight and stable by default)
        d_model = 256
        m_tokens = 8

        self.pre_ln = nn.LayerNorm(self.input_dim)
        self.token_expand = nn.Sequential(
            nn.Linear(self.input_dim, m_tokens * d_model),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )
        self.pos_emb = nn.Parameter(torch.randn(1, m_tokens, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # Output shape [B, L, D]
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.trans_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )


        self.gate = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, clinical: torch.Tensor) -> torch.Tensor:
        # Shape check: keep consistent with the old version
        if clinical.dim() != 2:
            raise ValueError("The clinical tensor must be two-dimensional, in the form of [batch_size, feature_dim].")
        if clinical.size(1) != self.input_dim:
            raise ValueError(f"The clinical feature dimension should be {self.input_dim}, but it is actually {clinical.size(1)}.")

        B = clinical.size(0)

        enc = self.encoder(clinical)               # [B, H]
        logits_mlp = self.mlp_head(enc)            # [B, 1]

        x = self.pre_ln(clinical)                  # [B, D_in]
        tok = self.token_expand(x)                 # [B, M*D]
        # Generate M pseudo tokens
        M = self.pos_emb.size(1)
        Dm = self.pos_emb.size(2)
        tok = tok.view(B, M, Dm) + self.pos_emb    # [B, M, Dm]
        tok = self.transformer(tok)                # [B, M, Dm]
        pooled = tok.mean(dim=1)                   # [B, Dm]
        logits_trans = self.trans_head(pooled)     # [B, 1]

        alpha = self.gate(clinical)                # [B, 1], the closer to 1, the more biased towards the Transformer
        logits = alpha * logits_trans + (1 - alpha) * logits_mlp
        return logits.squeeze(-1)


def build_clinical_classifier(aux_features_dim: int = 0) -> ClinicalMLP:
    """Quickly build a clinical feature classification model using the default structure."""
    return ClinicalMLP(aux_features_dim=aux_features_dim)