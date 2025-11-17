import torch
from torch import nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class SliceMamba(nn.Module):
    """
    Cross-slice Mamba image feature encoder
    --------------------------------
    Input:
        dec4: [B, C_in, H, W] - Features of multiple slices from the same patient
    Output:
        out: [out_channels] - Patient-level global image feature vector
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int = 16,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_ln: bool = True,
        pos_max_hw: int = 1024,
        max_slices: int = 1024,        # Maximum number of slices per patient
        merge_batch_as_sequence: bool = True,  # Enable cross-slice modeling
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d_model = d_model
        self.merge_batch_as_sequence = merge_batch_as_sequence
        self.pos_max_hw = pos_max_hw

        # Patch embedding (Conv2d)
        self.patch_embed = nn.Conv2d(in_channels, d_model,
                                     kernel_size=patch_size, stride=patch_size, bias=True)

        # 2D position encoding (row, column)
        self.row_emb = nn.Embedding(pos_max_hw, d_model)
        self.col_emb = nn.Embedding(pos_max_hw, d_model)

        # Slice position encoding
        self.slice_emb = nn.Embedding(max_slices, d_model)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.pre_ln = nn.LayerNorm(d_model) if use_ln else nn.Identity()

        # Mamba model
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model) if use_ln else nn.Identity(),
            nn.Linear(d_model, out_channels),
        )

  
    @staticmethod
    def _pad_to_divisible(x: torch.Tensor, div: int):
        B, C, H, W = x.shape
        pad_h = (div - H % div) % div
        pad_w = (div - W % div) % div
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        if pad_h or pad_w:
            x = F.pad(x, (left, right, top, bottom))
        return x, (left, right, top, bottom)


    def _flatten_with_2d_pos(self, x: torch.Tensor) -> torch.Tensor:
        B, D, Hp, Wp = x.shape
        device = x.device
        assert Hp < self.pos_max_hw and Wp < self.pos_max_hw, \
            f"Position encoding out of bounds: Hp={Hp}, Wp={Wp}, pos_max_hw={self.pos_max_hw}"

        tokens = x.permute(0, 2, 3, 1).contiguous().view(B, Hp * Wp, D)

        rows = torch.arange(Hp, device=device)
        cols = torch.arange(Wp, device=device)
        grid_r = rows.unsqueeze(1).expand(Hp, Wp).reshape(-1)
        grid_c = cols.unsqueeze(0).expand(Hp, Wp).reshape(-1)
        pos = self.row_emb(grid_r) + self.col_emb(grid_c)
        pos = pos.unsqueeze(0).expand(B, -1, -1)

        return tokens + pos  # [B, T, D]

    def forward(self, dec4: torch.Tensor) -> torch.Tensor:
        B, C, H, W = dec4.shape
        assert C == self.in_channels, f"in_channels={self.in_channels}, Input C={C}"

   
        x, _ = self._pad_to_divisible(dec4, self.patch_size)
        x = self.patch_embed(x)  # [B, D, H', W']

        
        tokens = self._flatten_with_2d_pos(x)  # [B, T, D]

      
        device = tokens.device
        B_, T, D = tokens.shape
        assert B_ == B
    
        slice_ids = torch.arange(B, device=device)
        slice_pos = self.slice_emb(slice_ids).unsqueeze(1).expand(B, T, D)
        tokens = tokens + slice_pos


        tokens = tokens.reshape(1, B * T, D)  # [1, B*T, D]
        tokens = self.pre_ln(tokens)
        cls = self.cls_token.expand(1, 1, D)
        seq = torch.cat([cls, tokens], dim=1)  # [1, 1+B*T, D]
        seq = self.mamba(seq)                  # [1, 1+B*T, D]
        z = seq[:, 0, :]                       # [1, D]
        out = self.head(z)                     # [1, out_channels]
        return out                  # [out_channels]

