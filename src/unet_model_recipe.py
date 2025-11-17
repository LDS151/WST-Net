import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple



def _haar_filters(device):
    s = 2 ** 0.5
    Lo = torch.tensor([1.0/s, 1.0/s], device=device)
    Hi = torch.tensor([1.0/s, -1.0/s], device=device)
    LL = torch.einsum('i,j->ij', Lo, Lo)
    LH = torch.einsum('i,j->ij', Lo, Hi)
    HL = torch.einsum('i,j->ij', Hi, Lo)
    HH = torch.einsum('i,j->ij', Hi, Hi)
    return LL, LH, HL, HH  # [2,2]
class DWT2D(nn.Module):
    """Haar DWT for each channel independently.
    Input:  x [B,C,H,W]
    Output: LL [B,C,H/2,W/2], highs [B,3C,H/2,W/2] where order is (LH,HL,HH).
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        assert C == self.channels
        # pad to even
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            H += pad_h
            W += pad_w
        device = x.device
        LLk, LHk, HLk, HHk = _haar_filters(device)
        k = torch.stack([LLk, LHk, HLk, HHk], dim=0)  # [4,2,2]
        weight = torch.zeros(4*C, 1, 2, 2, device=device)
        for c in range(C):
            weight[4*c+0, 0] = k[0]
            weight[4*c+1, 0] = k[1]
            weight[4*c+2, 0] = k[2]
            weight[4*c+3, 0] = k[3]
        y = F.conv2d(x, weight=weight, bias=None, stride=2, padding=0, groups=C)  # [B,4C,H/2,W/2]
        LL  = y[:, 0::4, :, :]
        LH  = y[:, 1::4, :, :]
        HL  = y[:, 2::4, :, :]
        HH  = y[:, 3::4, :, :]
        highs = torch.cat([LH, HL, HH], dim=1)  # [B,3C,H/2,W/2]
        return LL, highs

class IWT2D(nn.Module):
    """Inverse Haar DWT.
    Input:  LL [B,C,H,W], highs [B,3C,H,W] (order: LH, HL, HH)
    Output: x_up [B,C,2H,2W]
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

    def forward(self, LL: torch.Tensor, highs: torch.Tensor) -> torch.Tensor:
        B, C, H, W = LL.shape
        assert C == self.channels
        assert highs.shape[1] == 3*C
        LH, HL, HH = torch.split(highs, C, dim=1)
        device = LL.device
        # pack 4C input for transposed conv
        x = torch.zeros(B, 4*C, H, W, device=device)
        x[:, 0::4, :, :] = LL
        x[:, 1::4, :, :] = LH
        x[:, 2::4, :, :] = HL
        x[:, 3::4, :, :] = HH
        LLk, LHk, HLk, HHk = _haar_filters(device)
        k = torch.stack([LLk, LHk, HLk, HHk], dim=0)  # [4,2,2]
        # build transposed conv weight that maps per-group 4->1 channel, groups=C
        # reshape weight to [C, 4, 2, 2] then to [C, 4, 2, 2] per group via conv_transpose2d
        # PyTorch expects weight [in_channels, out_channels/groups, kH, kW] for groups>1
        # We'll implement with grouped conv_transpose by reshaping tensors.
        # Trick: split groups and apply conv_transpose per group via F.conv_transpose2d in a loop (C is small), then cat.
        outs = []
        for g in range(C):
            xg = x[:, 4*g:4*g+4, :, :]  # [B,4,H,W]
            wg = k.view(4, 1, 2, 2)     # [4,1,2,2]
            yg = F.conv_transpose2d(xg, weight=wg, stride=2, padding=0)  # [B,1,2H,2W]
            outs.append(yg)
        y = torch.cat(outs, dim=1)  # [B,C,2H,2W]
        return y


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)



class EncoderBlockWave(nn.Module):
    """Encoder: DoubleConv + Wavelet DWT pooling.
    Returns: pooled_LL, skip_conv, wave_cache(highs)
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.dwt = DWT2D(out_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        conv = self.double_conv(x)             # [B, C, H, W]
        LL, highs = self.dwt(conv)             # LL:[B,C,H/2,W/2], highs:[B,3C,H/2,W/2]
        return LL, conv, highs

class DecoderBlockWave(nn.Module):
    """Decoder: channel align -> IWT upsample -> concat skip -> DoubleConv.
    We align decoder channels to the encoder level channels so IWT can be applied.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        # Map current feature to LL channels required by IWT (== skip_channels)
        self.align = nn.Conv2d(in_channels, skip_channels, kernel_size=1, stride=1)
        self.iwt = IWT2D(skip_channels)
        self.double_conv = DoubleConv(skip_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, highs: torch.Tensor) -> torch.Tensor:
        x_ll = self.align(x)                  # [B, skip_C, H/2, W/2]
        x_up = self.iwt(x_ll, highs)          # [B, skip_C, H, W]
        if x_up.shape[-2:] != skip.shape[-2:]:
            diff_y = skip.shape[-2] - x_up.shape[-2]
            diff_x = skip.shape[-1] - x_up.shape[-1]
            x_up = F.pad(
                x_up,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        x_cat = torch.cat([x_up, skip], dim=1)
        return self.double_conv(x_cat)



class CrossAttentionBridge(nn.Module):
    """Cross-attend bridge features with conditioning tokens."""

    def __init__(
        self,
        bridge_channels: int,
        conditioning_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.query_proj = nn.Linear(bridge_channels, bridge_channels)
        self.key_proj = nn.Linear(conditioning_dim, bridge_channels)
        self.value_proj = nn.Linear(conditioning_dim, bridge_channels)
        self.attn = nn.MultiheadAttention(
            bridge_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(bridge_channels)
        self.output_proj = nn.Linear(bridge_channels, bridge_channels)

    def forward(self, bridge: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = bridge.shape
        spatial_tokens = bridge.view(batch, channels, height * width).permute(0, 2, 1)

        expected = self.key_proj.in_features  # = conditioning_dim
        if conditioning.dim() == 2:
          
            conditioning_tokens = conditioning.unsqueeze(-1) if expected == 1 else conditioning.unsqueeze(1)
        elif conditioning.dim() == 3:
            conditioning_tokens = conditioning
        else:
            raise ValueError("conditioning tensor must be rank-2 or rank-3.")

       
        if conditioning_tokens.size(-1) != expected:
            if conditioning_tokens.size(1) == expected and conditioning_tokens.size(-1) == 1:
                conditioning_tokens = conditioning_tokens.transpose(1, 2)  # (B, 1, D) <-> (B, D, 1)
            else:
                raise ValueError(
                    f"Expected conditioning last-dim {expected}, got {tuple(conditioning_tokens.shape)}."
                )

        queries = self.query_proj(spatial_tokens)
        keys = self.key_proj(conditioning_tokens)
        values = self.value_proj(conditioning_tokens)

        attn_output, _ = self.attn(queries, keys, values)
        attn_output = self.output_proj(attn_output)
        fused = self.norm(spatial_tokens + attn_output)

        fused = fused.permute(0, 2, 1).view(batch, channels, height, width)
        return fused


class UNetWavelet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        starting_filters: int = 32,
        conditioning_dim: Optional[int] = None,
        bridge_heads: int = 4,
        bridge_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        filters = starting_filters
        self.conditioning_dim = conditioning_dim

        # Encoders with wavelet pooling
        self.encoder1 = EncoderBlockWave(in_channels, filters)
        self.encoder2 = EncoderBlockWave(filters, filters * 2)
        self.encoder3 = EncoderBlockWave(filters * 2, filters * 4)
        self.encoder4 = EncoderBlockWave(filters * 4, filters * 8)

        # Bottleneck (keep as DoubleConv)
        self.bottleneck = DoubleConv(filters * 8, filters * 16)

        # Decoders with IWT upsampling
        self.decoder1 = DecoderBlockWave(filters * 16, filters * 8, filters * 8)
        self.decoder2 = DecoderBlockWave(filters * 8,  filters * 4, filters * 4)
        self.decoder3 = DecoderBlockWave(filters * 4,  filters * 2, filters * 2)
        self.decoder4 = DecoderBlockWave(filters * 2,  filters,      filters)

        self.final_conv = nn.Conv2d(filters, out_channels, kernel_size=1)

        if conditioning_dim is not None:
            self.bridge_attn = CrossAttentionBridge(
                bridge_channels=filters * 16,
                conditioning_dim=conditioning_dim,
                num_heads=bridge_heads,
                dropout=bridge_dropout,
            )
        else:
            self.bridge_attn = None

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        elif isinstance(module, nn.GroupNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")

    def _apply_bridge_attention(self, bridge: torch.Tensor, conditioning: Optional[torch.Tensor]) -> torch.Tensor:
        if self.bridge_attn is None or conditioning is None:
            return bridge

        if conditioning.dim() not in (2, 3):
            raise ValueError("conditioning tensor must be rank-2 or rank-3.")
        if conditioning.size(0) != bridge.size(0):
            raise ValueError("Batch size mismatch between image features and conditioning tokens.")

        expected_dim = self.conditioning_dim

        
        if conditioning.dim() == 2 and expected_dim == 1:
            conditioning = conditioning.unsqueeze(-1)            # (B, D) -> (B, D, 1)
        elif conditioning.dim() == 3 and conditioning.size(-1) != expected_dim:
            if conditioning.size(1) == expected_dim and conditioning.size(-1) == 1:
                conditioning = conditioning.transpose(1, 2)      # (B, 1, D) -> (B, D, 1)
            else:
                raise ValueError(f"Expected conditioning dim {expected_dim}, got {conditioning.size(-1)}.")

        return self.bridge_attn(bridge, conditioning)

    def forward(self, x: torch.Tensor, conditioning: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encoder: collect LL for next stage, skip for concat, highs for IWT
        enc1_LL, enc1_skip, enc1_highs = self.encoder1(x)
        enc2_LL, enc2_skip, enc2_highs = self.encoder2(enc1_LL)
        enc3_LL, enc3_skip, enc3_highs = self.encoder3(enc2_LL)
        enc4_LL, enc4_skip, enc4_highs = self.encoder4(enc3_LL)

        bridge_raw = self.bottleneck(enc4_LL)
        bridge = self._apply_bridge_attention(bridge_raw, conditioning)

        # Decoder: IWT upsample with stored highs from corresponding encoder level
        dec1 = self.decoder1(bridge, enc4_skip, enc4_highs)
        dec2 = self.decoder2(dec1,  enc3_skip, enc3_highs)
        dec3 = self.decoder3(dec2,  enc2_skip, enc2_highs)
        dec4 = self.decoder4(dec3,  enc1_skip, enc1_highs)

        output = self.final_conv(dec4)
        return torch.sigmoid(output), bridge_raw


def unet_model(
    starting_filters: int = 32,
    in_channels: int = 3,
    out_channels: int = 1,
    device: Optional[torch.device] = None,
    conditioning_dim: Optional[int] = None,
    bridge_heads: int = 4,
    bridge_dropout: float = 0.0,
) -> UNetWavelet:
    model = UNetWavelet(
        in_channels=in_channels,
        out_channels=out_channels,
        starting_filters=starting_filters,
        conditioning_dim=conditioning_dim,
        bridge_heads=bridge_heads,
        bridge_dropout=bridge_dropout,
    )
    if device is not None:
        model = model.to(device)
    return model
