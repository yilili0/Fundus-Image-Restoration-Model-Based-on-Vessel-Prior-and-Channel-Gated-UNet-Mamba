import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from torch.utils.checkpoint import checkpoint
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


# =========================================================================
# 1. 基础组件 (LayerNorm & FeedForward)
# =========================================================================
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        x_3d = rearrange(x, 'b c h w -> b (h w) c')
        out_3d = self.body(x_3d)
        return rearrange(out_3d, 'b (h w) c -> b c h w', h=h, w=w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


# =========================================================================
# 2. 核心：更严谨的 SS2D_Substitute_v2（官方 Mamba 实现 2D 四向扫描）
# =========================================================================
class ChannelGate(nn.Module):
    """
    保守的通道门控（SE-like）。对眼底恢复更稳：减少“造纹理”风险。
    """
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(dim // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=False),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SS2D_Substitute_v2(nn.Module):
    """
    2D 四向扫描替代块（仅依赖官方 mamba_ssm）：
      - 水平 forward/backward
      - 垂直 forward/backward
    使用 concat + Linear 投影融合，并保留 x/z 分支门控。

    share_mode:
        - "all": 四向共享同一个 Mamba
        - "hv" : 水平一套、垂直一套（推荐）
        - "none": 四向各自一套
    """
    def __init__(
        self,
        d_model: int = 48,
        expand: int = 1,            # 为了与你原来 ssm_ratio=1.0 的参数规模更接近，默认 expand=1
        d_state: int = 16,
        d_conv: int = 2,
        bias: bool = False,
        share_mode: str = "hv",
        gate: bool = True,
        gate_reduction: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert share_mode in ("all", "hv", "none")

        if Mamba is None:
            raise ImportError("mamba_ssm not found. Install: pip install mamba-ssm")

        self.d_model = d_model
        self.inner_dim = d_model * expand
        self.share_mode = share_mode
        self.gate_enabled = gate

        # 1) in_conv：分成 x / z 两支
        self.in_conv = nn.Conv2d(d_model, 2 * self.inner_dim, kernel_size=1, bias=bias)

        # 2) x 分支：局部 DWConv（稳一点）
        self.dwconv = nn.Conv2d(
            self.inner_dim, self.inner_dim,
            kernel_size=3, padding=1,
            groups=self.inner_dim, bias=bias
        )

        # 3) 四向 Mamba：按 share_mode 构造
        def make_mamba():
            return Mamba(d_model=self.inner_dim, d_state=d_state, d_conv=d_conv, expand=1)

        if share_mode == "all":
            m = make_mamba()
            self.m_h = m
            self.m_h_rev = m
            self.m_v = m
            self.m_v_rev = m
        elif share_mode == "hv":
            self.m_h = make_mamba()
            self.m_v = make_mamba()
            self.m_h_rev = self.m_h
            self.m_v_rev = self.m_v
        else:  # "none"
            self.m_h = make_mamba()
            self.m_h_rev = make_mamba()
            self.m_v = make_mamba()
            self.m_v_rev = make_mamba()

        # 4) concat + proj 融合：4C -> C
        self.fuse = nn.Linear(4 * self.inner_dim, self.inner_dim, bias=False)

        # 5) z 门控（逐点 + 通道门控）
        if self.gate_enabled:
            self.z_act = nn.SiLU(inplace=False)
            self.chan_gate = ChannelGate(self.inner_dim, reduction=gate_reduction)

        # 6) out_proj：回到 d_model
        self.out_conv = nn.Conv2d(self.inner_dim, d_model, kernel_size=1, bias=False)

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # --------- 序列化：水平（行优先） ---------
    @staticmethod
    def _to_seq_h(x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        return x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

    @staticmethod
    def _from_seq_h(s: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, l, c = s.shape
        assert l == h * w
        return s.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

    # --------- 序列化：垂直（等价“按列扫”） ---------
    @staticmethod
    def _to_seq_v(x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # [B,C,H,W] -> [B,W,H,C] -> [B, W*H, C]
        return x.permute(0, 3, 2, 1).contiguous().view(b, w * h, c)

    @staticmethod
    def _from_seq_v(s: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, l, c = s.shape
        assert l == w * h
        # [B, W*H, C] -> [B,W,H,C] -> [B,C,H,W]
        return s.view(b, w, h, c).permute(0, 3, 2, 1).contiguous()

    def _scan_4dir(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        # 水平 fwd/bwd
        s_h = self._to_seq_h(x)               # [B, H*W, C]
        y_h = self.m_h(s_h)
        y_h = self._from_seq_h(y_h, h, w)

        s_hr = torch.flip(s_h, dims=[1])
        y_hr = self.m_h_rev(s_hr)
        y_hr = torch.flip(y_hr, dims=[1])
        y_hr = self._from_seq_h(y_hr, h, w)

        # 垂直 fwd/bwd
        s_v = self._to_seq_v(x)               # [B, W*H, C]
        y_v = self.m_v(s_v)
        y_v = self._from_seq_v(y_v, h, w)

        s_vr = torch.flip(s_v, dims=[1])
        y_vr = self.m_v_rev(s_vr)
        y_vr = torch.flip(y_vr, dims=[1])
        y_vr = self._from_seq_v(y_vr, h, w)

        # concat + proj 融合
        y_cat = torch.cat([y_h, y_hr, y_v, y_vr], dim=1)                  # [B,4C,H,W]
        y_cat = y_cat.permute(0, 2, 3, 1).contiguous()                    # [B,H,W,4C]
        y = self.fuse(y_cat)                                              # [B,H,W,C]
        y = y.permute(0, 3, 1, 2).contiguous()                            # [B,C,H,W]
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xz = self.in_conv(x)
        x_branch, z_branch = torch.chunk(xz, 2, dim=1)

        # x 分支：局部 + 四向扫描
        x_branch = self.dwconv(x_branch)
        y = self._scan_4dir(x_branch)
        y = self.drop(y)

        # z 门控（保守）
        if self.gate_enabled:
            z = F.silu(z_branch, inplace=False)
            z = z * self.chan_gate(z)
            y = y * z

        return self.out_conv(y)


class MamberBlock(nn.Module):
    def __init__(self, dim, num_heads=None, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(MamberBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        # 全层都用 v2（share_mode 可按需改成 "all"/"hv"/"none"，默认 hv）
        self.attn = SS2D_Substitute_v2(
            d_model=dim,
            expand=1,          # 与你原先 ssm_ratio=1.0 的规模更接近
            d_state=16,
            d_conv=2,
            bias=bias,
            share_mode="hv",
            gate=True,
            gate_reduction=4,
            dropout=0.0,
        )
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        def _inner(inp):
            out = inp + self.attn(self.norm1(inp))
            out = out + self.ffn(self.norm2(out))
            return out

        # use_reentrant=False：新版本推荐，兼容性更好
        return checkpoint(_inner, x, use_reentrant=False)


# =========================================================================
# 3. 采样辅助模块与网络主体
# =========================================================================
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x): return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x): return self.body(x)


class MambaRealSR11(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, scale=1, dim=48,
                 num_blocks=[6, 2, 2, 1], num_refinement_blocks=6, heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(MambaRealSR11, self).__init__()
        self.scale = scale
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            MamberBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                        LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            MamberBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                        LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[
            MamberBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                        LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))
        self.latent = nn.Sequential(*[
            MamberBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                        LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            MamberBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                        LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            MamberBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                        LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))
        self.decoder_level1 = nn.Sequential(*[
            MamberBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                        LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            MamberBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                        LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        if self.scale > 1:
            self.tail = nn.Sequential(
                nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1) * (scale ** 2), kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(scale),
                nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1)
            )
        else:
            self.tail = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1)
        # after self.tail defined
        if isinstance(self.tail, nn.Conv2d):
            nn.init.zeros_(self.tail.weight)
            if self.tail.bias is not None:
                nn.init.zeros_(self.tail.bias)
        else:
            # scale>1 的 tail 是 Sequential，最后一层是 Conv2d
            last = self.tail[-1]
            if isinstance(last, nn.Conv2d):
                nn.init.zeros_(last.weight)
                if last.bias is not None:
                    nn.init.zeros_(last.bias)
    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        if self.scale > 1:
            base = F.interpolate(inp_img, scale_factor=self.scale, mode='nearest')
        else:
            base = inp_img

        out = self.tail(out_dec_level1) + base
        return out


if __name__ == "__main__":
    torch.cuda.empty_cache()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1, 3, 256, 256).to(device)

    # 针对眼底图像恢复（保持原图分辨率），scale 设为 1
    model = MambaRealSR11(scale=1, dim=48).to(device)
    y = model(x)

    print(f"输入图像尺寸: {x.shape}")
    print(f"输出图像尺寸: {y.shape}")