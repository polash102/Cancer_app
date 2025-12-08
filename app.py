import os
import base64
import asyncio
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import gradio as gr

# -----------------------------------------------------
# 1. Asyncio patch (same as your individual apps)
# -----------------------------------------------------
def _safe_loop_del(self):
    try:
        if not self.is_closed():
            self.close()
    except Exception:
        pass

asyncio.BaseEventLoop.__del__ = _safe_loop_del

# -----------------------------------------------------
# 2. Core model building blocks
#    (DropPath, MBConv, MaxViT, EfficientViT, Hybrid)
# -----------------------------------------------------
class DropPath(nn.Module):
    """Stochastic depth."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep * random_tensor


def conv_1x1(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)


def conv_3x3(in_ch, out_ch, stride=1, groups=1):
    return nn.Conv2d(
        in_ch, out_ch, kernel_size=3, stride=stride,
        padding=1, groups=groups, bias=False
    )


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, se_ratio: float = 0.25, act=nn.GELU):
        super().__init__()
        hidden = max(8, int(channels * se_ratio))
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=True)
        self.act = act()
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=True)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = self.act(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class MBConv(nn.Module):
    """
    1x1 expand -> 3x3 depthwise -> SE -> 1x1 project (+ residual)
    """
    def __init__(
        self, in_ch, out_ch, stride=1,
        expand_ratio=4, se_ratio=0.25,
        drop_path=0.0, act=nn.GELU
    ):
        super().__init__()
        mid = int(in_ch * expand_ratio)
        self.use_res = (stride == 1 and in_ch == out_ch)

        self.expand = nn.Identity() if expand_ratio == 1 else nn.Sequential(
            conv_1x1(in_ch, mid), nn.BatchNorm2d(mid), act()
        )
        self.dw = nn.Sequential(
            conv_3x3(mid, mid, stride=stride, groups=mid),
            nn.BatchNorm2d(mid),
            act()
        )
        self.se = SqueezeExcite(mid, se_ratio, act=act)
        self.project = nn.Sequential(conv_1x1(mid, out_ch), nn.BatchNorm2d(out_ch))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        out = self.expand(x)
        out = self.dw(out)
        out = self.se(out)
        out = self.project(out)
        if self.use_res:
            out = self.drop_path(out) + x
        return out


# ------------ MaxViT token utils ------------
def window_partition(x: torch.Tensor, window_size: int):
    B, H, W, C = x.shape
    assert H % window_size == 0 and W % window_size == 0
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows, (H, W)


def window_reverse(
    windows: torch.Tensor, window_size: int,
    H: int, W: int, B: int, C: int
) -> torch.Tensor:
    nH = H // window_size
    nW = W // window_size
    x = windows.view(B, nH, nW, window_size, window_size, C).permute(
        0, 1, 3, 2, 4, 5
    ).contiguous()
    x = x.view(B, H, W, C)
    return x


def grid_partition(x: torch.Tensor, grid_size: int):
    B, H, W, C = x.shape
    assert H % grid_size == 0 and W % grid_size == 0
    Kh, Gh = grid_size, H // grid_size
    Kw, Gw = grid_size, W // grid_size
    x = x.view(B, Gh, Kh, Gw, Kw, C)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    x = x.view(B * Kh * Kw, Gh * Gw, C)
    return x, (B, Kh, Kw, Gh, Gw, C)


def grid_reverse(tokens: torch.Tensor, meta):
    B, Kh, Kw, Gh, Gw, C = meta
    x = tokens.view(B, Kh, Kw, Gh, Gw, C).permute(
        0, 3, 1, 4, 2, 5
    ).contiguous()
    H, W = Gh * Kh, Gw * Kw
    x = x.view(B, H, W, C)
    return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, act=nn.GELU):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = act()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    """Block (window) attention using nn.MultiheadAttention."""
    def __init__(self, dim, num_heads, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads=num_heads,
            dropout=attn_drop, batch_first=True
        )
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_windows):
        x = self.norm(x_windows)
        out, _ = self.attn(x, x, x, need_weights=False)
        return x_windows + self.proj_drop(out)


class GridAttention(nn.Module):
    """Grid attention."""
    def __init__(self, dim, num_heads, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads=num_heads,
            dropout=attn_drop, batch_first=True
        )
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_grids):
        x = self.norm(x_grids)
        out, _ = self.attn(x, x, x, need_weights=False)
        return x_grids + self.proj_drop(out)


class MaxViTBlock(nn.Module):
    def __init__(
        self, dim: int, window_size: int, grid_size: int,
        num_heads_block: int, num_heads_grid: int,
        mlp_ratio: float = 4.0, drop_path: float = 0.0,
        proj_drop: float = 0.0, attn_drop: float = 0.0
    ):
        super().__init__()
        self.mbconv = MBConv(
            dim, dim, stride=1,
            expand_ratio=4, se_ratio=0.25,
            drop_path=drop_path
        )
        self.block_attn = WindowAttention(dim, num_heads_block, attn_drop, proj_drop)
        self.grid_attn  = GridAttention(dim, num_heads_grid, attn_drop, proj_drop)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = MLP(dim, mlp_ratio, drop=proj_drop)
        self.drop_path = DropPath(drop_path)
        self.window_size = window_size
        self.grid_size = grid_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.mbconv(x)
        x_cl = x.permute(0, 2, 3, 1).contiguous()

        win_tokens, _ = window_partition(x_cl, self.window_size)
        win_tokens = self.block_attn(win_tokens)
        x_cl = window_reverse(win_tokens, self.window_size, H, W, B, C)

        grid_tokens, meta = grid_partition(x_cl, self.grid_size)
        grid_tokens = self.grid_attn(grid_tokens)
        x_cl = grid_reverse(grid_tokens, meta)

        tokens = x_cl.view(B, H * W, C)
        tokens = self.norm_ffn(tokens)
        tokens = tokens + self.drop_path(self.ffn(tokens))
        x_cl = tokens.view(B, H, W, C)

        x = x_cl.permute(0, 3, 1, 2).contiguous()
        return x


class MaxViTOncoXBackbone(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        dims: Tuple[int,int,int,int] = (64, 128, 256, 512),
        depths: Tuple[int,int,int,int] = (2, 2, 3, 2),
        window_sizes: Tuple[int,int,int,int] = (8, 8, 7, 7),
        grid_sizes:   Tuple[int,int,int,int] = (8, 8, 7, 7),
        heads_block:  Tuple[int,int,int,int] = (4, 4, 8, 8),
        heads_grid:   Tuple[int,int,int,int] = (4, 4, 8, 8),
        drop_path_rate: float = 0.1,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        c1, c2, c3, c4 = dims
        total_blocks = sum(depths)
        dp_rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        dp_i = 0

        self.stem = nn.Sequential(
            conv_3x3(in_chans, c1 // 2, stride=2),
            nn.BatchNorm2d(c1 // 2),
            nn.GELU(),
            conv_3x3(c1 // 2, c1, stride=1),
            nn.BatchNorm2d(c1),
            nn.GELU()
        )

        def make_stage(in_ch, out_ch, depth, wsize, gsize, h_block, h_grid, downsample=True):
            nonlocal dp_i
            layers = []
            if downsample:
                layers.append(MBConv(in_ch, out_ch, stride=2, drop_path=dp_rates[dp_i]))
                dp_i += 1
                cur_dim = out_ch
                blocks_to_make = depth - 1
            else:
                cur_dim = in_ch
                blocks_to_make = depth

            for _ in range(blocks_to_make):
                layers.append(
                    MaxViTBlock(
                        dim=cur_dim,
                        window_size=wsize,
                        grid_size=gsize,
                        num_heads_block=h_block,
                        num_heads_grid=h_grid,
                        mlp_ratio=mlp_ratio,
                        drop_path=dp_rates[dp_i],
                        proj_drop=proj_drop,
                        attn_drop=attn_drop
                    )
                )
                dp_i += 1
            return nn.Sequential(*layers)

        self.stage1 = make_stage(
            c1, c1, depths[0],
            window_sizes[0], grid_sizes[0],
            heads_block[0], heads_grid[0],
            downsample=False
        )
        self.stage2 = make_stage(
            c1, c2, depths[1],
            window_sizes[1], grid_sizes[1],
            heads_block[1], heads_grid[1],
            downsample=True
        )
        self.stage3 = make_stage(
            c2, c3, depths[2],
            window_sizes[2], grid_sizes[2],
            heads_block[2], heads_grid[2],
            downsample=True
        )
        self.stage4 = make_stage(
            c3, c4, depths[3],
            window_sizes[3], grid_sizes[3],
            heads_block[3], heads_grid[3],
            downsample=True
        )

        self.out_dim = c4
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d,)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if getattr(m, 'bias', None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Linear,)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if getattr(m, 'bias', None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        feat = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return feat


class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, act=nn.GELU):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            act(),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            act()
        )

    def forward(self, x):
        return self.block(x)


class EfficientViTBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0, se_ratio=0.25, drop_path=0.0, act=nn.GELU):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.dw = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            act()
        )
        self.se = SqueezeExcite(dim, se_ratio, act=act)
        self.pw1 = nn.Conv2d(dim, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = act()
        self.pw2 = nn.Conv2d(hidden, dim, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        residual = x
        x = self.dw(x)
        x = self.se(x)
        x = self.pw2(self.act(self.bn1(self.pw1(x))))
        x = self.bn2(x)
        return residual + self.drop_path(x)


class EfficientViTL1Backbone(nn.Module):
    def __init__(
        self,
        in_chans=3,
        dims=(32, 64, 128, 256),
        depths=(1, 2, 3, 2),
        drop_path_rate=0.05
    ):
        super().__init__()
        c1, c2, c3, c4 = dims
        total = sum(depths)
        dpr = torch.linspace(0, drop_path_rate, total).tolist()
        i = 0

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, c1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),
            nn.Conv2d(c1, c1, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU()
        )

        def make_stage(in_ch, out_ch, depth, stride):
            nonlocal i
            layers = []
            layers.append(DSConv(in_ch, out_ch, stride=stride))
            for _ in range(depth):
                layers.append(
                    EfficientViTBlock(
                        out_ch,
                        mlp_ratio=2.0,
                        se_ratio=0.25,
                        drop_path=dpr[i]
                    )
                )
                i += 1
            return nn.Sequential(*layers)

        self.s1 = make_stage(c1, c1, depths[0], stride=1)
        self.s2 = make_stage(c1, c2, depths[1], stride=2)
        self.s3 = make_stage(c2, c3, depths[2], stride=2)
        self.s4 = make_stage(c3, c4, depths[3], stride=2)

        self.out_dim = c4
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d,)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if getattr(m, 'bias', None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Linear,)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if getattr(m, 'bias', None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        feat = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return feat


class HybridMaxEffViT(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_chans: int = 3,
        maxvit_dims=(64, 128, 256, 512),
        maxvit_depths=(2, 2, 3, 2),
        eff_dims=(32, 64, 128, 256),
        eff_depths=(1, 2, 3, 2),
        drop_path_rate_max=0.05,
        drop_path_rate_eff=0.02,
        dropout: float = 0.3
    ):
        super().__init__()

        self.branch_max = MaxViTOncoXBackbone(
            in_chans=in_chans,
            dims=maxvit_dims,
            depths=maxvit_depths,
            window_sizes=(8, 8, 7, 7),
            grid_sizes=(8, 8, 7, 7),
            heads_block=(4, 4, 8, 8),
            heads_grid=(4, 4, 8, 8),
            drop_path_rate=drop_path_rate_max
        )

        self.branch_eff = EfficientViTL1Backbone(
            in_chans=in_chans,
            dims=eff_dims,
            depths=eff_depths,
            drop_path_rate=drop_path_rate_eff
        )

        d_max = self.branch_max.out_dim
        d_eff = self.branch_eff.out_dim

        self.dropout = nn.Dropout(dropout)

        self.head_max = nn.Linear(d_max, num_classes)
        self.head_eff = nn.Linear(d_eff, num_classes)

        self.branch_weights = nn.Parameter(torch.zeros(2))
        self.fuse_mlp = nn.Linear(2 * num_classes, num_classes)

        self.apply(self._init_head)

    @staticmethod
    def _init_head(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        f_max = self.branch_max(x)
        f_eff = self.branch_eff(x)

        logits_max = self.head_max(self.dropout(f_max))
        logits_eff = self.head_eff(self.dropout(f_eff))

        alpha = torch.softmax(self.branch_weights, dim=0)
        w_max, w_eff = alpha[0], alpha[1]

        w_logits_max = w_max * logits_max
        w_logits_eff = w_eff * logits_eff

        z = torch.cat([w_logits_max, w_logits_eff], dim=1)
        fused_logits = self.fuse_mlp(z)

        return fused_logits


class MaxEffFusionViT(HybridMaxEffViT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# -----------------------------------------------------
# 3. Device & model loading utilities
# -----------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_MODEL_CFG = dict(
    in_chans=3,
    maxvit_dims=(32, 64, 128, 256),
    maxvit_depths=(1, 2, 2, 1),
    eff_dims=(16, 32, 64, 128),
    eff_depths=(1, 2, 3, 2),
    drop_path_rate_max=0.1,
    drop_path_rate_eff=0.05,
    dropout=0.4,
)

def load_hybrid_model(ckpt_path: str):
    raw_state = torch.load(ckpt_path, map_location=device)

    if isinstance(raw_state, dict) and "model_state_dict" in raw_state:
        state = raw_state["model_state_dict"]
        model_cfg_from_ckpt = raw_state.get("model_config", {})
        label_mapping = raw_state.get("label_mapping", None)
    elif isinstance(raw_state, dict) and "state_dict" in raw_state:
        state = raw_state["state_dict"]
        model_cfg_from_ckpt = raw_state.get("model_config", {})
        label_mapping = raw_state.get("label_mapping", None)
    else:
        state = raw_state
        model_cfg_from_ckpt = {}
        label_mapping = None

    num_classes_from_ckpt = None
    for k, v in state.items():
        if k.endswith("head_max.weight"):
            num_classes_from_ckpt = v.shape[0]
            break

    if num_classes_from_ckpt is None:
        raise ValueError(f"Could not infer num_classes from checkpoint {ckpt_path}")

    cfg = DEFAULT_MODEL_CFG.copy()
    if isinstance(model_cfg_from_ckpt, dict):
        cfg.update(model_cfg_from_ckpt)

    model = MaxEffFusionViT(num_classes=num_classes_from_ckpt, **cfg).to(device)

    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            new_state[k[7:]] = v
        else:
            new_state[k] = v

    model.load_state_dict(new_state, strict=True)
    model.eval()

    if label_mapping is not None:
        idx_to_class = {v: k for k, v in label_mapping.items()}
    else:
        idx_to_class = {i: f"class_{i}" for i in range(num_classes_from_ckpt)}

    return model, idx_to_class

# -----------------------------------------------------
# 4. Cancer types registry (7 cancers)
# -----------------------------------------------------
CANCER_TYPES = {
    "Brain (MRI)": {
        "ckpt": "hybrid_maxeffvit_brain.pth",
        "modality": "brain MRI scan",
    },
    "Breast (Histopathology)": {
        "ckpt": "hybrid_maxeffvit_breast.pth",
        "modality": "breast tissue histopathology slide",
    },
    "Lung (CT)": {
        "ckpt": "lung_max_.pth",
        "modality": "lung CT scan",
    },
    "Kidney (CT / Histopathology)": {
        "ckpt": "hybrid_maxeffvit_Kidney.pth",
        "modality": "kidney CT scan or histopathology slide",
    },
    "Leukemia (Blood smear)": {
        "ckpt": "luc_max_.pth",
        "modality": "microscopic blood smear image",
    },
    "Cervical (Pap smear / Histopathology)": {
        "ckpt": "cer_max_ (1).pth",
        "modality": "cervical tissue or Pap smear image",
    },
    "Lymphoma (Histopathology)": {
        "ckpt": "lym_max_.pth",
        "modality": "lymph node histopathology slide",
    },
}

print("🔁 Loading all cancer models...")
MODELS: Dict[str, Any] = {}
IDX_TO_CLASS: Dict[str, Dict[int, str]] = {}
for cancer_name, cfg in CANCER_TYPES.items():
    print(f"  ➜ Loading model for {cancer_name} from {cfg['ckpt']} ...")
    model_c, idx2c = load_hybrid_model(cfg["ckpt"])
    MODELS[cancer_name] = model_c
    IDX_TO_CLASS[cancer_name] = idx2c

print("✅ All cancer models loaded.")

# -----------------------------------------------------
# 5. Shared CLIP model + per-cancer CLIP config
# -----------------------------------------------------
clip_model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
clip_model.eval()

CLIP_CONFIG = {
    "Brain (MRI)": {
        "texts": [
            "an MRI scan of the human brain",
            "a brain MRI scan with a tumor",
            "a T1 or T2 weighted MRI of the brain",
        ],
        "threshold": 0.22,
    },
    "Breast (Histopathology)": {
        "texts": [
            "a microscopic histopathology image of breast tissue",
            "a histology slide of breast cancer",
            "a microscopic image of benign breast tumor tissue",
            "a microscopic image of malignant breast tumor tissue",
        ],
        "threshold": 0.22,
    },
    "Lung (CT)": {
        "texts": [
            "a CT scan of the lungs",
            "a chest CT with lungs",
            "a thoracic CT scan image",
        ],
        "threshold": 0.22,
    },
    "Kidney (CT / Histopathology)": {
        "texts": [
            "a CT scan of kidneys",
            "a CT scan of kidneys with a tumor",
            "a cross-sectional CT image of the abdomen showing kidneys",
        ],
        "threshold": 0.22,
    },
    "Leukemia (Blood smear)": {
        "texts": [
            "a microscopic image of blood smear",
            "a microscope image showing white blood cells",
            "a stained blood smear slide",
        ],
        "threshold": 0.22,
    },
    "Cervical (Pap smear / Histopathology)": {
        "texts": [
            "a pap smear cervical cell image",
            "a cytology slide of cervical squamous cells",
            "a microscopic image of cervical cells from Pap smear",
        ],
        "threshold": 0.22,
    },
    "Lymphoma (Histopathology)": {
        "texts": [
            "a histopathology slide of lymph node tissue",
            "a microscopic pathology image of lymphoma",
        ],
        "threshold": 0.22,
    },
}

def clip_gate(image: Image.Image, cancer_type: str):
    cfg = CLIP_CONFIG[cancer_type]
    texts = cfg["texts"]
    threshold = cfg["threshold"]

    inputs = clip_processor(
        text=texts,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        sims = (text_embeds @ image_embeds.T).squeeze(1)
        max_sim = float(sims.max().item())

    is_ok = max_sim >= threshold
    return is_ok, max_sim, threshold


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def preprocess_image(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB")
    x = transform(img)
    x = x.unsqueeze(0)
    return x.to(device)

DISCLAIMER = (
    "⚠️ **Research disclaimer**\n"
    "This system is an experimental prototype for classifying medical images from several cancer datasets. "
    "It has not undergone clinical validation and its outputs are algorithmic predictions intended solely "
    "for methodological and educational research. They must **not** be used for diagnostic or therapeutic "
    "decision-making or to delay seeing a doctor."
)

LOW_CONFIDENCE_THRESHOLD = 0.55

def predict_multicancer(image: Image.Image, cancer_type: str):
    if image is None:
        return {}, "⚠️ No image provided.", DISCLAIMER

    # 1) CLIP gate
    is_ok, sim, thr = clip_gate(image, cancer_type)
    if not is_ok:
        rejection_text = (
            f"🚫 **Input rejected for {cancer_type} model**\n\n"
            f"The uploaded image does not look like a typical {CANCER_TYPES[cancer_type]['modality']} "
            f"according to a CLIP similarity check "
            f"(max similarity = {sim:.2f}, threshold = {thr:.2f}).\n\n"
            "Please upload the correct type of image for this cancer model."
        )
        return {}, rejection_text, DISCLAIMER

    # 2) Run model
    model = MODELS[cancer_type]
    idx_to_class = IDX_TO_CLASS[cancer_type]

    x = preprocess_image(image)
    with torch.no_grad():
        logits = model(x)
        probs_tensor = F.softmax(logits, dim=1)[0].cpu()

    probs = probs_tensor.tolist()
    prob_dict = {idx_to_class[i]: float(probs[i]) for i in range(len(probs))}
    pred_idx = int(probs_tensor.argmax().item())
    pred_class = idx_to_class[pred_idx]
    max_prob = float(probs_tensor.max().item())

    # 3) Summary text
    if max_prob < LOW_CONFIDENCE_THRESHOLD:
        summary = (
            f"⚠️ **Low-confidence prediction**\n\n"
            f"Top predicted class: **{pred_class}** "
            f"(probability {max_prob:.2f}, below threshold {LOW_CONFIDENCE_THRESHOLD:.2f}).\n\n"
            "This may indicate that the image is outside the training distribution "
            "or of poor quality. Use only for research demonstration."
        )
    else:
        summary = (
            f"✅ **Predicted class:** **{pred_class}** "
            f"(probability {max_prob:.2f})\n\n"
            "_Note: this reflects similarity to patterns in the training data and is "
            "intended for research / educational use only._"
        )

    return prob_dict, summary, DISCLAIMER

# -----------------------------------------------------
# 6. Doctor assistant (symptom Q&A – research only)
# -----------------------------------------------------
def doctor_assistant(symptoms: str, age: str, gender: str, focus_cancer: str):
    if not symptoms.strip():
        return (
            "🩺 **Dr. Onco (virtual assistant)**\n\n"
            "Please describe your symptoms so I can give **general health tips** "
            "(not a diagnosis)."
        )

    base = (
        "🩺 **Dr. Onco – virtual assistant**\n\n"
        "I’m not a real doctor and I **cannot tell which cancer you have**. "
        "But based on your description, here are some **general suggestions** and "
        "**home-care tips** you might discuss with a healthcare professional.\n\n"
    )

    red_flags = (
        "🚨 **Red-flag symptoms – seek urgent medical care if you notice:**\n"
        "- Sudden severe pain, chest pain, or difficulty breathing\n"
        "- Loss of consciousness, confusion, or seizures\n"
        "- Very high fever that does not improve\n"
        "- Heavy uncontrolled bleeding or rapid weight loss\n\n"
    )

    lifestyle = (
        "🌿 **General supportive tips (for many chronic conditions):**\n"
        "- Stay hydrated with clean water, avoid sugary drinks\n"
        "- Eat simple, light meals: fruits, vegetables, whole grains, lean protein\n"
        "- Avoid smoking, alcohol, and self-medication without a doctor\n"
        "- Maintain regular sleep schedule (7–8 hours if possible)\n"
        "- Gentle walking / stretching if you are able and your doctor agrees\n\n"
    )

    home_remedy = (
        "🏠 **Home-care style ideas (only with medical approval):**\n"
        "- Warm compress for mild muscle aches (avoid on unknown tumors)\n"
        "- Simple breathing exercises for mild anxiety or stress\n"
        "- Keeping a symptom diary to show your doctor (time, severity, triggers)\n\n"
    )

    cancer_mention = ""
    if focus_cancer != "General":
        cancer_mention = (
            f"🧬 **You selected focus:** {focus_cancer}\n"
            "This app’s imaging models are trained only on research datasets. "
            "Even if your symptoms sound similar, **only imaging + a real doctor** "
            "can evaluate you properly.\n\n"
        )

    safety = (
        "📌 **Very important:**\n"
        "- These suggestions are **not medical advice** or a diagnosis.\n"
        "- Always consult a qualified doctor, especially if symptoms are new, "
        "worsening, or worrying.\n"
    )

    return base + cancer_mention + red_flags + lifestyle + home_remedy + safety

# -----------------------------------------------------
# 7. UI helpers: background image, CSS, login overlay
# -----------------------------------------------------
def load_base64_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

BG64 = load_base64_image("cancer_bg.jpg")

MEDICAL_CSS = f"""
<style>
  body {{
    margin: 0;
    padding: 0;
    background-image: url('data:image/jpeg;base64,{BG64}');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  .gradio-container {{
    background: radial-gradient(circle at top left, rgba(15, 76, 129, 0.9), rgba(0, 0, 0, 0.95)) !important;
    color: white !important;
  }}
  h1, h2, h3, h4, h5, h6, p, div, label {{
    color: #fdfdfd !important;
    text-shadow: 0 0 6px rgba(0,0,0,0.65);
  }}

  /* Top navigation bar */
  .top-nav {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.9rem 1.2rem;
    border-radius: 18px;
    background: rgba(0,0,0,0.55);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.15);
    margin-bottom: 0.8rem;
  }}
  .top-nav-title {{
    font-size: 1.1rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }}
  .top-nav-title span.logo-circle {{
    width: 30px;
    height: 30px;
    border-radius: 999px;
    background: radial-gradient(circle at 30% 20%, #00f5a0, #00d9f5);
    display: inline-block;
    box-shadow: 0 0 18px rgba(0, 245, 160, 0.7);
  }}
  .top-nav-sub {{
    font-size: 0.78rem;
    opacity: 0.85;
  }}
  .top-nav-links a {{
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    font-size: 0.82rem;
    padding: 0.28rem 0.7rem;
    border-radius: 999px;
    margin-left: 0.4rem;
    color: #fdfdfd;
    text-decoration: none;
    border: 1px solid rgba(255,255,255,0.28);
    background: linear-gradient(135deg, rgba(0,0,0,0.35), rgba(0,0,0,0.7));
    transition: all 0.18s ease-out;
  }}
  .top-nav-links a:hover {{
    transform: translateY(-1px) scale(1.02);
    background: linear-gradient(135deg, #00f5a0, #00d9f5);
    color: #000;
  }}

  .app-shell {{
    background: rgba(0, 0, 0, 0.7) !important;
    border-radius: 20px !important;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.12);
    padding: 1rem 1rem 1.2rem 1rem !important;
  }}

  .app-image, .app-probs {{
    background: rgba(255, 255, 255, 0.07) !important;
    border-radius: 18px !important;
    border: 1px solid rgba(255,255,255,0.22) !important;
    height: 360px !important;
    overflow: hidden !important;
  }}
  .app-image > div {{ height: 100% !important; }}
  .app-image img, .app-image canvas, .app-image video {{
    width: 100% !important;
    height: 100% !important;
    object-fit: contain !important;
    border-radius: 16px !important;
  }}

  .app-report, .app-disclaimer {{
    max-height: 360px !important;
    overflow-y: auto !important;
  }}

  /* Buttons micro-interaction */
  button, .gr-button-primary {{
    transition: transform 0.12s ease-out, box-shadow 0.12s ease-out, background 0.2s ease-out;
  }}
  .gr-button-primary {{
    background: linear-gradient(135deg, #00f5a0, #00d9f5) !important;
    color: #000 !important;
    border-radius: 999px !important;
    font-weight: 800 !important;
    border: none !important;
    box-shadow: 0 8px 18px rgba(0, 245, 160, 0.35);
  }}
  .gr-button-primary:hover {{
    transform: translateY(-1px) scale(1.03);
    box-shadow: 0 10px 24px rgba(0, 217, 245, 0.4);
  }}
  .gr-button-primary:active {{
    transform: translateY(0px) scale(0.99);
    box-shadow: 0 4px 10px rgba(0,0,0,0.6);
  }}

  .tag-chip {{
    display: inline-flex;
    align-items: center;
    padding: 0.20rem 0.7rem;
    border-radius: 999px;
    border: 1px dashed rgba(255,255,255,0.6);
    font-size: 0.8rem;
    margin-right: 0.4rem;
    margin-bottom: 0.2rem;
  }}
  .tag-chip strong {{
    margin-left: 0.25rem;
  }}

  .app-probs * {{
    color: #ffffff !important;
    text-shadow: 0px 0px 6px rgba(0,0,0,0.8);
    font-weight: 700 !important;
  }}
  .app-probs [data-testid="label"],
  .app-probs [data-testid="label"] * {{
    color: #000000 !important;
    text-shadow: none !important;
  }}
  .app-image [data-testid="label"],
  .app-image [data-testid="label"] * {{
    color: #000000 !important;
    text-shadow: none !important;
    font-weight: 700 !important;
  }}

  /* Floating 3D doctor bot */
  .doctor-bot {{
    position: fixed;
    bottom: 20px;
    right: 18px;
    width: 72px;
    height: 72px;
    border-radius: 999px;
    background: radial-gradient(circle at 30% 20%, #ffffff, #dff9fb);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 12px 30px rgba(0,0,0,0.6);
    cursor: grab;
    user-select: none;
    z-index: 9999;
    border: 2px solid rgba(0,0,0,0.2);
    animation: floatBot 3s ease-in-out infinite;
  }}
  .doctor-bot:active {{
    cursor: grabbing;
  }}
  .doctor-bot-emoji {{
    font-size: 2.4rem;
    filter: drop-shadow(0 0 12px rgba(0,0,0,0.45));
  }}
  @keyframes floatBot {{
    0% {{ transform: translateY(0px); }}
    50% {{ transform: translateY(-6px); }}
    100% {{ transform: translateY(0px); }}
  }}

  /* Simple draggable JS hook */
  @media (pointer: coarse) {{
    .doctor-bot {{
      width: 76px;
      height: 76px;
    }}
  }}

  /* Login overlay */
  #login-modal {{
    position: fixed;
    inset: 0;
    background: radial-gradient(circle at top, rgba(0,0,0,0.85), rgba(0,0,0,0.95));
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 9998;
  }}
  #login-card {{
    max-width: 420px;
    width: 90%;
    padding: 1.5rem 1.4rem;
    border-radius: 18px;
    background: linear-gradient(145deg, rgba(0,0,0,0.9), rgba(19, 84, 122, 0.9));
    border: 1px solid rgba(0, 245, 160, 0.35);
    box-shadow: 0 18px 40px rgba(0,0,0,0.85);
  }}
  #login-title {{
    font-size: 1.1rem;
    font-weight: 800;
    margin-bottom: 0.4rem;
  }}
  #login-sub {{
    font-size: 0.83rem;
    opacity: 0.9;
    margin-bottom: 0.8rem;
  }}
  #login-footer {{
    font-size: 0.72rem;
    opacity: 0.75;
    margin-top: 0.6rem;
  }}
</style>

<script>
  // Basic drag for doctor-bot (pure client side)
  window.addEventListener("DOMContentLoaded", () => {{
    const bot = document.querySelector(".doctor-bot");
    if (!bot) return;

    let isDown = false;
    let offsetX = 0;
    let offsetY = 0;

    bot.addEventListener("pointerdown", (e) => {{
      isDown = true;
      offsetX = e.clientX - bot.getBoundingClientRect().left;
      offsetY = e.clientY - bot.getBoundingClientRect().top;
    }});
    window.addEventListener("pointermove", (e) => {{
      if (!isDown) return;
      bot.style.left = (e.clientX - offsetX) + "px";
      bot.style.top = (e.clientY - offsetY) + "px";
      bot.style.right = "auto";
      bot.style.bottom = "auto";
    }});
    window.addEventListener("pointerup", () => {{ isDown = false; }});
  }});
</script>
"""

title_text = "🧬 Multi-Cancer Imaging Classifier – Research Prototype"

# -----------------------------------------------------
# 8. Login logic (guest / user / admin)
# -----------------------------------------------------
USER_CREDENTIALS = {
    "demo_user": "user123",     # change or extend for demo
}
ADMIN_CREDENTIALS = {
    "admin": "admin123",        # change for demo
}

def login_guest():
    return "guest", gr.update(visible=False), "🟢 Logged in as **Guest** (no account)."

def login_user(username, password):
    if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
        return "user", gr.update(visible=False), f"🟢 Logged in as **User: {username}**"
    else:
        return gr.update(value="guest"), gr.update(visible=True), (
            "🔴 Invalid **user** credentials. Try again or continue as guest."
        )

def login_admin(username, password):
    if username in ADMIN_CREDENTIALS and ADMIN_CREDENTIALS[username] == password:
        return "admin", gr.update(visible=False), f"🟣 Logged in as **Admin: {username}**"
    else:
        return gr.update(value="guest"), gr.update(visible=True), (
            "🔴 Invalid **admin** credentials. Try again or continue as guest."
        )

# -----------------------------------------------------
# 9. Build Gradio UI
# -----------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # Global CSS + doctor-bot
    gr.HTML(MEDICAL_CSS)
    gr.HTML(
        """
        <div class="doctor-bot">
          <div class="doctor-bot-emoji">🩺</div>
        </div>
        """
    )

    # State: who is logged in
    role_state = gr.State("guest")
    login_status = gr.Markdown("🔓 Not signed in (Guest mode).")

    # ----------------- LOGIN MODAL -----------------
    with gr.Group(visible=True, elem_id="login-modal") as login_group:
        gr.HTML(
            """
            <div id="login-card">
              <div id="login-title">🔐 Multi-Cancer Imaging – Sign in</div>
              <div id="login-sub">
                Choose how you want to explore this research prototype.
              </div>
            </div>
            """
        )
        with gr.Column(elem_id="login-card"):
            user_name = gr.Textbox(label="Username", placeholder="demo_user / admin", value="")
            user_pass = gr.Textbox(label="Password", type="password", placeholder="••••••••")

            with gr.Row():
                guest_btn = gr.Button("Continue as guest", variant="primary")
            with gr.Row():
                user_btn = gr.Button("User sign in")
                admin_btn = gr.Button("Admin sign in")

            login_msg = gr.Markdown("", elem_id="login-footer")

        guest_btn.click(
            fn=login_guest,
            inputs=None,
            outputs=[role_state, login_group, login_msg],
        )

        user_btn.click(
            fn=login_user,
            inputs=[user_name, user_pass],
            outputs=[role_state, login_group, login_msg],
        )

        admin_btn.click(
            fn=login_admin,
            inputs=[user_name, user_pass],
            outputs=[role_state, login_group, login_msg],
        )

    # ----------------- TOP NAV -----------------
    with gr.Row():
        gr.HTML(
            f"""
            <div class="top-nav">
              <div class="top-nav-title">
                <span class="logo-circle"></span>
                <div>
                  <div>{title_text}</div>
                  <div class="top-nav-sub">Hybrid MaxViT + EfficientViT • CLIP-gated • Research-only</div>
                </div>
              </div>
              <div class="top-nav-links">
                <a href="https://www.linkedin.com/in/md-saymon-hosen-polash-89703b244/" target="_blank">🔗 <span>LinkedIn</span></a>
                <a href="https://www.facebook.com/share/17S3D1DFC9/" target="_blank">📘 <span>Facebook</span></a>
                <a href="https://github.com/polash102" target="_blank">🐱 <span>GitHub</span></a>
              </div>
            </div>
            """
        )

    # Show login status under nav
    login_status.render()

    # ----------------- MAIN APP SHELL -----------------
    with gr.Column(elem_classes="app-shell"):
        with gr.Tabs():
            # ===== TAB 1: Multi-Cancer Classifier =====
            with gr.Tab("🩻 Multi-Cancer Classifier"):
                gr.Markdown(
                    "### Upload a medical image\n"
                    "1. Choose the **cancer imaging domain**.\n"
                    "2. Upload a matching image.\n"
                    "3. The model will run a **CLIP gate** to reject wrong modalities and "
                    "then output class probabilities with a **low-confidence warning**."
                )

                with gr.Row():
                    with gr.Column(scale=5):
                        cancer_type_in = gr.Dropdown(
                            choices=list(CANCER_TYPES.keys()),
                            value="Brain (MRI)",
                            label="Cancer type / dataset",
                            info="Select which model to use.",
                        )
                        img_in = gr.Image(
                            type="pil",
                            label="Input medical image",
                            elem_classes="app-image",
                        )

                        with gr.Row():
                            analyze_btn = gr.Button("Run analysis", variant="primary")
                            clear_btn = gr.Button("Clear")

                        gr.Markdown(
                            """
                            <div>
                              <span class="tag-chip">✅ <strong>Matched modality only</strong></span>
                              <span class="tag-chip">🧠 <strong>CLIP-based input gate</strong></span>
                              <span class="tag-chip">⚠️ <strong>Low-confidence warnings</strong></span>
                            </div>
                            """,
                        )

                    with gr.Column(scale=5):
                        probs_out = gr.Label(
                            num_top_classes=3,
                            label="📊 Class probabilities",
                            elem_classes="app-probs",
                        )
                        report_out = gr.Markdown(
                            label="🔎 Model summary",
                            elem_classes="app-report",
                        )
                        disc_out = gr.Markdown(
                            value=DISCLAIMER,
                            elem_classes="app-disclaimer",
                        )

                analyze_btn.click(
                    fn=predict_multicancer,
                    inputs=[img_in, cancer_type_in],
                    outputs=[probs_out, report_out, disc_out],
                )

                clear_btn.click(
                    fn=lambda: ({}, "", DISCLAIMER),
                    inputs=None,
                    outputs=[probs_out, report_out, disc_out],
                )

            # ===== TAB 2: Doctor Assistant =====
            with gr.Tab("👨‍⚕️ Dr. Onco – Assistant"):
                gr.Markdown(
                    "### Ask Dr. Onco (virtual assistant)\n"
                    "Describe your **symptoms in your own words**. "
                    "Dr. Onco will give **general guidance and home-care ideas**, "
                    "but **cannot diagnose cancer**."
                )
                with gr.Row():
                    with gr.Column(scale=5):
                        symptom_box = gr.Textbox(
                            label="📝 Describe your symptoms",
                            placeholder="Example: I have had a persistent cough for 3 weeks, mild fever and fatigue...",
                            lines=5,
                        )
                        age_box = gr.Textbox(
                            label="Approximate age (optional)",
                            placeholder="e.g., 25, 40, 60...",
                        )
                        gender_box = gr.Dropdown(
                            label="Gender (optional)",
                            choices=["Prefer not to say", "Female", "Male", "Other"],
                            value="Prefer not to say",
                        )
                        focus_cancer = gr.Dropdown(
                            label="Main concern (optional)",
                            choices=["General"] + list(CANCER_TYPES.keys()),
                            value="General",
                        )
                        ask_btn = gr.Button("Ask Dr. Onco", variant="primary")

                    with gr.Column(scale=5):
                        doctor_reply = gr.Markdown(
                            value=(
                                "🩺 **Dr. Onco is here.**\n\n"
                                "Tell me your symptoms and worries in a few sentences. "
                                "I will respond with **general guidance only**."
                            ),
                            elem_classes="app-report"
                        )

                ask_btn.click(
                    fn=doctor_assistant,
                    inputs=[symptom_box, age_box, gender_box, focus_cancer],
                    outputs=[doctor_reply],
                )

            # ===== TAB 3: About the Project =====
            with gr.Tab("📚 About the Project"):
                gr.Markdown(
                    """
                    ### Overview

                    This web application demonstrates a **unified multi-cancer imaging classifier**.  
                    It integrates trained models for:

                    - 🧠 Brain MRI  
                    - 🫁 Lung CT  
                    - 🩺 Breast / Kidney / Cervical / Lymphoma histopathology  
                    - 🧫 Leukemia blood smear  

                    The pipeline combines:

                    - A **hybrid backbone** (MaxViT + EfficientViT style)
                    - **CLIP-based input validation** to reject obviously wrong image types
                    - **Softmax-based low-confidence warnings** when the model is unsure

                    > This is strictly a **research / educational prototype** and not a medical device.
                    """
                )

            # ===== TAB 4: Team & Contact =====
            with gr.Tab("👨‍⚕️ Team & Credits"):
                gr.Markdown(
                    """
                    ### Project Lead

                    **Name:** Md. Saymon Hosen Polash  
                    **Role:** Researcher / Developer – Multi-Cancer Imaging Project

                    ### Contact & Profiles

                    - 🔗 [LinkedIn](https://www.linkedin.com/in/md-saymon-hosen-polash-89703b244/)
                    - 📘 [Facebook](https://www.facebook.com/share/17S3D1DFC9/)
                    - 🐱 [GitHub](https://github.com/polash102)
                    - ✉️ Email: `your_email@example.com`

                    """
                )

            # ===== TAB 5: How to Use / FAQ =====
            with gr.Tab("❓ How to Use / FAQ"):
                gr.Markdown(
                    """
                    ### How to use the multi-cancer demo

                    1. Go to **Multi-Cancer Classifier** tab.  
                    2. Select a **cancer type / dataset** from the dropdown.
                    3. Upload a matching medical image:
                       - Brain (MRI) → brain MRI slices  
                       - Lung (CT) → chest CT  
                       - Leukemia → blood smear microscopy  
                       - Histopathology models → stained tissue slides  
                    4. Click **“Run analysis”**.

                    ### Messages explained

                    - **Input rejected**  
                      CLIP decided the image is not similar to the expected modality  
                      (e.g. a selfie for a brain MRI model).  

                    - **Low-confidence prediction**  
                      The model's top probability is below the safety threshold.  
                      This suggests the image may be out of distribution or low quality.

                    - **Normal prediction**  
                      The app shows the most likely class and top probabilities.

                    ### Disclaimer

                    - This tool **does not** provide medical advice.  
                    - It is designed for:
                      - methodology demonstrations,
                      - project defenses and presentations,
                      - machine learning research prototyping.
                    """
                )

        # ----------------- FOOTER: credits + emoji tech stack -----------------
        gr.Markdown(
            """
            ---
            **Made by Md. Saymon Hosen Polash** 🧑‍💻🧠  

            Tech stack: 🐍 Python · 🔥 PyTorch · 🧬 CLIP · 🌐 Gradio · ☁️ Hugging Face  

            <!-- Only emojis for how app is made -->
            🧠📷 ➕ 🐍📦 ➕ 🔥🧮 ➕ 🧬🤖 ➕ ☁️🚀
            """,
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
