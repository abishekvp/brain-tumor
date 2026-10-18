# hybrid_quantum_vit.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import pennylane as qml


# ============================================================
# Configuration (matches the Colab notebook)
# ============================================================

@dataclass
class TrainConfig:
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    num_classes: int = 4

    # Vision Transformer (notebook sizes)
    embed_dim: int = 128
    transformer_depth: int = 6
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    # Quantum layer
    n_qubits: int = 6
    n_quantum_layers: int = 3

    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Patch Embedding
# ============================================================

class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        assert image_size % patch_size == 0
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)          # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


# ============================================================
# Multi-Head Self-Attention
# ============================================================

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out


# ============================================================
# Feed-Forward + Transformer Block
# ============================================================

class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float, dropout: float):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================
# Vision Transformer Backbone (returns CLS token)
# ============================================================

class VisionTransformerBackbone(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            cfg.image_size, cfg.patch_size, cfg.in_channels, cfg.embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, cfg.embed_dim))
        self.pos_dropout = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.transformer_depth)
        ])
        self.norm = nn.LayerNorm(cfg.embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, 0]  # CLS token


# ============================================================
# Quantum Layer (PennyLane)
# ============================================================

def build_quantum_layer(n_qubits: int, n_quantum_layers: int) -> qml.qnn.TorchLayer:
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    weight_shapes = {"weights": (n_quantum_layers, n_qubits, 3)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


# ============================================================
# Full Hybrid Quantum-Classical ViT
# ============================================================

class HybridQuantumViT(nn.Module):
    """
    Pipeline:
      image → ViT backbone (CLS) → Linear+Tanh → Quantum circuit → Dense head → logits
    """

    def __init__(self, cfg: Optional[TrainConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = TrainConfig()
        self.cfg = cfg

        self.backbone = VisionTransformerBackbone(cfg)

        # Compress CLS features to n_qubits for AngleEmbedding
        self.to_quantum = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.n_qubits),
            nn.Tanh(),
        )

        self.quantum_layer = build_quantum_layer(cfg.n_qubits, cfg.n_quantum_layers)

        self.classifier = nn.Sequential(
            nn.Linear(cfg.n_qubits, 64),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(64, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)           # (B, embed_dim)
        q_input = self.to_quantum(features)   # (B, n_qubits)
        q_output = self.quantum_layer(q_input)  # (B, n_qubits)
        logits = self.classifier(q_output)    # (B, num_classes)
        return logits


# ============================================================
# Convenience helpers for loading notebook checkpoints
# ============================================================

def load_checkpoint(
    path: str,
    device: Optional[str] = None,
    strict: bool = True,
) -> HybridQuantumViT:
    """
    Load a checkpoint saved by the Colab notebook
    (best_model.pth or last_checkpoint.pth).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device)

    # Notebook stores config as a plain dict under "config"
    cfg_dict = ckpt.get("config", {})
    # Filter only fields that TrainConfig knows about
    known = {f.name for f in TrainConfig.__dataclass_fields__.values()}
    cfg = TrainConfig(**{k: v for k, v in cfg_dict.items() if k in known})

    model = HybridQuantumViT(cfg).to(device)

    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=strict)
    model.eval()
    return model