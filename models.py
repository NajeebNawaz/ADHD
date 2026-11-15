import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel recalibration block for connectome embeddings."""
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(dim // reduction, 8)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.fc2(F.gelu(self.fc1(x))))
        return x * gate


class DualBranchResAttnConnNet(nn.Module):
    """
    Proposed Task-Decoupled Dual-Branch Residual Self-Attention Connectome Network (ResAttn-ConnNet).
    - Encodes task-specific functional connectome biomarker subnetworks (ADHD branch & Sex_F branch)
      into compact latent representations.
    - Applies Cross-Task Multi-Head Self-Attention (MHA) to capture neurobiological interplay
      between sex-specific functional connectivity and ADHD neurodevelopmental signatures.
    - Combines deep Pre-LayerNorm Residual + Squeeze-and-Excitation blocks with direct residual
      linear skip connections to prevent high-dimensional overfitting.
    """
    def __init__(self, dim_adhd: int, dim_sex: int, hidden_dim: int = 64, num_heads: int = 4, dropout: float = 0.35):
        super().__init__()
        self.proj_a = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_adhd, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.proj_s = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_sex, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=0.2, batch_first=True
        )
        self.res_a = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            SEBlock(hidden_dim),
        )
        self.res_s = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            SEBlock(hidden_dim),
        )
        self.head_a = nn.Linear(hidden_dim, 1)
        self.head_s = nn.Linear(hidden_dim, 1)
        self.skip_a = nn.Linear(dim_adhd, 1)
        self.skip_s = nn.Linear(dim_sex, 1)

    def forward(self, xa: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        ha = self.proj_a(xa)
        hs = self.proj_s(xs)
        tokens = torch.stack([ha, hs], dim=1)  # (B, 2, hidden_dim)
        attn_out, _ = self.mha(tokens, tokens, tokens)
        ha = ha + 0.3 * attn_out[:, 0, :] + self.res_a(ha)
        hs = hs + 0.3 * attn_out[:, 1, :] + self.res_s(hs)
        out_a = self.head_a(ha) + self.skip_a(xa)
        out_s = self.head_s(hs) + self.skip_s(xs)
        return torch.cat([out_a, out_s], dim=1)


class MultimodalDualBranchConnNet(nn.Module):
    """
    Multimodal Extension of DualBranchResAttnConnNet fusing Functional Connectome Biomarker
    Subnetworks with Quantitative (SDQ/APQ/EHQ/Age) and Categorical Metadata.
    """
    def __init__(self, dim_adhd: int, dim_sex: int, dim_meta: int, hidden_dim: int = 64, dropout: float = 0.35):
        super().__init__()
        self.conn_net = DualBranchResAttnConnNet(dim_adhd, dim_sex, hidden_dim=hidden_dim, dropout=dropout)
        self.meta_net = nn.Sequential(
            nn.Linear(dim_meta, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(32, 2),
        )
        self.alpha = nn.Parameter(torch.tensor([0.85, 0.90]))

    def forward(self, xa: torch.Tensor, xs: torch.Tensor, xm: torch.Tensor) -> torch.Tensor:
        conn_logits = self.conn_net(xa, xs)
        meta_logits = self.meta_net(xm)
        w = torch.sigmoid(self.alpha)
        return w * conn_logits + (1.0 - w) * meta_logits


class Brain1DCNN(nn.Module):
    """
    Multi-Scale 1D Convolutional Neural Network over Functional Connectome Biomarker Edges.
    Used for architectural ablation in the research paper.
    """
    def __init__(self, in_features: int, num_outputs: int = 2, dropout: float = 0.40):
        super().__init__()
        self.input_proj = nn.Linear(in_features, 128)
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool = nn.AdaptiveAvgPool1d(16)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(32 * 16, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, num_outputs),
        )
        self.skip = nn.Linear(in_features, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.input_proj(x)).unsqueeze(1)
        h = F.gelu(self.bn1(self.conv1(h)))
        h = F.gelu(self.bn2(self.conv2(h)))
        h = self.pool(h).flatten(1)
        return 0.5 * self.fc(h) + 0.5 * self.skip(x)


class DeepConnectomeMLP(nn.Module):
    """
    Deep Regularized Multi-Layer Perceptron (MLP) with LayerNorm, GELU, and Dropout.
    """
    def __init__(self, in_features: int, num_outputs: int = 2, dropout: float = 0.45):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
