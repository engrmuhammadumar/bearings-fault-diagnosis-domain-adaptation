"""
physgen_model.py
----------------
PhysGen-Bearing: dual-path physics-disentangled domain-generalisation model.

Architecture (Pillars 1–4 from the proposal):

  ┌─────────────────────────────────────────────────────────────────────┐
  │  Input: (signal, fs, rpm, geom_vec)                                 │
  │                                                                     │
  │  ┌───────────────────────────────┐                                  │
  │  │ Order-tracking layer (Pillar 1)                                  │
  │  │   Resamples signal → order domain spectrum (geometry-invariant   │
  │  │   coordinate system).                                            │
  │  └─────────────┬─────────────────┘                                  │
  │                ▼                                                    │
  │      ┌─────────┴─────────┐                                          │
  │      ▼                   ▼                                          │
  │  Path B               Path A                                        │
  │  (geometry-invariant) (geometry-conditional, FiLM on geom_vec)      │
  │   morphology encoder  spectral-peak encoder                         │
  │      │                   │                                          │
  │      │                   ▼                                          │
  │      │            Physics-consistency loss                          │
  │      │            (peaks should lie near predicted BPFO/BPFI)       │
  │      │                                                              │
  │      ├──────► Adversarial domain head (gradient reversal)           │
  │      │        forces Path B to forget which dataset/geometry        │
  │      │                                                              │
  │      ▼                                                              │
  │   Fault classifier (4-way: healthy/inner/outer/ball)                │
  └─────────────────────────────────────────────────────────────────────┘

Pillar 4 (Mondrian conformal calibration) is applied at evaluation time
in conformal.py — it wraps this model's logits without changing training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import N_CLASSES, GEOM_VEC_DIM, ORDER_BINS
from order_tracking import OrderTrackingLayer


# ─────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────
class ConvBlock1d(nn.Module):
    def __init__(self, c_in, c_out, k=7, stride=2, pad=None):
        super().__init__()
        pad = (k - 1) // 2 if pad is None else pad
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, stride=stride, padding=pad),
            nn.BatchNorm1d(c_out),
            nn.GELU(),
        )
    def forward(self, x):
        return self.net(x)


class GradReverse(torch.autograd.Function):
    """Gradient reversal layer for adversarial training."""
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class FiLM(nn.Module):
    """Feature-wise linear modulation: scales/shifts features by a conditioning vector."""
    def __init__(self, cond_dim, feat_channels):
        super().__init__()
        self.to_scale = nn.Linear(cond_dim, feat_channels)
        self.to_shift = nn.Linear(cond_dim, feat_channels)

    def forward(self, feat, cond):
        # feat: (B, C, T)  cond: (B, cond_dim)
        gamma = self.to_scale(cond).unsqueeze(-1)
        beta  = self.to_shift(cond).unsqueeze(-1)
        return gamma * feat + beta


# ─────────────────────────────────────────────────────────────────────────
# Two encoders
# ─────────────────────────────────────────────────────────────────────────
class GeomInvariantEncoder(nn.Module):
    """
    Path B: takes order-domain spectrum and produces a feature vector
    that is intended to be invariant to bearing geometry. Larger than Path A.
    """
    def __init__(self, in_len=ORDER_BINS, feat_dim=128):
        super().__init__()
        self.feat_dim = feat_dim
        self.net = nn.Sequential(
            ConvBlock1d(1, 32, k=15, stride=2),
            ConvBlock1d(32, 64, k=9, stride=2),
            ConvBlock1d(64, 128, k=5, stride=2),
            ConvBlock1d(128, 128, k=3, stride=2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, feat_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class GeomConditionalEncoder(nn.Module):
    """
    Path A: takes order-domain spectrum and is CONDITIONED on bearing geometry
    via FiLM. Trained with a physics-consistency loss to put activations near
    the BPFO/BPFI/BSF order locations.
    """
    def __init__(self, in_len=ORDER_BINS, cond_dim=GEOM_VEC_DIM, feat_dim=64):
        super().__init__()
        self.feat_dim = feat_dim
        self.b1 = ConvBlock1d(1, 32, k=9, stride=2)
        self.f1 = FiLM(cond_dim, 32)
        self.b2 = ConvBlock1d(32, 64, k=5, stride=2)
        self.f2 = FiLM(cond_dim, 64)
        self.b3 = ConvBlock1d(64, 64, k=3, stride=2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, feat_dim),
            nn.GELU(),
        )
        # Attention map (1 channel) for physics-consistency loss
        self.attn = nn.Conv1d(64, 1, kernel_size=1)

    def forward(self, x, cond):
        h = self.f1(self.b1(x), cond)
        h = self.f2(self.b2(h), cond)
        h = self.b3(h)
        attn_logits = self.attn(h)                  # (B, 1, T')
        attn = torch.softmax(attn_logits, dim=-1)   # (B, 1, T') — soft mass
        feat = self.head(h)
        return feat, attn, h


# ─────────────────────────────────────────────────────────────────────────
# Heads
# ─────────────────────────────────────────────────────────────────────────
class FaultClassifier(nn.Module):
    def __init__(self, in_dim, n_classes=N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )
    def forward(self, x):
        return self.net(x)


class DomainHead(nn.Module):
    """Predicts which dataset/geometry the sample came from (adversarial)."""
    def __init__(self, in_dim, n_domains):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, n_domains),
        )
    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────
class PhysGenBearing(nn.Module):
    def __init__(self, n_domains, n_classes=N_CLASSES, geom_dim=GEOM_VEC_DIM,
                 path_b_feat=128, path_a_feat=64):
        super().__init__()
        self.order_layer = OrderTrackingLayer()
        self.enc_inv = GeomInvariantEncoder(feat_dim=path_b_feat)
        self.enc_cond = GeomConditionalEncoder(cond_dim=geom_dim, feat_dim=path_a_feat)
        self.classifier = FaultClassifier(in_dim=path_b_feat, n_classes=n_classes)
        self.domain_head = DomainHead(in_dim=path_b_feat, n_domains=n_domains)
        # Aux classifier that uses BOTH paths concatenated — only used as a
        # regulariser to make sure Path A also has discriminative info.
        self.aux_classifier = FaultClassifier(
            in_dim=path_b_feat + path_a_feat, n_classes=n_classes,
        )

    def forward(self, signal, fs, rpm, geom_vec, adv_lambda=1.0):
        # 1. Order-domain spectrum
        spec = self.order_layer(signal, fs, rpm)               # (B,1,O)

        # 2. Path B (invariant)
        feat_inv = self.enc_inv(spec)                          # (B, dB)

        # 3. Path A (conditional)
        feat_cond, attn, hidden_a = self.enc_cond(spec, geom_vec)   # (B, dA), (B,1,T'), (B,C,T')

        # 4. Heads
        logits_main = self.classifier(feat_inv)
        logits_aux  = self.aux_classifier(torch.cat([feat_inv, feat_cond], dim=-1))
        # Adversarial domain prediction on the INVARIANT features
        feat_rev = grad_reverse(feat_inv, adv_lambda)
        logits_domain = self.domain_head(feat_rev)

        return {
            "logits_main":   logits_main,
            "logits_aux":    logits_aux,
            "logits_domain": logits_domain,
            "spec":          spec,           # (B,1,O) — passed to physics loss
            "attn":          attn,           # (B,1,T') — passed to physics loss
            "feat_inv":      feat_inv,
            "feat_cond":     feat_cond,
        }


# Quick smoke test
if __name__ == "__main__":
    B = 4
    model = PhysGenBearing(n_domains=4)
    out = model(
        signal=torch.randn(B, 1, 4096),
        fs=torch.tensor([12000., 24000., 64000., 97656.]),
        rpm=torch.tensor([1772., 700., 900., 1500.]),
        geom_vec=torch.randn(B, GEOM_VEC_DIM),
    )
    for k, v in out.items():
        if hasattr(v, "shape"):
            print(f"  {k}: {tuple(v.shape)}")
