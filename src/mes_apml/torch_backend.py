"""Compact PyTorch implementation of APML and the original squared-distance MES.

This preview intentionally implements no experimental/relative MES variants.
Class proxies belong to the classifier head and should be trainable Parameters.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _check_weights(weights: torch.Tensor) -> None:
    if not isinstance(weights, torch.Tensor) or weights.ndim != 2:
        raise ValueError("class proxies must be a tensor with shape (C, d)")
    if weights.shape[0] < 2 or weights.shape[1] < 1:
        raise ValueError("require C >= 2 and d >= 1")
    if not weights.is_floating_point():
        raise TypeError("class proxies must be floating point")


def apml_loss(embeddings: torch.Tensor,
              labels: torch.Tensor,
              class_weights: torch.Tensor) -> torch.Tensor:
    """Mean negative log of normalized Cauchy affinities to class proxies."""
    _check_weights(class_weights)
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (N, d)")
    if embeddings.shape[0] < 1 or embeddings.shape[1] != class_weights.shape[1]:
        raise ValueError("require nonempty embeddings and a matching feature dimension")
    if not embeddings.is_floating_point():
        raise TypeError("embeddings must be floating point")
    if embeddings.device != class_weights.device:
        raise ValueError("embeddings and class proxies must share a device")
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1 or labels.numel() != embeddings.shape[0]:
        raise ValueError("labels must have shape (N,)")
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError("labels must contain integer class indices")
    if labels.device != embeddings.device:
        raise ValueError("labels and embeddings must share a device")
    # Each logit is the log of a Cauchy affinity, not a dot product.
    diff = embeddings[:, None, :] - class_weights[None, :, :]
    logits = -torch.log1p(diff.square().sum(dim=-1))
    return F.cross_entropy(logits, labels.to(dtype=torch.int64), reduction="mean")


def mes_loss(class_weights: torch.Tensor) -> torch.Tensor:
    """Population variance of the C*(C-1)/2 unique SQUARED pair distances."""
    _check_weights(class_weights)
    c = class_weights.shape[0]
    left, right = torch.triu_indices(c, c, offset=1, device=class_weights.device)
    q = (class_weights[left] - class_weights[right]).square().sum(dim=1)
    return (q - q.mean()).square().mean()


def combined_loss(embeddings: torch.Tensor,
                  labels: torch.Tensor,
                  class_weights: torch.Tensor,
                  *, alpha: float = 1.0,
                  beta: float = 1.0) -> torch.Tensor:
    """alpha * APML + beta * original MES using the same class_weights."""
    for name, value in (("alpha", alpha), ("beta", beta)):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    return alpha * apml_loss(embeddings, labels, class_weights) + beta * mes_loss(class_weights)
