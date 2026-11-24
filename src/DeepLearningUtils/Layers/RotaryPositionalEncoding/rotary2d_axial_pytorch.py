"""PyTorch 2D axial Rotary Positional Encoding (RoPE) for spatial and video attention.

This module provides the canonical PyTorch 2D axial RoPE implementation
(:class:`RotaryPositionalEncoding2D`), its validated configuration dataclass
(:class:`AxialRotary2DConfig`), and rotation helpers (:func:`rotate_half_2d` and
:func:`apply_rotary_pos_emb`).

2D Axial RoPE Best Practices
----------------------------
* **Self-attention vs. cross-attention (`apply_rope_q`, `apply_rope_k`):**
  Multiplicative RoPE relies on the orthogonal rotation identity
  ``<R(p_q) q, R(p_k) k> = q^T R(p_k - p_q) k`` (Su et al., 2021). Both queries and
  spatial keys should be rotated (``apply_rope_q=True, apply_rope_k=True``) in **both**
  spatial self-attention and spatial cross-attention so attention logits depend on
  shift-equivariant 2D relative displacement ``(p_k - p_q)``. Setting
  ``apply_rope_k=False`` reduces logits to ``q^T R(-p_q) k``, erasing key coordinates
  ``p_k`` and turning attention into position-conditioned content lookup over an
  unordered bag of keys. In SAM 2 (Ravi et al., 2024), asymmetry between self- and
  cross-attention applies to *additive* sinusoidal/temporal encodings
  (``pos_enc_at_cross_attn_keys=True, pos_enc_at_cross_attn_queries=False``) and key
  tiling/exclusion, whereas ``RoPEAttention`` rotates both ``Q`` and spatial ``K`` in
  both blocks.
* **Single-frame vs. multi-frame sequences and non-spatial tokens:**
  In single-frame self-attention (``t_q = t_k = 1``), ``Q`` and ``K`` share one
  ``(h, w)`` grid. In multi-frame cross-attention (``t_k >= 1`` concatenated memory
  frames), the spatial ``(k_h, k_w)`` rotary ``cos``/``sin`` tables tile ``t_k`` times
  across the spatial key prefix ``t_k * k_h * k_w`` (handled automatically in
  :class:`RotaryPositionalEncoding2D`, analogous to SAM 2's ``rope_k_repeat=True``).
  Trailing non-spatial tokens appended to ``K``/``V`` (such as mask-pooled
  ``object_pointers``) must bypass spatial RoPE by slicing only the leading spatial
  prefix before calling :class:`RotaryPositionalEncoding2D` (analogous to SAM 2's
  ``num_k_exclude_rope``).
* **Impact of 2D Axial RoPE on per-head dimension (`head_dim`):**
  Whereas 1D RoPE provides ``head_dim / 2`` rotation frequencies per head, 2D axial
  RoPE splits ``head_dim = key_dim // attention_heads`` evenly between height and width,
  leaving only ``head_dim // 4`` frequency pairs per spatial axis (requiring
  ``head_dim % 4 == 0``; Heo et al., 2024). Keep ``head_dim >= 32`` (``>= 8`` frequency
  pairs per axis, ideally ``48-64`` = ``12-16`` pairs per axis); initializing with
  ``head_dim < 32`` emits a :class:`UserWarning` in :class:`AxialRotary2DConfig`.

References
----------
* Su et al. (2021), *RoFormer: Enhanced Transformer with Rotary Position Embedding*,
  arXiv:2104.09864.
* Heo et al. (2024), *Rotary Position Embedding for Vision Transformer*, ECCV 2024,
  arXiv:2403.13298.
* Ravi et al. (2024), *SAM 2: Segment Anything in Images and Videos*, arXiv:2408.00714.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
import torch.nn as nn

MIN_RECOMMENDED_2D_ROPE_HEAD_DIM: int = 32


def _buffer_as_tensor(buffer: object, name: str) -> torch.Tensor:
    """Return a registered module buffer as a validated tensor.

    Parameters
    ----------
    buffer
        Buffer retrieved from an :class:`nn.Module`.
    name
        Buffer name used in error messages.

    Returns
    -------
    torch.Tensor
        Validated buffer tensor.

    Raises
    ------
    TypeError
        If ``buffer`` is not a :class:`torch.Tensor`.
    """
    if not isinstance(buffer, torch.Tensor):
        msg = f"{name} must be a tensor buffer, got {type(buffer).__name__}"
        raise TypeError(msg)
    return buffer


def warn_if_low_2d_rope_head_dim(head_dim: int, *, stacklevel: int = 2) -> None:
    """Emit a :class:`UserWarning` when 2D axial RoPE has fewer than 8 frequencies/axis.

    Parameters
    ----------
    head_dim
        Per-head query/key dimension ``key_dim // attention_heads``.
    stacklevel
        Warning stack level passed to :func:`warnings.warn`.
    """
    if head_dim < MIN_RECOMMENDED_2D_ROPE_HEAD_DIM:
        freqs_per_axis = head_dim // 4
        warnings.warn(
            f"2D axial RoPE configured with head_dim={head_dim} ({freqs_per_axis} rotary "
            f"frequency pairs per spatial axis); head_dim >= "
            f"{MIN_RECOMMENDED_2D_ROPE_HEAD_DIM} (>= 8 frequency pairs per axis, ideally "
            f"48-64) is recommended for 2D spatial/directional selectivity.",
            UserWarning,
            stacklevel=stacklevel,
        )


@dataclass(frozen=True)
class AxialRotary2DConfig:
    """Validated configuration for 2D axial Rotary Positional Encoding.

    Parameters
    ----------
    head_dim
        Per-head key/query dimension ``key_dim // attention_heads`` (must be a positive
        integer divisible by ``4``).
    max_h
        Maximum cached spatial grid height.
    max_w
        Maximum cached spatial grid width.
    theta
        Base rotary period parameter (default ``10000.0``).
    apply_rope_q
        Whether spatial query prefixes are rotated when applied inside attention.
    apply_rope_k
        Whether spatial key prefixes are rotated when applied inside attention.

    Notes
    -----
    ``__post_init__`` enforces structural preconditions with :class:`TypeError` /
    :class:`ValueError` and emits a :class:`UserWarning` when ``head_dim < 32``
    (``< 8`` rotary frequency pairs per spatial axis).
    """

    head_dim: int
    max_h: int = 128
    max_w: int = 128
    theta: float = 10000.0
    apply_rope_q: bool = True
    apply_rope_k: bool = True

    def __post_init__(self) -> None:
        """Validate 2D axial RoPE parameters and warn on starved frequency budgets."""
        if isinstance(self.head_dim, bool) or not isinstance(self.head_dim, int):
            msg = f"head_dim must be an integer, got {type(self.head_dim).__name__}"
            raise TypeError(msg)
        if self.head_dim <= 0 or self.head_dim % 4 != 0:
            msg = f"head_dim must be positive and divisible by 4 for 2D RoPE, got {self.head_dim}"
            raise ValueError(msg)

        if isinstance(self.max_h, bool) or not isinstance(self.max_h, int):
            msg = f"max_h must be an integer, got {type(self.max_h).__name__}"
            raise TypeError(msg)
        if isinstance(self.max_w, bool) or not isinstance(self.max_w, int):
            msg = f"max_w must be an integer, got {type(self.max_w).__name__}"
            raise TypeError(msg)
        if self.max_h <= 0 or self.max_w <= 0:
            msg = f"max_h and max_w must be positive, got ({self.max_h}, {self.max_w})"
            raise ValueError(msg)

        if isinstance(self.theta, bool) or not isinstance(self.theta, int | float):
            msg = f"theta must be a positive number, got {type(self.theta).__name__}"
            raise TypeError(msg)
        if self.theta <= 0:
            msg = f"theta must be positive, got {self.theta}"
            raise ValueError(msg)

        if not isinstance(self.apply_rope_q, bool):
            msg = f"apply_rope_q must be a bool, got {type(self.apply_rope_q).__name__}"
            raise TypeError(msg)
        if not isinstance(self.apply_rope_k, bool):
            msg = f"apply_rope_k must be a bool, got {type(self.apply_rope_k).__name__}"
            raise TypeError(msg)

        warn_if_low_2d_rope_head_dim(self.head_dim, stacklevel=3)

    @property
    def freqs_per_axis(self) -> int:
        """Number of rotary sine/cosine frequency pairs per spatial axis."""
        return self.head_dim // 4


def rotate_half_2d(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the hidden dimensions independently for height and width halves.

    Parameters
    ----------
    x
        Input tensor whose last dimension is divisible by ``4`` (even pairs for both
        height and width halves).

    Returns
    -------
    torch.Tensor
        Orthogonally rotated tensor matching ``x.shape``.
    """
    half_dim = x.shape[-1] // 2
    x_h = x[..., :half_dim]
    x_w = x[..., half_dim:]

    xh1, xh2 = x_h.chunk(2, dim=-1)
    x_h_rot = torch.cat((-xh2, xh1), dim=-1)

    xw1, xw2 = x_w.chunk(2, dim=-1)
    x_w_rot = torch.cat((-xw2, xw1), dim=-1)

    return torch.cat((x_h_rot, x_w_rot), dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply precomputed 2D rotary cosine and sine embeddings to ``x``.

    Parameters
    ----------
    x
        Query or key tensor with shape ``(b, heads, seq_len, head_dim)``.
    cos
        Broadcastable cosine frequencies.
    sin
        Broadcastable sine frequencies.

    Returns
    -------
    torch.Tensor
        Rotated query or key tensor matching ``x.shape``.
    """
    return (x * cos) + (rotate_half_2d(x) * sin)


class RotaryPositionalEncoding2D(nn.Module):
    """2D axial Rotary Positional Encoding applied independently to height and width halves.

    Parameters
    ----------
    head_dim
        Per-head key/query dimension ``key_dim // attention_heads`` (must be positive
        and divisible by ``4``), or an :class:`AxialRotary2DConfig` instance.
    max_h
        Maximum supported spatial height (ignored when ``head_dim`` or ``config`` is an
        :class:`AxialRotary2DConfig`).
    max_w
        Maximum supported spatial width (ignored when ``head_dim`` or ``config`` is an
        :class:`AxialRotary2DConfig`).
    theta
        Base rotary period parameter (default ``10000.0``).
    config
        Optional explicit :class:`AxialRotary2DConfig`.

    Notes
    -----
    2D axial RoPE splits ``head_dim`` evenly across height and width, yielding
    ``head_dim // 4`` rotary frequency pairs per spatial axis (prefer ``head_dim >= 32``,
    and ideally ``48-64``). When ``seq_len`` spans ``num_frames = seq_len // (h * w)``
    concatenated frames (``num_frames >= 1``), the single-frame ``(h, w)`` frequency
    grid is automatically tiled ``num_frames`` times along the sequence dimension.
    """

    def __init__(
        self,
        head_dim: int | AxialRotary2DConfig | None = None,
        max_h: int = 128,
        max_w: int = 128,
        *,
        theta: float = 10000.0,
        config: AxialRotary2DConfig | None = None,
    ) -> None:
        if config is not None:
            resolved_cfg = config
        elif isinstance(head_dim, AxialRotary2DConfig):
            resolved_cfg = head_dim
        elif head_dim is not None:
            resolved_cfg = AxialRotary2DConfig(
                head_dim=head_dim,
                max_h=max_h,
                max_w=max_w,
                theta=theta,
            )
        else:
            msg = "Either head_dim or config must be provided to RotaryPositionalEncoding2D"
            raise ValueError(msg)

        super().__init__()
        self.config = resolved_cfg
        self.head_dim = resolved_cfg.head_dim
        self.max_h = resolved_cfg.max_h
        self.max_w = resolved_cfg.max_w
        self.theta = resolved_cfg.theta

        dim_h = self.head_dim // 2
        dim_w = self.head_dim // 2

        inv_freq_h = 1.0 / (
            float(self.theta) ** (torch.arange(0, dim_h, 2).float() / max(1e-5, dim_h))
        )
        inv_freq_w = 1.0 / (
            float(self.theta) ** (torch.arange(0, dim_w, 2).float() / max(1e-5, dim_w))
        )

        pos_h = torch.arange(self.max_h).float()
        pos_w = torch.arange(self.max_w).float()

        freqs_h = torch.outer(pos_h, inv_freq_h)
        freqs_h = torch.cat((freqs_h, freqs_h), dim=-1)

        freqs_w = torch.outer(pos_w, inv_freq_w)
        freqs_w = torch.cat((freqs_w, freqs_w), dim=-1)

        freqs_h_grid = freqs_h.unsqueeze(1).expand(-1, self.max_w, -1)
        freqs_w_grid = freqs_w.unsqueeze(0).expand(self.max_h, -1, -1)
        freqs_2d = torch.cat([freqs_h_grid, freqs_w_grid], dim=-1)  # (max_h, max_w, head_dim)

        self.register_buffer("cos_cached", freqs_2d.cos())
        self.register_buffer("sin_cached", freqs_2d.sin())

    def _rotate_tokens(
        self,
        tokens: torch.Tensor,
        *,
        name: str,
        grid_h: int | None,
        grid_w: int | None,
        cos_cached: torch.Tensor,
        sin_cached: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.ndim != 4:
            msg = f"Expected {name} to be 4D (batch, heads, seq_len, head_dim), got {tokens.ndim}D"
            raise ValueError(msg)
        if tokens.shape[-1] != self.head_dim:
            msg = f"{name} head_dim {tokens.shape[-1]} doesn't match RoPE expected {self.head_dim}"
            raise ValueError(msg)
        if grid_h is None or grid_w is None or grid_h <= 0 or grid_w <= 0:
            msg = f"Positive {name}_h and {name}_w must be provided, got ({grid_h}, {grid_w})"
            raise ValueError(msg)
        if grid_h > self.max_h or grid_w > self.max_w:
            msg = (
                f"{name} spatial {(grid_h, grid_w)} exceeds cached RoPE max "
                f"{(self.max_h, self.max_w)}"
            )
            raise ValueError(msg)
        spatial_tokens = grid_h * grid_w
        if tokens.shape[2] % spatial_tokens != 0:
            msg = (
                f"{name} seq_len {tokens.shape[2]} must be divisible by "
                f"{grid_h}*{grid_w}={spatial_tokens}"
            )
            raise ValueError(msg)

        num_frames = tokens.shape[2] // spatial_tokens
        cos = cos_cached[:grid_h, :grid_w, :].reshape(-1, self.head_dim)
        sin = sin_cached[:grid_h, :grid_w, :].reshape(-1, self.head_dim)
        cos = cos.repeat(num_frames, 1).view(1, 1, num_frames * spatial_tokens, self.head_dim)
        sin = sin.repeat(num_frames, 1).view(1, 1, num_frames * spatial_tokens, self.head_dim)
        return apply_rotary_pos_emb(tokens, cos, sin)

    def forward(
        self,
        q: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
        q_h: int | None = None,
        q_w: int | None = None,
        k_h: int | None = None,
        k_w: int | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Apply 2D axial RoPE to optional ``q`` and ``k`` token sequences.

        Parameters
        ----------
        q
            Optional query tensor with shape ``(b, heads, t_q * q_h * q_w, head_dim)``.
        k
            Optional key tensor with shape ``(b, heads, t_k * k_h * k_w, head_dim)``.
        q_h, q_w
            Spatial grid dimensions for ``q`` (required when ``q`` is provided).
        k_h, k_w
            Spatial grid dimensions for ``k`` (required when ``k`` is provided).

        Returns
        -------
        tuple[torch.Tensor | None, torch.Tensor | None]
            Rotated ``(q_out, k_out)`` tensors (``None`` where the input was ``None``).
        """
        if q is None and k is None:
            return None, None

        cos_cached = _buffer_as_tensor(self.cos_cached, "cos_cached")
        sin_cached = _buffer_as_tensor(self.sin_cached, "sin_cached")

        q_out = (
            self._rotate_tokens(
                q,
                name="q",
                grid_h=q_h,
                grid_w=q_w,
                cos_cached=cos_cached,
                sin_cached=sin_cached,
            )
            if q is not None
            else None
        )
        k_out = (
            self._rotate_tokens(
                k,
                name="k",
                grid_h=k_h,
                grid_w=k_w,
                cos_cached=cos_cached,
                sin_cached=sin_cached,
            )
            if k is not None
            else None
        )
        return q_out, k_out
