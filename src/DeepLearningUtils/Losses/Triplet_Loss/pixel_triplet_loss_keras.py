#!/usr/bin/env python3
"""
Pixel-based Triplet Loss for Whisker Segmentation.

This module implements a Keras 3 compatible triplet loss that operates on pixel-level
embeddings for semantic segmentation tasks. Designed for whisker tracking applications
where background vs whisker classification is important.
"""

import tensorflow as tf
import keras
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

# Import the existing triplet loss functions to reuse them
from .triplet_loss_keras import (
    _pairwise_distances,
    _get_anchor_positive_triplet_mask,
    _get_anchor_negative_triplet_mask,
    _get_triplet_mask,
    batch_hard_triplet_loss,
    batch_all_triplet_loss
)

try:
    # Keras 3 style import
    from keras.src.losses.loss import Loss
except Exception:
    try:
        # Public API for many installations
        from keras.losses import Loss
    except Exception:
        # Fallback to TensorFlow Keras if needed
        from tensorflow.keras.losses import Loss


# --- Added: masks that exclude background from positive pairs but allow it as negatives ---
def _get_anchor_positive_triplet_mask_exclude_background(labels: tf.Tensor) -> tf.Tensor:
    """
    Valid anchor-positive pairs where labels match and are non-background (>0).
    Shape: (batch, batch)
    """
    # Ensure labels is rank-1 [B]
    labels = tf.reshape(labels, [-1])
    batch_size = tf.shape(labels)[0]

    # i != j
    indices_not_equal = tf.logical_not(tf.eye(batch_size, dtype=tf.bool))

    # label match
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))  # (B,B)

    # anchor must be non-background; since labels_equal, positive is also non-bg
    anchor_non_bg = tf.greater(tf.expand_dims(labels, 0), 0)  # (B,B) broadcast along columns

    mask = tf.logical_and(indices_not_equal, tf.logical_and(labels_equal, anchor_non_bg))
    return mask


def _get_triplet_mask_exclude_background(labels: tf.Tensor) -> tf.Tensor:
    """
    Valid triplets (a,p,n) where:
    - a != p, a != n, p != n
    - labels[a] == labels[p] and labels[a] > 0
    - labels[a] != labels[n] (negative can be background or any other class)
    Shape: (batch, batch, batch)
    """
    # Ensure labels is rank-1 [B]
    labels = tf.reshape(labels, [-1])
    batch_size = tf.shape(labels)[0]

    # distinct indices masks
    eye = tf.eye(batch_size, dtype=tf.bool)  # (B,B)
    i_not_j = tf.logical_not(eye)  # (B,B)
    i_not_k = tf.logical_not(eye)  # (B,B)
    j_not_k = tf.logical_not(eye)  # (B,B)

    # Expand to (B,B,B)
    i_not_j = tf.expand_dims(i_not_j, 2)  # (B,B,1)
    i_not_k = tf.expand_dims(i_not_k, 1)  # (B,1,B)
    j_not_k = tf.expand_dims(j_not_k, 0)  # (1,B,B)

    distinct = tf.logical_and(tf.logical_and(i_not_j, i_not_k), j_not_k)  # (B,B,B)

    # labels equality/inequality across axes (a,p,n)
    labels_a = tf.expand_dims(tf.expand_dims(labels, 1), 2)   # (B,1,1)
    labels_p = tf.expand_dims(tf.expand_dims(labels, 0), 2)   # (1,B,1)
    labels_n = tf.expand_dims(tf.expand_dims(labels, 0), 1)   # (1,1,B)

    labels_equal_ap = tf.equal(labels_a, labels_p)            # (B,B,B)
    labels_not_equal_an = tf.not_equal(labels_a, labels_n)    # (B,B,B)

    # anchor must be non-background
    anchor_non_bg = tf.greater(labels_a, 0)                   # (B,B,B)

    mask = tf.logical_and(distinct, tf.logical_and(labels_equal_ap, anchor_non_bg))
    mask = tf.logical_and(mask, labels_not_equal_an)
    return mask


@dataclass
class PixelTripletConfig:
    """Configuration for pixel-based triplet loss."""
    margin: float = 1.0
    # Legacy parameters (kept for backward compatibility)
    background_pixels: int = 1000
    whisker_pixels: int = 500
    # New balanced sampling parameters
    max_samples_per_class: Optional[int] = None  # Maximum samples per class for balanced sampling
    use_balanced_sampling: bool = True  # Whether to use balanced sampling (recommended)
    strict_per_class_balancing: bool = False  # True per-class balancing
    prefer_graph_mode_strict: bool = True  # Use graph-mode strict balancing when possible
    distance_metric: str = "euclidean"
    triplet_strategy: str = "semi_hard"
    reduction: str = "mean"
    remove_easy_triplets: bool = False  # Whether to exclude easy triplets in batch_all mode
    memory_warning_threshold: int = 10_000_000  # Warn if pairwise matrix > 10M elements
    
    # Exact computation (loop-based) - avoids memory issues with large images
    use_exact: bool = False  # If True, compute distances via looping instead of full pairwise matrix
    batch_size_for_exact: int = 100  # Number of anchor pixels to process per iteration in exact mode

    # Class-balanced weighting for exact mode
    # When True: uses ALL pixels, accumulates loss per anchor class, then weights classes equally
    # This ensures each class contributes equally to the final loss despite pixel count imbalance
    class_balanced_weighting: bool = False  # Only used when use_exact=True

    def __post_init__(self):
        if self.margin <= 0:
            raise ValueError(f"margin must be > 0, got {self.margin}")
        if self.background_pixels <= 0:
            raise ValueError(f"background_pixels must be > 0, got {self.background_pixels}")
        if self.whisker_pixels <= 0:
            raise ValueError(f"whisker_pixels must be > 0, got {self.whisker_pixels}")
        if self.max_samples_per_class is not None and self.max_samples_per_class <= 0:
            raise ValueError(f"max_samples_per_class must be > 0 or None, got {self.max_samples_per_class}")
        if self.distance_metric not in ["euclidean", "cosine", "manhattan"]:
            raise ValueError(f"distance_metric must be one of ['euclidean', 'cosine', 'manhattan'], got {self.distance_metric}")
        if self.triplet_strategy not in ["hard", "semi_hard", "all"]:
            raise ValueError(f"triplet_strategy must be one of ['hard', 'semi_hard', 'all'], got {self.triplet_strategy}")
        if self.reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"reduction must be one of ['mean', 'sum', 'none'], got {self.reduction}")
        if self.batch_size_for_exact <= 0:
            raise ValueError(f"batch_size_for_exact must be > 0, got {self.batch_size_for_exact}")

        # class_balanced_weighting requires use_exact
        if self.class_balanced_weighting and not self.use_exact:
            print("Warning: class_balanced_weighting=True requires use_exact=True. "
                  "Setting use_exact=True automatically.")
            object.__setattr__(self, 'use_exact', True)

        # Provide guidance on the new parameters
        if self.use_balanced_sampling and self.max_samples_per_class is None:
            # Set a reasonable default based on legacy parameters
            self.max_samples_per_class = min(self.background_pixels, self.whisker_pixels)
        
        # Provide guidance about strict balancing requirements
        if self.strict_per_class_balancing:
            if self.prefer_graph_mode_strict:
                print("Info: strict_per_class_balancing=True with graph mode. "
                      "For full compatibility, compile your model with jit_compile=False to avoid XLA issues.")
            else:
                print("Info: strict_per_class_balancing=True with eager mode preference. "
                      "Graph mode fallback available if needed.")


class PixelTripletLoss(Loss):
    """
    Pixel-based triplet loss for semantic segmentation.
    
    This loss operates on pixel-level embeddings, sampling a specified number of pixels
    per class and computing triplet loss with semi-hard or hard negative mining.
    
    Key Features:
    - Upsamples embeddings to match label resolution (preserves thin structures)
    - Configurable distance metrics (Euclidean, cosine, Manhattan)
    - Balanced sampling to prevent class imbalance issues
    - Graph-mode compatible with strict per-class balancing
    - Literature-compliant easy triplet handling
    
    Args:
        config: PixelTripletConfig instance with loss parameters
        name: Name for the loss function
        
    Note:
        When label resolution > embedding resolution, embeddings are upsampled using
        nearest neighbor interpolation. This prevents aliasing artifacts that would
        occur when downsampling thin structures like whiskers.
    """
    
    def __init__(
        self,
        config: Optional[PixelTripletConfig] = None,
        name: str = "pixel_triplet_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.config = config or PixelTripletConfig()
        
    def get_config(self):
        """Return configuration for serialization."""
        base_config = super().get_config()
        config_dict = {
            'margin': self.config.margin,
            'background_pixels': self.config.background_pixels,
            'whisker_pixels': self.config.whisker_pixels,
            'max_samples_per_class': self.config.max_samples_per_class,
            'use_balanced_sampling': self.config.use_balanced_sampling,
            'strict_per_class_balancing': self.config.strict_per_class_balancing,
            'prefer_graph_mode_strict': self.config.prefer_graph_mode_strict,
            'distance_metric': self.config.distance_metric,
            'triplet_strategy': self.config.triplet_strategy,
            'reduction': self.config.reduction,
            'remove_easy_triplets': self.config.remove_easy_triplets,
            'memory_warning_threshold': self.config.memory_warning_threshold,
            'use_exact': self.config.use_exact,
            'batch_size_for_exact': self.config.batch_size_for_exact,
            'class_balanced_weighting': self.config.class_balanced_weighting,
        }
        return {**base_config, **config_dict}
    
    @classmethod
    def from_config(cls, config):
        """Create instance from configuration."""
        triplet_config = PixelTripletConfig(
            margin=config.pop('margin', 1.0),
            background_pixels=config.pop('background_pixels', 1000),
            whisker_pixels=config.pop('whisker_pixels', 500),
            max_samples_per_class=config.pop('max_samples_per_class', None),
            use_balanced_sampling=config.pop('use_balanced_sampling', True),
            strict_per_class_balancing=config.pop('strict_per_class_balancing', False),
            prefer_graph_mode_strict=config.pop('prefer_graph_mode_strict', True),
            distance_metric=config.pop('distance_metric', 'euclidean'),
            triplet_strategy=config.pop('triplet_strategy', 'semi_hard'),
            reduction=config.pop('reduction', 'mean'),
            remove_easy_triplets=config.pop('remove_easy_triplets', False),
            memory_warning_threshold=config.pop('memory_warning_threshold', 10_000_000),
            use_exact=config.pop('use_exact', False),
            batch_size_for_exact=config.pop('batch_size_for_exact', 100),
            class_balanced_weighting=config.pop('class_balanced_weighting', False),
        )
        return cls(config=triplet_config, **config)
    
    def _compute_pairwise_distances(self, embeddings: tf.Tensor) -> tf.Tensor:
        """
        Compute pairwise distances using the specified distance metric.
        
        Args:
            embeddings: tensor of shape (batch_size, embed_dim)
            
        Returns:
            pairwise_distances: tensor of shape (batch_size, batch_size)
        """
        if self.config.distance_metric == "euclidean":
            return _pairwise_distances(embeddings, squared=False)
        elif self.config.distance_metric == "cosine":
            return self._cosine_distances(embeddings)
        elif self.config.distance_metric == "manhattan":
            return self._manhattan_distances(embeddings)
        else:
            raise ValueError(f"Unsupported distance metric: {self.config.distance_metric}")
    
    def _cosine_distances(self, embeddings: tf.Tensor) -> tf.Tensor:
        """
        Compute pairwise cosine distances.
        
        Cosine distance = 1 - cosine_similarity
        Cosine similarity = dot(a,b) / (||a|| * ||b||)
        
        Args:
            embeddings: tensor of shape (batch_size, embed_dim)
            
        Returns:
            pairwise_distances: tensor of shape (batch_size, batch_size)
        """
        # Normalize embeddings to unit vectors
        normalized_embeddings = tf.nn.l2_normalize(embeddings, axis=1)
        
        # Compute cosine similarities
        cosine_similarities = tf.matmul(normalized_embeddings, normalized_embeddings, transpose_b=True)
        
        # Convert to cosine distances
        cosine_distances = 1.0 - cosine_similarities
        
        # Ensure distances are non-negative (handle numerical errors)
        cosine_distances = tf.maximum(cosine_distances, 0.0)
        
        return cosine_distances
    
    def _manhattan_distances(self, embeddings: tf.Tensor) -> tf.Tensor:
        """
        Compute pairwise Manhattan (L1) distances.
        
        Manhattan distance = sum(|a_i - b_i|)
        
        Args:
            embeddings: tensor of shape (batch_size, embed_dim)
            
        Returns:
            pairwise_distances: tensor of shape (batch_size, batch_size)
        """
        # Expand embeddings for broadcasting
        # embeddings_i: (batch_size, 1, embed_dim)
        # embeddings_j: (1, batch_size, embed_dim)
        embeddings_i = tf.expand_dims(embeddings, 1)
        embeddings_j = tf.expand_dims(embeddings, 0)
        
        # Compute absolute differences and sum along the embedding dimension
        manhattan_distances = tf.reduce_sum(tf.abs(embeddings_i - embeddings_j), axis=2)
        
        return manhattan_distances
    
    def _batch_hard_triplet_loss_custom(self, labels: tf.Tensor, embeddings: tf.Tensor) -> tf.Tensor:
        """
        Custom implementation of batch hard triplet loss with configurable distance metrics.
        
        This is adapted from the original batch_hard_triplet_loss but uses our custom distance function.
        """
        # Get the pairwise distance matrix using the configured metric
        pairwise_dist = self._compute_pairwise_distances(embeddings)
        
        # For each anchor, get the hardest positive (exclude background positives)
        mask_anchor_positive = _get_anchor_positive_triplet_mask_exclude_background(labels)
        mask_anchor_positive = keras.ops.cast(mask_anchor_positive, "float32")
        
        # We put to 0 any element where (a, p) is not valid 
        anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)
        
        # shape (batch_size, 1)
        hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)
        
        # For each anchor, get the hardest negative (background allowed as negative)
        mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
        mask_anchor_negative = keras.ops.cast(mask_anchor_negative, "float32")
        
        # We add the maximum value in each row to the invalid negatives
        max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
        
        # shape (batch_size,)
        hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)
        scaling = tf.reduce_mean(anchor_negative_dist, axis=1, keepdims=True) + 1e-16
        
        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        triplet_loss = tf.maximum(
            (hardest_positive_dist - hardest_negative_dist) / scaling + self.config.margin,
            0.0
        )
        
        # Get final mean triplet loss
        triplet_loss = tf.reduce_mean(triplet_loss)
        
        return triplet_loss
    
    def _batch_all_triplet_loss_custom(self, labels: tf.Tensor, embeddings: tf.Tensor) -> tf.Tensor:
        """
        Custom implementation of batch all triplet loss with configurable distance metrics.
        
        This implementation follows the typical "batch all" behavior from literature:
        - By default, includes easy triplets (negative losses) as they provide gradient information
        - Optionally excludes easy triplets if remove_easy_triplets=True for harder training
        """
        # Get the pairwise distance matrix using the configured metric
        pairwise_dist = self._compute_pairwise_distances(embeddings)
        
        anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
        anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
        
        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.config.margin
        
        # Put to zero the invalid triplets (exclude background positives)
        mask = _get_triplet_mask_exclude_background(labels)
        mask = keras.ops.cast(mask, "float32")
        triplet_loss = tf.multiply(mask, triplet_loss)
        
        num_valid_triplets = tf.reduce_sum(mask)
        
        # Handle easy triplets based on configuration
        if self.config.remove_easy_triplets:
            # Remove negative losses (i.e. the easy triplets) - focus on hard examples
            triplet_loss = tf.maximum(triplet_loss, 0.0)
            
            # Count number of positive triplets (where triplet_loss > 0)
            valid_triplets = keras.ops.cast(tf.greater(triplet_loss, 1e-16), "float32")
            num_positive_triplets = tf.reduce_sum(valid_triplets)
            
            # Get final mean triplet loss over the positive valid triplets
            triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)
        else:
            # Include easy triplets (typical literature behavior) - use all valid triplets
            # Easy triplets have negative loss values but still provide gradient information
            triplet_loss = tf.reduce_sum(triplet_loss) / (num_valid_triplets + 1e-16)
        
        return triplet_loss

    def _compute_distances_from_anchors(
        self,
        anchor_embeddings: tf.Tensor,
        all_embeddings: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute distances from anchor embeddings to all embeddings.

        Args:
            anchor_embeddings: (B, D) tensor of anchor embeddings
            all_embeddings: (N, D) tensor of all embeddings

        Returns:
            distances: (B, N) tensor of distances
        """
        if self.config.distance_metric == "euclidean":
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
            anchor_sq_norm = tf.reduce_sum(tf.square(anchor_embeddings), axis=1, keepdims=True)  # (B, 1)
            all_sq_norm = tf.reduce_sum(tf.square(all_embeddings), axis=1)  # (N,)
            dot_product = tf.matmul(anchor_embeddings, all_embeddings, transpose_b=True)  # (B, N)
            sq_distances = anchor_sq_norm - 2.0 * dot_product + tf.expand_dims(all_sq_norm, 0)
            sq_distances = tf.maximum(sq_distances, 0.0)
            return tf.sqrt(sq_distances + 1e-16)
        elif self.config.distance_metric == "cosine":
            # Normalize embeddings
            anchor_norm = tf.nn.l2_normalize(anchor_embeddings, axis=1)
            all_norm = tf.nn.l2_normalize(all_embeddings, axis=1)
            similarities = tf.matmul(anchor_norm, all_norm, transpose_b=True)
            return tf.maximum(1.0 - similarities, 0.0)
        else:  # manhattan
            anchor_exp = tf.expand_dims(anchor_embeddings, 1)  # (B, 1, D)
            all_exp = tf.expand_dims(all_embeddings, 0)  # (1, N, D)
            return tf.reduce_sum(tf.abs(anchor_exp - all_exp), axis=2)

    def _batch_hard_triplet_loss_exact(self, labels: tf.Tensor, embeddings: tf.Tensor) -> tf.Tensor:
        """
        Batch hard triplet loss using loop-based exact computation.

        This iterates through anchors in batches, computing the hardest positive
        and hardest negative for each anchor against ALL other embeddings.

        Memory usage is O(batch_size * N) instead of O(N^2).
        """
        num_pixels = tf.shape(embeddings)[0]
        batch_size = self.config.batch_size_for_exact
        num_batches = (num_pixels + batch_size - 1) // batch_size

        # Pre-compute label properties
        all_labels = labels  # (N,)

        # Initialize accumulators
        loss_sum = tf.constant(0.0, dtype=tf.float32)
        valid_anchor_count = tf.constant(0.0, dtype=tf.float32)

        def compute_batch_loss(batch_idx, accumulators):
            """Compute hard triplet loss for one batch of anchors."""
            loss_sum, valid_anchor_count = accumulators

            start_idx = batch_idx * batch_size
            end_idx = tf.minimum(start_idx + batch_size, num_pixels)

            anchor_embeddings = embeddings[start_idx:end_idx]  # (B, D)
            anchor_labels = labels[start_idx:end_idx]  # (B,)
            actual_batch_size = end_idx - start_idx

            # Compute distances from anchors to all pixels
            distances = self._compute_distances_from_anchors(anchor_embeddings, embeddings)  # (B, N)

            # Create label masks
            anchor_labels_exp = tf.expand_dims(anchor_labels, 1)  # (B, 1)
            all_labels_exp = tf.expand_dims(all_labels, 0)  # (1, N)

            # Positive mask: same class, not self, anchor is non-background
            same_class = tf.equal(anchor_labels_exp, all_labels_exp)  # (B, N)
            anchor_non_bg = tf.greater(anchor_labels, 0)  # (B,) - anchors must be non-background
            anchor_non_bg_exp = tf.expand_dims(anchor_non_bg, 1)  # (B, 1)

            # Self mask
            batch_indices = tf.range(start_idx, end_idx)
            all_indices = tf.range(num_pixels)
            self_mask = tf.equal(
                tf.expand_dims(batch_indices, 1),
                tf.expand_dims(all_indices, 0)
            )
            not_self = tf.logical_not(self_mask)

            # Valid positive: same class, not self, anchor is foreground
            positive_mask = tf.logical_and(same_class, not_self)
            positive_mask = tf.logical_and(positive_mask, anchor_non_bg_exp)

            # Negative mask: different class
            negative_mask = tf.logical_not(same_class)

            # For hardest positive: max distance among positives
            positive_mask_float = tf.cast(positive_mask, tf.float32)
            # Set invalid positions to -inf for max
            masked_positive_dist = distances * positive_mask_float + (1.0 - positive_mask_float) * (-1e9)
            hardest_positive_dist = tf.reduce_max(masked_positive_dist, axis=1)  # (B,)

            # For hardest negative: min distance among negatives
            negative_mask_float = tf.cast(negative_mask, tf.float32)
            # Set invalid positions to +inf for min
            masked_negative_dist = distances + (1.0 - negative_mask_float) * 1e9
            hardest_negative_dist = tf.reduce_min(masked_negative_dist, axis=1)  # (B,)

            # Scaling factor (mean of negative distances)
            scaling = tf.reduce_sum(distances * negative_mask_float, axis=1) / (tf.reduce_sum(negative_mask_float, axis=1) + 1e-16)

            # Compute triplet loss for each anchor
            triplet_loss = tf.maximum(
                (hardest_positive_dist - hardest_negative_dist) / (scaling + 1e-16) + self.config.margin,
                0.0
            )

            # Only count anchors that have valid positives (foreground anchors with same-class pixels)
            has_valid_positive = tf.reduce_any(positive_mask, axis=1)  # (B,)
            has_valid_negative = tf.reduce_any(negative_mask, axis=1)  # (B,)
            valid_anchor = tf.logical_and(has_valid_positive, has_valid_negative)
            valid_anchor_float = tf.cast(valid_anchor, tf.float32)

            batch_loss = tf.reduce_sum(triplet_loss * valid_anchor_float)
            batch_count = tf.reduce_sum(valid_anchor_float)

            return (loss_sum + batch_loss, valid_anchor_count + batch_count)

        # Loop through batches
        def cond(batch_idx, accumulators):
            return batch_idx < num_batches

        def body(batch_idx, accumulators):
            new_accumulators = compute_batch_loss(batch_idx, accumulators)
            return batch_idx + 1, new_accumulators

        _, (final_loss_sum, final_count) = tf.while_loop(
            cond,
            body,
            loop_vars=(tf.constant(0), (loss_sum, valid_anchor_count)),
            parallel_iterations=1
        )

        return final_loss_sum / (final_count + 1e-16)

    def _batch_all_triplet_loss_exact(self, labels: tf.Tensor, embeddings: tf.Tensor) -> tf.Tensor:
        """
        Batch all triplet loss using loop-based exact computation.

        This iterates through anchors in batches, and for each anchor iterates
        through its positives, computing triplet loss against ALL negatives.

        When remove_easy_triplets=True, we properly compute max(0, d_ap - d_an + margin)
        for each individual triplet before summing.

        When remove_easy_triplets=False, we can use an efficient approximation
        since all triplets contribute to the gradient.

        Memory usage is O(batch_size * N) instead of O(N^3).
        """
        num_pixels = tf.shape(embeddings)[0]
        batch_size = self.config.batch_size_for_exact
        num_batches = (num_pixels + batch_size - 1) // batch_size

        all_labels = labels

        # Initialize accumulators
        loss_sum = tf.constant(0.0, dtype=tf.float32)
        triplet_count = tf.constant(0.0, dtype=tf.float32)
        positive_triplet_count = tf.constant(0.0, dtype=tf.float32)  # For remove_easy_triplets

        def compute_batch_loss(batch_idx, accumulators):
            """Compute batch all triplet loss for one batch of anchors."""
            loss_sum, triplet_count, positive_triplet_count = accumulators

            start_idx = batch_idx * batch_size
            end_idx = tf.minimum(start_idx + batch_size, num_pixels)

            anchor_embeddings = embeddings[start_idx:end_idx]  # (B, D)
            anchor_labels = labels[start_idx:end_idx]  # (B,)
            actual_batch_size = end_idx - start_idx

            # Compute distances from anchors to all pixels
            distances = self._compute_distances_from_anchors(anchor_embeddings, embeddings)  # (B, N)

            # Create label masks
            anchor_labels_exp = tf.expand_dims(anchor_labels, 1)  # (B, 1)
            all_labels_exp = tf.expand_dims(all_labels, 0)  # (1, N)

            same_class = tf.equal(anchor_labels_exp, all_labels_exp)  # (B, N)
            diff_class = tf.logical_not(same_class)

            # Anchor must be non-background
            anchor_non_bg = tf.greater(anchor_labels, 0)  # (B,)
            anchor_non_bg_exp = tf.expand_dims(anchor_non_bg, 1)  # (B, 1)

            # Self mask
            batch_indices = tf.range(start_idx, end_idx)
            all_indices = tf.range(num_pixels)
            self_mask = tf.equal(
                tf.expand_dims(batch_indices, 1),
                tf.expand_dims(all_indices, 0)
            )
            not_self = tf.logical_not(self_mask)

            # Valid positive: same class, not self, anchor is foreground
            positive_mask = tf.logical_and(same_class, not_self)
            positive_mask = tf.logical_and(positive_mask, anchor_non_bg_exp)  # (B, N)

            # Valid negative: different class
            negative_mask = diff_class  # (B, N)

            positive_mask_float = tf.cast(positive_mask, tf.float32)
            negative_mask_float = tf.cast(negative_mask, tf.float32)

            # Count positives and negatives per anchor
            pos_count = tf.reduce_sum(positive_mask_float, axis=1)  # (B,)
            neg_count = tf.reduce_sum(negative_mask_float, axis=1)  # (B,)

            if self.config.remove_easy_triplets:
                # Need to compute each triplet loss individually: max(0, d_ap - d_an + margin)
                # For each anchor, for each positive, compute loss against all negatives

                # d_ap: (B, N) distances to positives (masked)
                # d_an: (B, N) distances to negatives (masked)

                # For each anchor i with positive j:
                #   triplet_loss_ij = sum over negatives k of: max(0, d[i,j] - d[i,k] + margin)

                # Expand for broadcasting:
                # d_ap: (B, N_pos, 1) - need to iterate over positives
                # d_an: (B, 1, N_neg) - all negatives

                # To avoid B*N*N tensor, we compute per-anchor contribution
                # For each anchor: sum of max(0, d_pos - d_neg + margin) over all pos-neg pairs

                # Efficient approach: for each anchor
                # positive_dists[i] = distances[i] where positive_mask[i] is True
                # negative_dists[i] = distances[i] where negative_mask[i] is True
                # Then compute: sum over pos of sum over neg of max(0, pos - neg + margin)

                # This can be rewritten as:
                # For pos p and neg n: max(0, p - n + margin)
                # = max(0, p - n + margin)
                # if p - n + margin > 0: contributes (p - n + margin)
                # else: contributes 0

                # For each anchor, we can compute this by:
                # 1. Sort negative distances
                # 2. For each positive, find how many negatives are "hard" (d_neg < d_pos + margin)
                # 3. Sum up contributions

                # Simpler but still exact: nested loop with tf.while_loop
                # For each anchor, for each positive, compute contribution

                # Even simpler: compute all pairwise (pos, neg) differences for each anchor
                # d_pos_expanded: (B, N, 1) with invalid set to 0
                # d_neg_expanded: (B, 1, N) with invalid set to large value
                # diff = d_pos_expanded - d_neg_expanded + margin  # (B, N, N)
                # But this is B*N*N which we want to avoid

                # Compromise: iterate per anchor within the batch
                def compute_anchor_loss(anchor_idx):
                    """Compute triplet loss for a single anchor."""
                    anchor_distances = distances[anchor_idx]  # (N,)
                    anchor_pos_mask = positive_mask[anchor_idx]  # (N,)
                    anchor_neg_mask = negative_mask[anchor_idx]  # (N,)

                    # Get positive and negative distances
                    pos_indices = tf.where(anchor_pos_mask)[:, 0]
                    neg_indices = tf.where(anchor_neg_mask)[:, 0]

                    pos_dists = tf.gather(anchor_distances, pos_indices)  # (num_pos,)
                    neg_dists = tf.gather(anchor_distances, neg_indices)  # (num_neg,)

                    num_pos = tf.shape(pos_dists)[0]
                    num_neg = tf.shape(neg_dists)[0]

                    # Compute all pairwise triplet losses
                    # pos_dists: (num_pos,) -> (num_pos, 1)
                    # neg_dists: (num_neg,) -> (1, num_neg)
                    pos_expanded = tf.expand_dims(pos_dists, 1)  # (num_pos, 1)
                    neg_expanded = tf.expand_dims(neg_dists, 0)  # (1, num_neg)

                    # Triplet loss for each pair
                    triplet_losses = pos_expanded - neg_expanded + self.config.margin  # (num_pos, num_neg)
                    triplet_losses = tf.maximum(triplet_losses, 0.0)  # Remove easy triplets

                    # Sum all losses and count positive (non-zero) triplets
                    anchor_loss = tf.reduce_sum(triplet_losses)
                    anchor_total_triplets = tf.cast(num_pos * num_neg, tf.float32)
                    anchor_positive_triplets = tf.reduce_sum(tf.cast(triplet_losses > 1e-16, tf.float32))

                    return anchor_loss, anchor_total_triplets, anchor_positive_triplets

                # Map over anchors in this batch
                anchor_results = tf.map_fn(
                    compute_anchor_loss,
                    tf.range(actual_batch_size),
                    fn_output_signature=(
                        tf.TensorSpec([], dtype=tf.float32),
                        tf.TensorSpec([], dtype=tf.float32),
                        tf.TensorSpec([], dtype=tf.float32),
                    )
                )

                batch_loss = tf.reduce_sum(anchor_results[0])
                batch_total_triplets = tf.reduce_sum(anchor_results[1])
                batch_positive_triplets = tf.reduce_sum(anchor_results[2])

                return (
                    loss_sum + batch_loss,
                    triplet_count + batch_total_triplets,
                    positive_triplet_count + batch_positive_triplets
                )
            else:
                # Without remove_easy_triplets, we can use the efficient approximation
                # since all triplets contribute (even negative ones)
                # Loss = sum over all triplets of (d_ap - d_an + margin)
                #      = sum over anchors of: (sum_pos d_ap) * num_neg - (sum_neg d_an) * num_pos + margin * num_pos * num_neg

                # Sum of positive distances per anchor
                pos_dist_sum = tf.reduce_sum(distances * positive_mask_float, axis=1)  # (B,)

                # Sum of negative distances per anchor
                neg_dist_sum = tf.reduce_sum(distances * negative_mask_float, axis=1)  # (B,)

                # Number of triplets per anchor
                triplets_per_anchor = pos_count * neg_count  # (B,)

                # Total loss contribution per anchor:
                # sum over pos,neg of (d_pos - d_neg + margin)
                # = sum_pos(d_pos) * num_neg - sum_neg(d_neg) * num_pos + margin * num_pos * num_neg
                anchor_loss = (
                    pos_dist_sum * neg_count -
                    neg_dist_sum * pos_count +
                    self.config.margin * triplets_per_anchor
                )

                # Only count anchors with valid triplets
                valid_anchor = triplets_per_anchor > 0
                valid_anchor_float = tf.cast(valid_anchor, tf.float32)

                batch_loss = tf.reduce_sum(anchor_loss * valid_anchor_float)
                batch_count = tf.reduce_sum(triplets_per_anchor * valid_anchor_float)

                return (loss_sum + batch_loss, triplet_count + batch_count, positive_triplet_count)

        # Loop through batches
        def cond(batch_idx, accumulators):
            return batch_idx < num_batches

        def body(batch_idx, accumulators):
            new_accumulators = compute_batch_loss(batch_idx, accumulators)
            return batch_idx + 1, new_accumulators

        initial_accumulators = (loss_sum, triplet_count, positive_triplet_count)
        _, final_accumulators = tf.while_loop(
            cond,
            body,
            loop_vars=(tf.constant(0), initial_accumulators),
            parallel_iterations=1
        )

        final_loss_sum, final_count, final_positive_count = final_accumulators

        if self.config.remove_easy_triplets:
            # Divide by number of positive (non-zero loss) triplets
            return final_loss_sum / (final_positive_count + 1e-16)
        else:
            # Divide by total number of triplets
            return final_loss_sum / (final_count + 1e-16)

    def _batch_hard_triplet_loss_exact_class_balanced(
        self, labels: tf.Tensor, embeddings: tf.Tensor
    ) -> tf.Tensor:
        """
        Batch hard triplet loss with class-balanced weighting.

        Uses ALL pixels, accumulates loss per anchor class, then weights
        each class equally in the final loss. This ensures each class
        contributes equally despite pixel count imbalance.

        For example, if 80% of pixels are background and 5% are each of 4 classes,
        each class (including background) will contribute 1/5 of the final loss.
        """
        num_pixels = tf.shape(embeddings)[0]
        batch_size = self.config.batch_size_for_exact
        num_batches = (num_pixels + batch_size - 1) // batch_size

        all_labels = labels

        # Get unique foreground classes (exclude background=0 as anchors)
        unique_classes, _ = tf.unique(labels)
        # Filter to only foreground classes (> 0)
        foreground_classes = tf.boolean_mask(unique_classes, unique_classes > 0)
        num_fg_classes = tf.shape(foreground_classes)[0]

        # We'll accumulate loss and count per class using a fixed-size tensor
        # Use a large enough size to handle any number of classes
        max_classes = 100  # Maximum number of foreground classes supported

        # Accumulators: loss_per_class[c] and count_per_class[c] for class c
        loss_per_class = tf.zeros([max_classes], dtype=tf.float32)
        count_per_class = tf.zeros([max_classes], dtype=tf.float32)

        def compute_batch_loss(batch_idx, accumulators):
            """Compute hard triplet loss for one batch of anchors, accumulating per class."""
            loss_per_class, count_per_class = accumulators

            start_idx = batch_idx * batch_size
            end_idx = tf.minimum(start_idx + batch_size, num_pixels)

            anchor_embeddings = embeddings[start_idx:end_idx]
            anchor_labels = labels[start_idx:end_idx]

            # Compute distances from anchors to all pixels
            distances = self._compute_distances_from_anchors(anchor_embeddings, embeddings)

            # Create label masks
            anchor_labels_exp = tf.expand_dims(anchor_labels, 1)
            all_labels_exp = tf.expand_dims(all_labels, 0)

            same_class = tf.equal(anchor_labels_exp, all_labels_exp)
            anchor_non_bg = tf.greater(anchor_labels, 0)
            anchor_non_bg_exp = tf.expand_dims(anchor_non_bg, 1)

            # Self mask
            batch_indices = tf.range(start_idx, end_idx)
            all_indices = tf.range(num_pixels)
            self_mask = tf.equal(
                tf.expand_dims(batch_indices, 1),
                tf.expand_dims(all_indices, 0)
            )
            not_self = tf.logical_not(self_mask)

            # Valid positive: same class, not self, anchor is foreground
            positive_mask = tf.logical_and(same_class, not_self)
            positive_mask = tf.logical_and(positive_mask, anchor_non_bg_exp)

            # Negative mask: different class
            negative_mask = tf.logical_not(same_class)

            positive_mask_float = tf.cast(positive_mask, tf.float32)
            negative_mask_float = tf.cast(negative_mask, tf.float32)

            # Hardest positive
            masked_positive_dist = distances * positive_mask_float + (1.0 - positive_mask_float) * (-1e9)
            hardest_positive_dist = tf.reduce_max(masked_positive_dist, axis=1)

            # Hardest negative
            masked_negative_dist = distances + (1.0 - negative_mask_float) * 1e9
            hardest_negative_dist = tf.reduce_min(masked_negative_dist, axis=1)

            # Scaling factor
            neg_count = tf.reduce_sum(negative_mask_float, axis=1)
            scaling = tf.reduce_sum(distances * negative_mask_float, axis=1) / (neg_count + 1e-16)

            # Triplet loss per anchor
            triplet_loss = tf.maximum(
                (hardest_positive_dist - hardest_negative_dist) / (scaling + 1e-16) + self.config.margin,
                0.0
            )

            # Valid anchors (have positives and negatives)
            has_valid_positive = tf.reduce_any(positive_mask, axis=1)
            has_valid_negative = tf.reduce_any(negative_mask, axis=1)
            valid_anchor = tf.logical_and(has_valid_positive, has_valid_negative)
            valid_anchor_float = tf.cast(valid_anchor, tf.float32)

            # Accumulate per anchor class
            # For each anchor, add its loss to the corresponding class bucket
            anchor_labels_clipped = tf.clip_by_value(anchor_labels, 0, max_classes - 1)

            # Use scatter_nd to accumulate losses per class
            indices = tf.expand_dims(anchor_labels_clipped, 1)
            batch_loss_per_class = tf.scatter_nd(
                indices,
                triplet_loss * valid_anchor_float,
                [max_classes]
            )
            batch_count_per_class = tf.scatter_nd(
                indices,
                valid_anchor_float,
                [max_classes]
            )

            return (
                loss_per_class + batch_loss_per_class,
                count_per_class + batch_count_per_class
            )

        # Loop through batches
        def cond(batch_idx, accumulators):
            return batch_idx < num_batches

        def body(batch_idx, accumulators):
            new_accumulators = compute_batch_loss(batch_idx, accumulators)
            return batch_idx + 1, new_accumulators

        _, (final_loss_per_class, final_count_per_class) = tf.while_loop(
            cond,
            body,
            loop_vars=(tf.constant(0), (loss_per_class, count_per_class)),
            parallel_iterations=1
        )

        # Compute mean loss per class (only for classes with valid triplets)
        mean_loss_per_class = final_loss_per_class / (final_count_per_class + 1e-16)

        # Weight each class equally: average over classes with non-zero counts
        valid_class_mask = final_count_per_class > 0
        num_valid_classes = tf.reduce_sum(tf.cast(valid_class_mask, tf.float32))

        # Final loss: average of per-class mean losses
        total_loss = tf.reduce_sum(mean_loss_per_class * tf.cast(valid_class_mask, tf.float32))
        final_loss = total_loss / (num_valid_classes + 1e-16)

        return final_loss

    def _batch_all_triplet_loss_exact_class_balanced(
        self, labels: tf.Tensor, embeddings: tf.Tensor
    ) -> tf.Tensor:
        """
        Batch all triplet loss with class-balanced weighting.

        Uses ALL pixels, computes triplet loss for all valid triplets,
        accumulates per anchor class, then weights each class equally.
        """
        num_pixels = tf.shape(embeddings)[0]
        batch_size = self.config.batch_size_for_exact
        num_batches = (num_pixels + batch_size - 1) // batch_size

        all_labels = labels
        max_classes = 100

        # Accumulators per class
        loss_per_class = tf.zeros([max_classes], dtype=tf.float32)
        count_per_class = tf.zeros([max_classes], dtype=tf.float32)
        positive_count_per_class = tf.zeros([max_classes], dtype=tf.float32)  # For remove_easy_triplets

        def compute_batch_loss(batch_idx, accumulators):
            """Compute batch all triplet loss for one batch of anchors."""
            loss_per_class, count_per_class, positive_count_per_class = accumulators

            start_idx = batch_idx * batch_size
            end_idx = tf.minimum(start_idx + batch_size, num_pixels)
            actual_batch_size = end_idx - start_idx

            anchor_embeddings = embeddings[start_idx:end_idx]
            anchor_labels = labels[start_idx:end_idx]

            distances = self._compute_distances_from_anchors(anchor_embeddings, embeddings)

            anchor_labels_exp = tf.expand_dims(anchor_labels, 1)
            all_labels_exp = tf.expand_dims(all_labels, 0)

            same_class = tf.equal(anchor_labels_exp, all_labels_exp)
            diff_class = tf.logical_not(same_class)
            anchor_non_bg = tf.greater(anchor_labels, 0)
            anchor_non_bg_exp = tf.expand_dims(anchor_non_bg, 1)

            batch_indices = tf.range(start_idx, end_idx)
            all_indices = tf.range(num_pixels)
            self_mask = tf.equal(
                tf.expand_dims(batch_indices, 1),
                tf.expand_dims(all_indices, 0)
            )
            not_self = tf.logical_not(self_mask)

            positive_mask = tf.logical_and(same_class, not_self)
            positive_mask = tf.logical_and(positive_mask, anchor_non_bg_exp)
            negative_mask = diff_class

            positive_mask_float = tf.cast(positive_mask, tf.float32)
            negative_mask_float = tf.cast(negative_mask, tf.float32)

            pos_count = tf.reduce_sum(positive_mask_float, axis=1)
            neg_count = tf.reduce_sum(negative_mask_float, axis=1)

            if self.config.remove_easy_triplets:
                # Need to compute each triplet individually
                def compute_anchor_loss(anchor_idx):
                    anchor_distances = distances[anchor_idx]
                    anchor_pos_mask = positive_mask[anchor_idx]
                    anchor_neg_mask = negative_mask[anchor_idx]

                    pos_indices = tf.where(anchor_pos_mask)[:, 0]
                    neg_indices = tf.where(anchor_neg_mask)[:, 0]

                    pos_dists = tf.gather(anchor_distances, pos_indices)
                    neg_dists = tf.gather(anchor_distances, neg_indices)

                    num_pos = tf.shape(pos_dists)[0]
                    num_neg = tf.shape(neg_dists)[0]

                    pos_expanded = tf.expand_dims(pos_dists, 1)
                    neg_expanded = tf.expand_dims(neg_dists, 0)

                    triplet_losses = pos_expanded - neg_expanded + self.config.margin
                    triplet_losses = tf.maximum(triplet_losses, 0.0)

                    anchor_loss = tf.reduce_sum(triplet_losses)
                    anchor_total_triplets = tf.cast(num_pos * num_neg, tf.float32)
                    anchor_positive_triplets = tf.reduce_sum(tf.cast(triplet_losses > 1e-16, tf.float32))

                    return anchor_loss, anchor_total_triplets, anchor_positive_triplets

                anchor_results = tf.map_fn(
                    compute_anchor_loss,
                    tf.range(actual_batch_size),
                    fn_output_signature=(
                        tf.TensorSpec([], dtype=tf.float32),
                        tf.TensorSpec([], dtype=tf.float32),
                        tf.TensorSpec([], dtype=tf.float32),
                    )
                )

                anchor_losses = anchor_results[0]
                anchor_total_triplets = anchor_results[1]
                anchor_positive_triplets = anchor_results[2]
            else:
                # Efficient computation without remove_easy_triplets
                pos_dist_sum = tf.reduce_sum(distances * positive_mask_float, axis=1)
                neg_dist_sum = tf.reduce_sum(distances * negative_mask_float, axis=1)
                triplets_per_anchor = pos_count * neg_count

                anchor_losses = (
                    pos_dist_sum * neg_count -
                    neg_dist_sum * pos_count +
                    self.config.margin * triplets_per_anchor
                )
                anchor_total_triplets = triplets_per_anchor
                anchor_positive_triplets = triplets_per_anchor  # All triplets counted

            # Accumulate per class
            anchor_labels_clipped = tf.clip_by_value(anchor_labels, 0, max_classes - 1)
            indices = tf.expand_dims(anchor_labels_clipped, 1)

            batch_loss_per_class = tf.scatter_nd(indices, anchor_losses, [max_classes])
            batch_count_per_class = tf.scatter_nd(indices, anchor_total_triplets, [max_classes])
            batch_positive_per_class = tf.scatter_nd(indices, anchor_positive_triplets, [max_classes])

            return (
                loss_per_class + batch_loss_per_class,
                count_per_class + batch_count_per_class,
                positive_count_per_class + batch_positive_per_class
            )

        def cond(batch_idx, accumulators):
            return batch_idx < num_batches

        def body(batch_idx, accumulators):
            new_accumulators = compute_batch_loss(batch_idx, accumulators)
            return batch_idx + 1, new_accumulators

        initial_accumulators = (loss_per_class, count_per_class, positive_count_per_class)
        _, final_accumulators = tf.while_loop(
            cond,
            body,
            loop_vars=(tf.constant(0), initial_accumulators),
            parallel_iterations=1
        )

        final_loss_per_class, final_count_per_class, final_positive_per_class = final_accumulators

        # Compute mean loss per class
        if self.config.remove_easy_triplets:
            # Divide by positive triplet count per class
            mean_loss_per_class = final_loss_per_class / (final_positive_per_class + 1e-16)
            valid_class_mask = final_positive_per_class > 0
        else:
            # Divide by total triplet count per class
            mean_loss_per_class = final_loss_per_class / (final_count_per_class + 1e-16)
            valid_class_mask = final_count_per_class > 0

        # Weight each class equally
        num_valid_classes = tf.reduce_sum(tf.cast(valid_class_mask, tf.float32))
        total_loss = tf.reduce_sum(mean_loss_per_class * tf.cast(valid_class_mask, tf.float32))
        final_loss = total_loss / (num_valid_classes + 1e-16)

        return final_loss

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute pixel-based triplet loss.
        
        Args:
            y_true: Ground truth labels (batch_size, h2, w2, num_whiskers)
            y_pred: Encoder predictions (batch_size, h, w, feature_dim)
            
        Returns:
            Scalar loss value
        """
        # Validate inputs
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        batch_size = tf.shape(y_pred)[0]
        pred_h, pred_w = tf.shape(y_pred)[1], tf.shape(y_pred)[2]
        feature_dim = tf.shape(y_pred)[3]
        
        label_h, label_w = tf.shape(y_true)[1], tf.shape(y_true)[2]
        
        # Instead of downsampling labels (which causes aliasing of thin structures),
        # upsample embeddings to match label resolution for better accuracy
        y_pred_resized = self._resize_embeddings(y_pred, label_h, label_w)
        
        # Convert labels to class indices (background=0, whisker_i=i+1)
        class_labels = self._labels_to_classes(y_true)
        
        # Get pixels: either all pixels (class_balanced_weighting) or sampled pixels
        if self.config.class_balanced_weighting:
            # Use ALL pixels - class balancing is done via weighting in the loss
            sampled_embeddings, sampled_labels = self._get_all_pixels(
                y_pred_resized, class_labels
            )
        elif self.config.use_balanced_sampling:
            if self.config.strict_per_class_balancing:
                # Try strict per-class balancing
                if self.config.prefer_graph_mode_strict:
                    # Try graph-mode strict balancing first
                    try:
                        sampled_embeddings, sampled_labels = self._sample_pixels_strict_balanced_graph(
                            y_pred_resized, class_labels
                        )
                    except Exception:
                        # Fall back to eager-mode strict balancing
                        try:
                            sampled_embeddings, sampled_labels = self._sample_pixels_strict_balanced_eager(
                                y_pred_resized, class_labels
                            )
                        except Exception:
                            # Fall back to regular balanced sampling
                            sampled_embeddings, sampled_labels = self._sample_pixels_balanced(
                                y_pred_resized, class_labels
                            )
                else:
                    # Try eager-mode strict balancing first
                    try:
                        sampled_embeddings, sampled_labels = self._sample_pixels_strict_balanced_eager(
                            y_pred_resized, class_labels
                        )
                    except Exception:
                        # Fall back to graph-mode strict balancing
                        try:
                            sampled_embeddings, sampled_labels = self._sample_pixels_strict_balanced_graph(
                                y_pred_resized, class_labels
                            )
                        except Exception:
                            # Fall back to regular balanced sampling
                            sampled_embeddings, sampled_labels = self._sample_pixels_balanced(
                                y_pred_resized, class_labels
                            )
            else:
                # Regular balanced sampling (background vs whiskers)
                sampled_embeddings, sampled_labels = self._sample_pixels_balanced(
                    y_pred_resized, class_labels
                )
        else:
            # Legacy sampling approach
            sampled_embeddings, sampled_labels = self._sample_pixels_per_class_simple(
                y_pred_resized, class_labels
            )
        
        # Check for memory warning (only if not using exact mode)
        if not self.config.use_exact:
            self._check_memory_usage_simple(sampled_embeddings)

        # Use custom triplet loss functions that support different distance metrics
        if self.config.use_exact:
            # Use loop-based exact computation (memory efficient for large images)
            if self.config.class_balanced_weighting:
                # Class-balanced weighting: accumulate per-class, then weight equally
                if self.config.triplet_strategy == "hard":
                    loss = self._batch_hard_triplet_loss_exact_class_balanced(sampled_labels, sampled_embeddings)
                elif self.config.triplet_strategy == "all":
                    loss = self._batch_all_triplet_loss_exact_class_balanced(sampled_labels, sampled_embeddings)
                else:  # semi_hard - use hard as fallback
                    loss = self._batch_hard_triplet_loss_exact_class_balanced(sampled_labels, sampled_embeddings)
            else:
                # Standard exact computation (no class weighting)
                if self.config.triplet_strategy == "hard":
                    loss = self._batch_hard_triplet_loss_exact(sampled_labels, sampled_embeddings)
                elif self.config.triplet_strategy == "all":
                    loss = self._batch_all_triplet_loss_exact(sampled_labels, sampled_embeddings)
                else:  # semi_hard - use hard as fallback
                    loss = self._batch_hard_triplet_loss_exact(sampled_labels, sampled_embeddings)
        else:
            # Use standard pairwise matrix computation (faster but memory intensive)
            if self.config.triplet_strategy == "hard":
                loss = self._batch_hard_triplet_loss_custom(sampled_labels, sampled_embeddings)
            elif self.config.triplet_strategy == "all":
                loss = self._batch_all_triplet_loss_custom(sampled_labels, sampled_embeddings)
            else:  # semi_hard - use hard as fallback since semi_hard is complex
                loss = self._batch_hard_triplet_loss_custom(sampled_labels, sampled_embeddings)

        return loss
    
    def _resize_embeddings(self, embeddings: tf.Tensor, target_h: int, target_w: int) -> tf.Tensor:
        """
        Resize embeddings to match label dimensions using nearest neighbor interpolation.
        
        This approach upsamples embeddings instead of downsampling labels to avoid
        aliasing artifacts that can cause thin structures (like whiskers) to disappear.
        
        Args:
            embeddings: tensor of shape (batch_size, h, w, feature_dim)
            target_h: target height (from labels)
            target_w: target width (from labels)
            
        Returns:
            resized_embeddings: tensor of shape (batch_size, target_h, target_w, feature_dim)
        """
        # embeddings: (batch_size, h, w, feature_dim)
        
        shape = tf.shape(embeddings)
        batch_size = shape[0]
        original_h = shape[1]
        original_w = shape[2]
        feature_dim = shape[3]
        
        # Check if resize is needed
        def no_resize_needed():
            return embeddings
        
        def resize_needed():
            # Use tf.image.resize with nearest neighbor to preserve embedding values
            # This creates redundancy but preserves all spatial relationships
            resized = tf.image.resize(
                embeddings,
                size=[target_h, target_w],
                method='nearest'  # Nearest neighbor to avoid interpolation artifacts
            )
            
            return tf.cast(resized, tf.float32)
        
        return tf.cond(
            tf.logical_and(tf.equal(original_h, target_h), tf.equal(original_w, target_w)),
            no_resize_needed,
            resize_needed
        )
    
    def _resize_labels(self, labels: tf.Tensor, target_h: int, target_w: int) -> tf.Tensor:
        """Resize labels to match prediction dimensions."""
        # labels: (batch_size, h2, w2, num_whiskers)
        
        shape = tf.shape(labels)
        batch_size = shape[0]
        original_h = shape[1]
        original_w = shape[2]
        num_whiskers = shape[3]
        
        # Check if resize is needed
        def no_resize_needed():
            return labels
        
        def resize_needed():
            # Reshape to (batch_size * num_whiskers, h, w, 1) for tf.image.resize
            labels_reshaped = tf.transpose(labels, [0, 3, 1, 2])  # (batch, whiskers, h, w)
            labels_reshaped = tf.reshape(labels_reshaped, [batch_size * num_whiskers, original_h, original_w, 1])
            
            # Resize all channels at once
            resized = tf.image.resize(
                labels_reshaped,
                size=[target_h, target_w],
                method='nearest'
            )
            
            # Reshape back to (batch_size, target_h, target_w, num_whiskers)
            resized = tf.reshape(resized, [batch_size, num_whiskers, target_h, target_w])
            resized = tf.transpose(resized, [0, 2, 3, 1])  # (batch, h, w, whiskers)
            
            return tf.cast(resized, tf.float32)
        
        return tf.cond(
            tf.logical_and(tf.equal(original_h, target_h), tf.equal(original_w, target_w)),
            no_resize_needed,
            resize_needed
        )
    
    def _labels_to_classes(self, labels: tf.Tensor) -> tf.Tensor:
        """Convert multi-channel binary masks to class indices."""
        # labels: (batch_size, h, w, num_whiskers)
        # Background pixels are where all channels are 0
        # Whisker pixels are where exactly one channel is 1
        
        # Sum across whisker channels to find background (sum=0) and whisker pixels (sum>0)
        whisker_sum = tf.reduce_sum(labels, axis=-1)  # (batch_size, h, w)
        
        # Use argmax to find which whisker channel is active
        # argmax returns 0-based indices, so we add 1 to get whisker IDs
        whisker_classes = tf.argmax(labels, axis=-1, output_type=tf.int32) + 1  # (batch_size, h, w)
        
        # Set background pixels (where sum=0) to class 0
        class_labels = tf.where(
            whisker_sum > 0.5,
            whisker_classes,
            tf.zeros_like(whisker_classes, dtype=tf.int32)
        )
        
        return class_labels
    
    def _sample_pixels_per_class_simple(
        self, 
        embeddings: tf.Tensor, 
        class_labels: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Simplified pixel sampling per class - graph mode compatible."""
        # embeddings: (batch_size, h, w, feature_dim)
        # class_labels: (batch_size, h, w)
        
        # Flatten embeddings and labels
        shape = tf.shape(embeddings)
        batch_size = shape[0]
        h = shape[1]
        w = shape[2]
        feature_dim = shape[3]
        
        embeddings_flat = tf.reshape(embeddings, [-1, feature_dim])  # (N, feature_dim)
        class_labels_flat = tf.reshape(class_labels, [-1])  # (N,)
        
        # Sample from each class separately
        sampled_embeddings_list = []
        sampled_labels_list = []
        
        # Handle background class (class 0)
        background_mask = tf.equal(class_labels_flat, 0)
        background_indices = tf.where(background_mask)[:, 0]
        
        # Sample background pixels
        num_background = tf.minimum(self.config.background_pixels, tf.shape(background_indices)[0])
        sampled_background_indices = tf.random.shuffle(background_indices)[:num_background]
        background_embeddings = tf.gather(embeddings_flat, sampled_background_indices)
        background_labels = tf.zeros([num_background], dtype=tf.int32)
        
        sampled_embeddings_list.append(background_embeddings)
        sampled_labels_list.append(background_labels)
        
        # Handle whisker classes (classes 1, 2, 3, etc.)
        # We'll sample from all non-background classes
        non_background_mask = tf.greater(class_labels_flat, 0)
        non_background_indices = tf.where(non_background_mask)[:, 0]
        
        # Sample whisker pixels
        num_whisker = tf.minimum(self.config.whisker_pixels, tf.shape(non_background_indices)[0])
        sampled_whisker_indices = tf.random.shuffle(non_background_indices)[:num_whisker]
        whisker_embeddings = tf.gather(embeddings_flat, sampled_whisker_indices)
        whisker_labels = tf.gather(class_labels_flat, sampled_whisker_indices)
        
        sampled_embeddings_list.append(whisker_embeddings)
        sampled_labels_list.append(whisker_labels)
        
        # Concatenate all samples
        final_embeddings = tf.concat(sampled_embeddings_list, axis=0)
        final_labels = tf.concat(sampled_labels_list, axis=0)
        
        # Handle edge case where no samples are found
        final_embeddings = tf.cond(
            tf.equal(tf.shape(final_embeddings)[0], 0),
            lambda: tf.zeros((2, feature_dim), dtype=tf.float32),
            lambda: final_embeddings
        )
        
        final_labels = tf.cond(
            tf.equal(tf.shape(final_labels)[0], 0),
            lambda: tf.constant([0, 1], dtype=tf.int32),
            lambda: final_labels
        )
        
        return final_embeddings, final_labels

    def _get_all_pixels(
        self,
        embeddings: tf.Tensor,
        class_labels: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Get ALL pixels without any sampling.

        Use this with use_exact=True to process every pixel in the image.
        This is useful when you want exact statistics over the entire image
        and can afford the computation time.

        Args:
            embeddings: (batch_size, h, w, feature_dim)
            class_labels: (batch_size, h, w)

        Returns:
            all_embeddings: (N, feature_dim) where N = batch_size * h * w
            all_labels: (N,)
        """
        shape = tf.shape(embeddings)
        feature_dim = shape[3]

        # Simply flatten without any sampling
        embeddings_flat = tf.reshape(embeddings, [-1, feature_dim])  # (N, feature_dim)
        labels_flat = tf.reshape(class_labels, [-1])  # (N,)

        return embeddings_flat, labels_flat

    def _sample_pixels_balanced(
        self, 
        embeddings: tf.Tensor, 
        class_labels: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Balanced pixel sampling that ensures equal samples per class - graph mode compatible.
        
        This method:
        1. Finds the minimum number of available pixels across all classes
        2. Caps this at max_samples_per_class if specified
        3. Samples exactly this many pixels from each class
        
        This prevents class imbalance issues that can hurt triplet loss training.
        """
        # embeddings: (batch_size, h, w, feature_dim)
        # class_labels: (batch_size, h, w)
        
        # Flatten embeddings and labels
        shape = tf.shape(embeddings)
        batch_size = shape[0]
        h = shape[1]
        w = shape[2]
        feature_dim = shape[3]
        
        embeddings_flat = tf.reshape(embeddings, [-1, feature_dim])  # (N, feature_dim)
        class_labels_flat = tf.reshape(class_labels, [-1])  # (N,)
        
        # Count pixels for background and non-background classes
        background_mask = tf.equal(class_labels_flat, 0)
        background_count = tf.reduce_sum(tf.cast(background_mask, tf.int32))
        
        non_background_mask = tf.greater(class_labels_flat, 0)
        non_background_count = tf.reduce_sum(tf.cast(non_background_mask, tf.int32))
        
        # Find minimum available samples across classes
        min_available = tf.minimum(background_count, non_background_count)
        
        # Apply the maximum cap if specified
        if self.config.max_samples_per_class is not None:
            samples_per_class = tf.minimum(min_available, self.config.max_samples_per_class)
        else:
            samples_per_class = min_available
        
        # Ensure we have at least 1 sample per class
        samples_per_class = tf.maximum(samples_per_class, 1)
        
        # Sample exactly samples_per_class from each class
        sampled_embeddings_list = []
        sampled_labels_list = []
        
        # Sample background pixels
        background_indices = tf.where(background_mask)[:, 0]
        sampled_background_indices = tf.random.shuffle(background_indices)[:samples_per_class]
        background_embeddings = tf.gather(embeddings_flat, sampled_background_indices)
        background_labels = tf.zeros([samples_per_class], dtype=tf.int32)
        
        sampled_embeddings_list.append(background_embeddings)
        sampled_labels_list.append(background_labels)
        
        # Sample non-background (whisker) pixels
        non_background_indices = tf.where(non_background_mask)[:, 0]
        sampled_whisker_indices = tf.random.shuffle(non_background_indices)[:samples_per_class]
        whisker_embeddings = tf.gather(embeddings_flat, sampled_whisker_indices)
        whisker_labels = tf.gather(class_labels_flat, sampled_whisker_indices)
        
        sampled_embeddings_list.append(whisker_embeddings)
        sampled_labels_list.append(whisker_labels)
        
        # Concatenate all samples
        final_embeddings = tf.concat(sampled_embeddings_list, axis=0)
        final_labels = tf.concat(sampled_labels_list, axis=0)
        
        # Handle edge case where no samples are found
        final_embeddings = tf.cond(
            tf.equal(tf.shape(final_embeddings)[0], 0),
            lambda: tf.zeros((2, feature_dim), dtype=tf.float32),
            lambda: final_embeddings
        )
        
        final_labels = tf.cond(
            tf.equal(tf.shape(final_labels)[0], 0),
            lambda: tf.constant([0, 1], dtype=tf.int32),
            lambda: final_labels
        )
        
        return final_embeddings, final_labels
    
    def _sample_pixels_strict_balanced_eager(
        self, 
        embeddings: tf.Tensor, 
        class_labels: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Strict per-class balanced sampling - EAGER MODE ONLY.
        
        This method achieves true per-class balancing where each individual class
        (background, whisker1, whisker2, etc.) gets exactly the same number of samples.
        This is ideal for discriminating between different whisker instances.
        
        Note: This method uses Python for loops and only works in eager execution mode.
        """
        # embeddings: (batch_size, h, w, feature_dim)
        # class_labels: (batch_size, h, w)
        
        # Flatten embeddings and labels
        shape = tf.shape(embeddings)
        batch_size = shape[0]
        h = shape[1]
        w = shape[2]
        feature_dim = shape[3]
        
        embeddings_flat = tf.reshape(embeddings, [-1, feature_dim])  # (N, feature_dim)
        class_labels_flat = tf.reshape(class_labels, [-1])  # (N,)
        
        # Get unique classes and count pixels per class
        unique_classes, _, class_counts = tf.unique_with_counts(class_labels_flat)
        
        # Find minimum available samples across ALL classes
        min_available = tf.reduce_min(class_counts)
        
        # Apply the maximum cap if specified
        if self.config.max_samples_per_class is not None:
            samples_per_class = tf.minimum(min_available, self.config.max_samples_per_class)
        else:
            samples_per_class = min_available
        
        # Ensure we have at least 1 sample per class
        samples_per_class = tf.maximum(samples_per_class, 1)
        
        # Sample exactly samples_per_class from each individual class
        sampled_embeddings_list = []
        sampled_labels_list = []
        
        # Use Python for loop (only works in eager mode)
        for i in range(len(unique_classes)):
            class_id = unique_classes[i]
            
            # Get indices for this specific class
            class_mask = tf.equal(class_labels_flat, class_id)
            class_indices = tf.where(class_mask)[:, 0]
            
            # Randomly sample exactly samples_per_class indices
            sampled_indices = tf.random.shuffle(class_indices)[:samples_per_class]
            
            # Gather embeddings and labels
            class_embeddings = tf.gather(embeddings_flat, sampled_indices)
            class_labels_sampled = tf.fill([samples_per_class], class_id)
            
            sampled_embeddings_list.append(class_embeddings)
            sampled_labels_list.append(class_labels_sampled)
        
        # Concatenate all samples
        if len(sampled_embeddings_list) > 0:
            final_embeddings = tf.concat(sampled_embeddings_list, axis=0)
            final_labels = tf.concat(sampled_labels_list, axis=0)
        else:
            # No samples found, create dummy
            final_embeddings = tf.zeros((2, feature_dim), dtype=tf.float32)
            final_labels = tf.constant([0, 1], dtype=tf.int32)
        
        return final_embeddings, final_labels
    
    def _sample_pixels_strict_balanced_graph(
        self, 
        embeddings: tf.Tensor, 
        class_labels: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Strict per-class balanced sampling - GRAPH MODE COMPATIBLE.
        
        This method achieves true per-class balancing where each individual class
        (background, whisker1, whisker2, etc.) gets exactly the same number of samples.
        Uses tf.map_fn instead of Python for loops to work in graph mode.
        
        This is ideal for discriminating between different whisker instances.
        """
        # embeddings: (batch_size, h, w, feature_dim)
        # class_labels: (batch_size, h, w)
        
        # Flatten embeddings and labels
        shape = tf.shape(embeddings)
        batch_size = shape[0]
        h = shape[1]
        w = shape[2]
        feature_dim = shape[3]
        
        embeddings_flat = tf.reshape(embeddings, [-1, feature_dim])  # (N, feature_dim)
        class_labels_flat = tf.reshape(class_labels, [-1])  # (N,)
        
        # Get unique classes and count pixels per class
        unique_classes, _, class_counts = tf.unique_with_counts(class_labels_flat)
        
        # Find minimum available samples across ALL classes
        min_available = tf.reduce_min(class_counts)
        
        # Apply the maximum cap if specified
        if self.config.max_samples_per_class is not None:
            samples_per_class = tf.minimum(min_available, self.config.max_samples_per_class)
        else:
            samples_per_class = min_available
        
        # Ensure we have at least 1 sample per class
        samples_per_class = tf.maximum(samples_per_class, 1)
        
        # Define a function to sample from a single class
        def sample_single_class(class_id):
            """Sample pixels from a single class."""
            # Get indices for this specific class
            class_mask = tf.equal(class_labels_flat, class_id)
            class_indices = tf.where(class_mask)[:, 0]
            
            # Randomly sample exactly samples_per_class indices
            sampled_indices = tf.random.shuffle(class_indices)[:samples_per_class]
            
            # Gather embeddings and labels
            class_embeddings = tf.gather(embeddings_flat, sampled_indices)
            class_labels_sampled = tf.fill([samples_per_class], class_id)
            
            return class_embeddings, class_labels_sampled
        
        # Use tf.map_fn to apply sampling to each unique class (graph mode compatible)
        sampled_results = tf.map_fn(
            sample_single_class,
            unique_classes,
            fn_output_signature=(
                tf.TensorSpec([None, feature_dim], dtype=tf.float32),  # embeddings
                tf.TensorSpec([None], dtype=tf.int32)  # labels
            ),
            parallel_iterations=10
        )
        
        # Extract embeddings and labels from results
        sampled_embeddings_per_class, sampled_labels_per_class = sampled_results
        
        # Reshape and concatenate all samples
        # sampled_embeddings_per_class: (num_classes, samples_per_class, feature_dim)
        # sampled_labels_per_class: (num_classes, samples_per_class)
        
        num_classes = tf.shape(unique_classes)[0]
        total_samples = num_classes * samples_per_class
        
        final_embeddings = tf.reshape(
            sampled_embeddings_per_class, 
            [total_samples, feature_dim]
        )
        final_labels = tf.reshape(
            sampled_labels_per_class, 
            [total_samples]
        )
        
        # Handle edge case where no samples are found
        final_embeddings = tf.cond(
            tf.equal(tf.shape(final_embeddings)[0], 0),
            lambda: tf.zeros((2, feature_dim), dtype=tf.float32),
            lambda: final_embeddings
        )
        
        final_labels = tf.cond(
            tf.equal(tf.shape(final_labels)[0], 0),
            lambda: tf.constant([0, 1], dtype=tf.int32),
            lambda: final_labels
        )
        
        return final_embeddings, final_labels
    
    def _check_memory_usage_simple(self, embeddings: tf.Tensor) -> None:
        """Simple memory usage check without tf.py_function."""
        num_pixels = tf.shape(embeddings)[0]
        matrix_size = num_pixels * num_pixels
        
        # Simple warning based on threshold
        if self.config.memory_warning_threshold > 0:
            # We can't use tf.py_function here, so we'll skip the warning
            # This is a limitation but avoids graph compilation issues
            pass


# Convenience functions for easy usage
def create_pixel_triplet_loss(
    margin: float = 1.0,
    background_pixels: int = 1000,
    whisker_pixels: int = 500,
    max_samples_per_class: Optional[int] = None,
    use_balanced_sampling: bool = True,
    strict_per_class_balancing: bool = False,
    prefer_graph_mode_strict: bool = True,
    distance_metric: str = "euclidean",
    triplet_strategy: str = "semi_hard",
    reduction: str = "mean",
    remove_easy_triplets: bool = False,
    use_exact: bool = False,
    batch_size_for_exact: int = 100,
    class_balanced_weighting: bool = False,
    **kwargs
) -> PixelTripletLoss:
    """
    Create a pixel triplet loss with specified parameters.
    
    Args:
        margin: Triplet loss margin
        background_pixels: Number of background pixels to sample (legacy, for backward compatibility)
        whisker_pixels: Number of whisker pixels to sample per whisker (legacy, for backward compatibility)
        max_samples_per_class: Maximum samples per class for balanced sampling (recommended)
        use_balanced_sampling: Whether to use balanced sampling (recommended: True)
        strict_per_class_balancing: True per-class balancing for each whisker class
        prefer_graph_mode_strict: Use graph-mode strict balancing when possible (recommended: True)
        distance_metric: Distance metric ('euclidean', 'cosine', 'manhattan')
        triplet_strategy: Triplet mining strategy ('semi_hard', 'hard', 'all')
        reduction: Loss reduction ('mean', 'sum', 'none')
        remove_easy_triplets: Whether to exclude easy triplets in batch_all mode
        use_exact: If True, use loop-based exact computation instead of full pairwise matrix.
            This is memory efficient for large images but slower. Default is False.
        batch_size_for_exact: Number of anchor pixels to process per iteration when use_exact=True.
            Larger values are faster but use more memory. Default is 100.
        class_balanced_weighting: If True, uses ALL pixels without sampling, accumulates
            loss per anchor class, then weights each class equally. This ensures each class
            contributes equally to the loss despite pixel count imbalance. Only works with
            use_exact=True. Default is False.

    Returns:
        PixelTripletLoss instance
        
    Note:
        For whisker discrimination, use strict_per_class_balancing=True. 
        Graph-mode strict balancing works in both eager and graph execution modes,
        providing better performance than eager-only strict balancing.
        
        Important: When using strict_per_class_balancing=True in graph mode,
        compile your model with jit_compile=False to avoid XLA compilation issues:
        model.compile(optimizer='adam', loss=pixel_loss, jit_compile=False)

        For large images where memory is an issue, use use_exact=True:
        loss = create_pixel_triplet_loss(
            use_exact=True,
            batch_size_for_exact=200  # Adjust based on available memory
        )

        For class-balanced loss that uses ALL pixels with equal class contribution:
        loss = create_pixel_triplet_loss(
            class_balanced_weighting=True,  # Automatically enables use_exact
            batch_size_for_exact=200
        )
    """
    config = PixelTripletConfig(
        margin=margin,
        background_pixels=background_pixels,
        whisker_pixels=whisker_pixels,
        max_samples_per_class=max_samples_per_class,
        use_balanced_sampling=use_balanced_sampling,
        strict_per_class_balancing=strict_per_class_balancing,
        prefer_graph_mode_strict=prefer_graph_mode_strict,
        distance_metric=distance_metric,
        triplet_strategy=triplet_strategy,
        reduction=reduction,
        remove_easy_triplets=remove_easy_triplets,
        use_exact=use_exact,
        batch_size_for_exact=batch_size_for_exact,
        class_balanced_weighting=class_balanced_weighting,
        **kwargs
    )
    return PixelTripletLoss(config=config)
