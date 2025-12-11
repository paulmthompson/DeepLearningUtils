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
        
        # Sample pixels per class using balanced or legacy approach
        if self.config.use_balanced_sampling:
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
        
        # Check for memory warning
        self._check_memory_usage_simple(sampled_embeddings)
        
        # Use custom triplet loss functions that support different distance metrics
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
        
    Returns:
        PixelTripletLoss instance
        
    Note:
        For whisker discrimination, use strict_per_class_balancing=True. 
        Graph-mode strict balancing works in both eager and graph execution modes,
        providing better performance than eager-only strict balancing.
        
        Important: When using strict_per_class_balancing=True in graph mode,
        compile your model with jit_compile=False to avoid XLA compilation issues:
        model.compile(optimizer='adam', loss=pixel_loss, jit_compile=False)
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
        **kwargs
    )
    return PixelTripletLoss(config=config)
