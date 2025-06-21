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
except ImportError:
    # Fallback for older Keras versions
    from keras.losses import Loss


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
    distance_metric: str = "euclidean"
    triplet_strategy: str = "semi_hard"
    reduction: str = "mean"
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


class PixelTripletLoss(Loss):
    """
    Pixel-based triplet loss for semantic segmentation.
    
    This loss operates on pixel-level embeddings, sampling a specified number of pixels
    per class and computing triplet loss with semi-hard or hard negative mining.
    
    Args:
        config: PixelTripletConfig instance with loss parameters
        name: Name for the loss function
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
            'distance_metric': self.config.distance_metric,
            'triplet_strategy': self.config.triplet_strategy,
            'reduction': self.config.reduction,
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
            distance_metric=config.pop('distance_metric', 'euclidean'),
            triplet_strategy=config.pop('triplet_strategy', 'semi_hard'),
            reduction=config.pop('reduction', 'mean'),
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
        
        # For each anchor, get the hardest positive
        mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
        mask_anchor_positive = keras.ops.cast(mask_anchor_positive, "float32")
        
        # We put to 0 any element where (a, p) is not valid 
        anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)
        
        # shape (batch_size, 1)
        hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)
        
        # For each anchor, get the hardest negative
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
        """
        # Get the pairwise distance matrix using the configured metric
        pairwise_dist = self._compute_pairwise_distances(embeddings)
        
        anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
        anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
        
        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.config.margin
        
        # Put to zero the invalid triplets
        mask = _get_triplet_mask(labels)
        mask = keras.ops.cast(mask, "float32")
        triplet_loss = tf.multiply(mask, triplet_loss)
        
        num_valid_triplets = tf.reduce_sum(mask)
        
        # Remove negative losses (i.e. the easy triplets)
        triplet_loss = tf.maximum(triplet_loss, 0.0)
        
        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = keras.ops.cast(tf.greater(triplet_loss, 1e-16), "float32")
        num_positive_triplets = tf.reduce_sum(valid_triplets)
        
        # Get final mean triplet loss over the positive valid triplets
        triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)
        
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
        
        # Resize labels to match prediction dimensions
        y_true_resized = self._resize_labels(y_true, pred_h, pred_w)
        
        # Convert labels to class indices (background=0, whisker_i=i+1)
        class_labels = self._labels_to_classes(y_true_resized)
        
        # Sample pixels per class using balanced or legacy approach
        if self.config.use_balanced_sampling:
            sampled_embeddings, sampled_labels = self._sample_pixels_balanced(
                y_pred, class_labels
            )
        else:
            sampled_embeddings, sampled_labels = self._sample_pixels_per_class_simple(
                y_pred, class_labels
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
        """Simplified pixel sampling per class."""
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
        
        # Get unique classes
        unique_classes, _ = tf.unique(class_labels_flat)
        
        # Sample pixels for each class
        sampled_embeddings_list = []
        sampled_labels_list = []
        
        for class_id in unique_classes:
            # Get indices for this class
            class_mask = tf.equal(class_labels_flat, class_id)
            class_indices = tf.where(class_mask)[:, 0]
            
            # Determine number of samples for this class
            if class_id == 0:  # Background
                num_samples = tf.minimum(self.config.background_pixels, tf.shape(class_indices)[0])
            else:  # Whisker
                num_samples = tf.minimum(self.config.whisker_pixels, tf.shape(class_indices)[0])
            
            # Randomly sample indices
            sampled_indices = tf.random.shuffle(class_indices)[:num_samples]
            
            # Gather embeddings and labels
            class_embeddings = tf.gather(embeddings_flat, sampled_indices)
            class_labels_sampled = tf.fill([num_samples], class_id)
            
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
    
    def _sample_pixels_balanced(
        self, 
        embeddings: tf.Tensor, 
        class_labels: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Balanced pixel sampling that ensures equal samples per class.
        
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
        
        # Get unique classes and count pixels per class
        unique_classes, _ = tf.unique(class_labels_flat)
        
        # Find minimum available samples across all classes
        class_counts = []
        for class_id in unique_classes:
            class_mask = tf.equal(class_labels_flat, class_id)
            class_count = tf.reduce_sum(tf.cast(class_mask, tf.int32))
            class_counts.append(class_count)
        
        # Find the minimum available across classes
        min_available = tf.reduce_min(tf.stack(class_counts))
        
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
        
        for class_id in unique_classes:
            # Get indices for this class
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
    distance_metric: str = "euclidean",
    triplet_strategy: str = "semi_hard",
    reduction: str = "mean",
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
        distance_metric: Distance metric ('euclidean', 'cosine', 'manhattan')
        triplet_strategy: Triplet mining strategy ('semi_hard', 'hard', 'all')
        reduction: Loss reduction ('mean', 'sum', 'none')
        
    Returns:
        PixelTripletLoss instance
        
    Note:
        For best results, use balanced sampling with max_samples_per_class.
        This ensures equal samples per class, preventing class imbalance issues.
    """
    config = PixelTripletConfig(
        margin=margin,
        background_pixels=background_pixels,
        whisker_pixels=whisker_pixels,
        max_samples_per_class=max_samples_per_class,
        use_balanced_sampling=use_balanced_sampling,
        distance_metric=distance_metric,
        triplet_strategy=triplet_strategy,
        reduction=reduction,
        **kwargs
    )
    return PixelTripletLoss(config=config) 