#!/usr/bin/env python3
"""
Embedding Separation Metric for Pixel-based Triplet Loss.

This module provides a Keras metric to monitor embedding quality during training,
specifically designed for pixel-level embedding spaces used with triplet loss.

The separation ratio (inter-class / intra-class distance) provides a more meaningful
measure of embedding quality than triplet loss when easy triplets are excluded,
as the same loss value can indicate different levels of actual separation.

Key Features:
- Computes inter-class and intra-class distances via sampling (tractable for large images)
- Handles upsampling effects by optionally using strided sampling
- Accumulates statistics across batches for more robust estimates
- Works with multi-class segmentation masks (background + multiple foreground classes)
"""

import tensorflow as tf
import keras
import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class EmbeddingSeparationConfig:
    """Configuration for embedding separation metric."""
    # Sampling parameters
    max_samples_per_class: int = 500  # Max pixels to sample per class (only used if use_exact=False)
    max_pairs_per_class: int = 10000  # Max pairs for distance computation (only used if use_exact=False)

    # Distance metric
    distance_metric: str = "euclidean"  # 'euclidean', 'cosine', 'manhattan'

    # Upsampling handling
    stride: int = 1  # Stride for sampling (helps avoid redundant upsampled pixels)

    # Background handling
    exclude_background_intra: bool = True  # Exclude background-to-background pairs from intra-class distance
    # When True: intra-class distance only measures how tight foreground clusters are
    # Background is still used for inter-class distances (foreground vs background)

    # Exact computation (loop-based)
    use_exact: bool = False  # If True, compute ALL pairwise distances via looping (no sampling)
    batch_size_for_exact: int = 100  # Number of anchor pixels to process per iteration in exact mode

    # Numerical stability
    epsilon: float = 1e-8

    def __post_init__(self):
        if self.max_samples_per_class <= 0:
            raise ValueError(f"max_samples_per_class must be > 0, got {self.max_samples_per_class}")
        if self.stride <= 0:
            raise ValueError(f"stride must be > 0, got {self.stride}")
        if self.batch_size_for_exact <= 0:
            raise ValueError(f"batch_size_for_exact must be > 0, got {self.batch_size_for_exact}")
        if self.distance_metric not in ["euclidean", "cosine", "manhattan"]:
            raise ValueError(f"distance_metric must be one of ['euclidean', 'cosine', 'manhattan']")


class EmbeddingSeparationRatio(keras.metrics.Metric):
    """
    Metric that computes the separation ratio of pixel embeddings.

    Separation Ratio = mean(inter-class distances) / mean(intra-class distances)

    A higher ratio indicates better embedding quality:
    - Ratio > 1: Classes are well separated on average
    - Ratio >> 1: Excellent separation (embeddings are highly discriminative)
    - Ratio < 1: Classes overlap (embeddings need improvement)
    - Ratio ~ 1: Marginal separation

    This metric is particularly useful when:
    1. Triplet loss filters out easy triplets (making loss values hard to interpret)
    2. You want to track whether embeddings are actually improving
    3. You need to compare models with different triplet mining strategies

    Background Handling:
        By default (exclude_background_intra=True), intra-class distance only
        considers foreground classes. This is appropriate for segmentation tasks
        where you care about tight foreground clusters but background is typically
        heterogeneous. Background pixels are still included in inter-class distances,
        so the metric measures how well foreground classes separate from background.

        Set exclude_background_intra=False to include background-to-background
        distances in the intra-class computation.

    Note on Upsampling:
        When embeddings are upsampled to match mask resolution, neighboring pixels
        share identical embeddings, which would artificially inflate intra-class
        similarity (many zero-distance pairs). Use stride > 1 to sample from the
        original embedding resolution to avoid this bias.

        For example, if embeddings are upsampled 4x (64x64 -> 256x256),
        use stride=4 to sample at the original resolution.

    Args:
        config: EmbeddingSeparationConfig instance
        name: Name for the metric
    """

    def __init__(
        self,
        config: Optional[EmbeddingSeparationConfig] = None,
        name: str = "embedding_separation_ratio",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.config = config or EmbeddingSeparationConfig()

        # Accumulators for batch statistics
        self.inter_class_sum = self.add_weight(
            name="inter_class_sum", initializer="zeros"
        )
        self.inter_class_count = self.add_weight(
            name="inter_class_count", initializer="zeros"
        )
        self.intra_class_sum = self.add_weight(
            name="intra_class_sum", initializer="zeros"
        )
        self.intra_class_count = self.add_weight(
            name="intra_class_count", initializer="zeros"
        )

    def _compute_pairwise_distances(self, embeddings: tf.Tensor) -> tf.Tensor:
        """Compute pairwise distances using the configured metric."""
        if self.config.distance_metric == "euclidean":
            return self._euclidean_distances(embeddings)
        elif self.config.distance_metric == "cosine":
            return self._cosine_distances(embeddings)
        elif self.config.distance_metric == "manhattan":
            return self._manhattan_distances(embeddings)
        else:
            raise ValueError(f"Unsupported distance metric: {self.config.distance_metric}")

    def _euclidean_distances(self, embeddings: tf.Tensor) -> tf.Tensor:
        """Compute pairwise Euclidean distances."""
        # embeddings: (N, D)
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        dot_product = tf.matmul(embeddings, embeddings, transpose_b=True)
        square_norm = tf.linalg.diag_part(dot_product)

        distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)
        distances = tf.maximum(distances, 0.0)  # Numerical stability
        distances = tf.sqrt(distances + self.config.epsilon)

        return distances

    def _cosine_distances(self, embeddings: tf.Tensor) -> tf.Tensor:
        """Compute pairwise cosine distances (1 - cosine_similarity)."""
        normalized = tf.nn.l2_normalize(embeddings, axis=1)
        similarities = tf.matmul(normalized, normalized, transpose_b=True)
        distances = 1.0 - similarities
        return tf.maximum(distances, 0.0)

    def _manhattan_distances(self, embeddings: tf.Tensor) -> tf.Tensor:
        """Compute pairwise Manhattan (L1) distances."""
        # Use broadcasting: (N, 1, D) - (1, N, D) -> (N, N, D)
        diff = tf.expand_dims(embeddings, 1) - tf.expand_dims(embeddings, 0)
        return tf.reduce_sum(tf.abs(diff), axis=-1)

    def _labels_to_classes(self, labels: tf.Tensor) -> tf.Tensor:
        """Convert multi-channel binary masks to class indices."""
        # labels: (batch_size, h, w, num_classes)
        # Returns: (batch_size, h, w) with class indices (0=background, 1+=foreground)

        label_sum = tf.reduce_sum(labels, axis=-1)  # (batch, h, w)
        class_indices = tf.argmax(labels, axis=-1, output_type=tf.int32) + 1
        class_labels = tf.where(label_sum > 0.5, class_indices, tf.zeros_like(class_indices))

        return class_labels

    def _sample_embeddings_with_stride(
        self,
        embeddings: tf.Tensor,
        class_labels: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Sample embeddings with optional striding to handle upsampling.

        Args:
            embeddings: (batch, h, w, feature_dim)
            class_labels: (batch, h, w)

        Returns:
            sampled_embeddings: (N, feature_dim)
            sampled_labels: (N,)
        """
        shape = tf.shape(embeddings)
        batch_size = shape[0]
        h = shape[1]
        w = shape[2]
        feature_dim = shape[3]

        # Apply stride to avoid redundant upsampled pixels
        stride = self.config.stride
        if stride > 1:
            # Subsample with stride
            embeddings = embeddings[:, ::stride, ::stride, :]
            class_labels = class_labels[:, ::stride, ::stride]

            # Update dimensions
            h = tf.shape(embeddings)[1]
            w = tf.shape(embeddings)[2]

        # Flatten
        embeddings_flat = tf.reshape(embeddings, [-1, feature_dim])
        labels_flat = tf.reshape(class_labels, [-1])

        return embeddings_flat, labels_flat

    def _compute_class_distances(
        self,
        embeddings: tf.Tensor,
        labels: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Compute inter-class and intra-class distances via sampling.

        Returns:
            inter_sum: Sum of inter-class distances
            inter_count: Number of inter-class pairs
            intra_sum: Sum of intra-class distances
            intra_count: Number of intra-class pairs
        """
        # Get unique classes
        unique_classes, _, class_counts = tf.unique_with_counts(labels)
        num_classes = tf.shape(unique_classes)[0]

        # Sample from each class
        sampled_embeddings_list = []
        sampled_labels_list = []

        def sample_from_class(class_id):
            """Sample embeddings from a single class."""
            mask = tf.equal(labels, class_id)
            indices = tf.where(mask)[:, 0]
            num_available = tf.shape(indices)[0]
            num_samples = tf.minimum(num_available, self.config.max_samples_per_class)

            sampled_indices = tf.random.shuffle(indices)[:num_samples]
            sampled_emb = tf.gather(embeddings, sampled_indices)
            sampled_lab = tf.fill([num_samples], class_id)

            return sampled_emb, sampled_lab, num_samples

        # Process each class using map_fn for graph compatibility
        # We'll use a simpler approach: sample from all pixels and filter

        # Randomly shuffle all pixels
        num_pixels = tf.shape(embeddings)[0]
        shuffled_indices = tf.random.shuffle(tf.range(num_pixels))

        # Take a manageable subset
        max_total_samples = self.config.max_samples_per_class * num_classes
        max_total_samples = tf.minimum(max_total_samples, num_pixels)
        subset_indices = shuffled_indices[:max_total_samples]

        sampled_embeddings = tf.gather(embeddings, subset_indices)
        sampled_labels = tf.gather(labels, subset_indices)

        # Compute pairwise distances for the sample
        num_sampled = tf.shape(sampled_embeddings)[0]

        # Limit distance computation to avoid memory issues
        max_for_distances = tf.minimum(num_sampled, 2000)  # Cap at 2000 for 2000x2000 = 4M pairs
        sampled_embeddings = sampled_embeddings[:max_for_distances]
        sampled_labels = sampled_labels[:max_for_distances]

        pairwise_dist = self._compute_pairwise_distances(sampled_embeddings)

        # Create masks for intra-class and inter-class pairs
        labels_i = tf.expand_dims(sampled_labels, 1)  # (N, 1)
        labels_j = tf.expand_dims(sampled_labels, 0)  # (1, N)

        same_class = tf.equal(labels_i, labels_j)  # (N, N)
        diff_class = tf.logical_not(same_class)

        # Exclude diagonal (self-pairs)
        n = tf.shape(pairwise_dist)[0]
        diagonal_mask = tf.logical_not(tf.eye(n, dtype=tf.bool))

        # Compute intra-class statistics (same class, not self)
        intra_mask = tf.logical_and(same_class, diagonal_mask)

        # Optionally exclude background-to-background pairs from intra-class distance
        if self.config.exclude_background_intra:
            # Both pixels must be non-background (label > 0) for intra-class
            non_background_i = tf.greater(labels_i, 0)  # (N, 1)
            non_background_j = tf.greater(labels_j, 0)  # (1, N)
            both_non_background = tf.logical_and(non_background_i, non_background_j)  # (N, N)
            intra_mask = tf.logical_and(intra_mask, both_non_background)

        intra_mask_float = tf.cast(intra_mask, tf.float32)
        intra_sum = tf.reduce_sum(pairwise_dist * intra_mask_float)
        intra_count = tf.reduce_sum(intra_mask_float)

        # Compute inter-class statistics (different class)
        # Background is still included here - we want to measure foreground vs background separation
        inter_mask = diff_class  # Already excludes diagonal since different classes
        inter_mask_float = tf.cast(inter_mask, tf.float32)
        inter_sum = tf.reduce_sum(pairwise_dist * inter_mask_float)
        inter_count = tf.reduce_sum(inter_mask_float)

        return inter_sum, inter_count, intra_sum, intra_count

    def _compute_class_distances_exact(
        self,
        embeddings: tf.Tensor,
        labels: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Compute ALL inter-class and intra-class distances using a loop.

        This iterates through pixels in batches, computing distances from each
        batch of anchor pixels to ALL other pixels, then accumulating statistics.

        This gives exact statistics (no sampling) at the cost of more computation,
        but memory usage is controlled by processing in batches.

        Returns:
            inter_sum: Sum of inter-class distances
            inter_count: Number of inter-class pairs
            intra_sum: Sum of intra-class distances
            intra_count: Number of intra-class pairs
        """
        num_pixels = tf.shape(embeddings)[0]
        feature_dim = tf.shape(embeddings)[1]
        batch_size = self.config.batch_size_for_exact

        # Normalize embeddings once if using cosine distance
        if self.config.distance_metric == "cosine":
            embeddings = tf.nn.l2_normalize(embeddings, axis=1)

        # Initialize accumulators
        inter_sum = tf.constant(0.0, dtype=tf.float32)
        inter_count = tf.constant(0.0, dtype=tf.float32)
        intra_sum = tf.constant(0.0, dtype=tf.float32)
        intra_count = tf.constant(0.0, dtype=tf.float32)

        # Pre-compute label properties for efficiency
        all_labels = labels  # (N,)

        # Loop through anchor pixels in batches
        num_batches = (num_pixels + batch_size - 1) // batch_size

        def compute_batch_distances(batch_idx, accumulators):
            """Compute distances for one batch of anchor pixels."""
            inter_sum, inter_count, intra_sum, intra_count = accumulators

            # Get anchor batch indices
            start_idx = batch_idx * batch_size
            end_idx = tf.minimum(start_idx + batch_size, num_pixels)

            # Get anchor embeddings and labels for this batch
            anchor_embeddings = embeddings[start_idx:end_idx]  # (B, D)
            anchor_labels = labels[start_idx:end_idx]  # (B,)
            actual_batch_size = end_idx - start_idx

            # Compute distances from anchors to ALL pixels
            # anchor_embeddings: (B, D), embeddings: (N, D)
            if self.config.distance_metric == "euclidean":
                # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
                anchor_sq_norm = tf.reduce_sum(tf.square(anchor_embeddings), axis=1, keepdims=True)  # (B, 1)
                all_sq_norm = tf.reduce_sum(tf.square(embeddings), axis=1, keepdims=False)  # (N,)
                dot_product = tf.matmul(anchor_embeddings, embeddings, transpose_b=True)  # (B, N)
                sq_distances = anchor_sq_norm - 2.0 * dot_product + tf.expand_dims(all_sq_norm, 0)
                sq_distances = tf.maximum(sq_distances, 0.0)
                distances = tf.sqrt(sq_distances + self.config.epsilon)  # (B, N)
            elif self.config.distance_metric == "cosine":
                # Already normalized above
                similarities = tf.matmul(anchor_embeddings, embeddings, transpose_b=True)  # (B, N)
                distances = 1.0 - similarities
                distances = tf.maximum(distances, 0.0)
            else:  # manhattan
                # (B, 1, D) - (1, N, D) -> (B, N, D) -> sum -> (B, N)
                anchor_exp = tf.expand_dims(anchor_embeddings, 1)  # (B, 1, D)
                all_exp = tf.expand_dims(embeddings, 0)  # (1, N, D)
                distances = tf.reduce_sum(tf.abs(anchor_exp - all_exp), axis=2)  # (B, N)

            # Create masks for classification
            anchor_labels_exp = tf.expand_dims(anchor_labels, 1)  # (B, 1)
            all_labels_exp = tf.expand_dims(all_labels, 0)  # (1, N)

            same_class = tf.equal(anchor_labels_exp, all_labels_exp)  # (B, N)
            diff_class = tf.logical_not(same_class)

            # Create mask to exclude self-pairs (diagonal elements for this batch)
            # self_mask[i, j] = True if anchor i corresponds to pixel j
            batch_indices = tf.range(start_idx, end_idx)  # (B,)
            all_indices = tf.range(num_pixels)  # (N,)
            self_mask = tf.equal(
                tf.expand_dims(batch_indices, 1),  # (B, 1)
                tf.expand_dims(all_indices, 0)  # (1, N)
            )  # (B, N)
            not_self = tf.logical_not(self_mask)

            # Intra-class: same class, not self
            intra_mask = tf.logical_and(same_class, not_self)

            # Optionally exclude background-to-background from intra-class
            if self.config.exclude_background_intra:
                anchor_non_bg = tf.greater(anchor_labels_exp, 0)  # (B, 1)
                all_non_bg = tf.greater(all_labels_exp, 0)  # (1, N)
                both_non_bg = tf.logical_and(anchor_non_bg, all_non_bg)  # (B, N)
                intra_mask = tf.logical_and(intra_mask, both_non_bg)

            # Inter-class: different class (already excludes self since same pixel = same class)
            inter_mask = diff_class

            # Accumulate statistics
            intra_mask_float = tf.cast(intra_mask, tf.float32)
            inter_mask_float = tf.cast(inter_mask, tf.float32)

            batch_intra_sum = tf.reduce_sum(distances * intra_mask_float)
            batch_intra_count = tf.reduce_sum(intra_mask_float)
            batch_inter_sum = tf.reduce_sum(distances * inter_mask_float)
            batch_inter_count = tf.reduce_sum(inter_mask_float)

            new_inter_sum = inter_sum + batch_inter_sum
            new_inter_count = inter_count + batch_inter_count
            new_intra_sum = intra_sum + batch_intra_sum
            new_intra_count = intra_count + batch_intra_count

            return (new_inter_sum, new_inter_count, new_intra_sum, new_intra_count)

        # Use tf.while_loop for graph-mode compatibility
        def cond(batch_idx, accumulators):
            return batch_idx < num_batches

        def body(batch_idx, accumulators):
            new_accumulators = compute_batch_distances(batch_idx, accumulators)
            return batch_idx + 1, new_accumulators

        initial_accumulators = (inter_sum, inter_count, intra_sum, intra_count)
        _, final_accumulators = tf.while_loop(
            cond,
            body,
            loop_vars=(tf.constant(0), initial_accumulators),
            parallel_iterations=1  # Process sequentially to control memory
        )

        inter_sum, inter_count, intra_sum, intra_count = final_accumulators

        # Note: Each pair (i, j) is counted twice (once when i is anchor, once when j is anchor)
        # So we divide by 2 to get the correct count
        # Actually, the sum is also doubled, so the mean is correct without adjustment
        # But for reporting purposes, let's keep the counts as-is since they represent
        # the number of distance computations, not unique pairs

        return inter_sum, inter_count, intra_sum, intra_count

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        """
        Update metric state with a batch of predictions.

        Args:
            y_true: Ground truth masks (batch, h, w, num_classes)
            y_pred: Predicted embeddings (batch, h', w', feature_dim)
                   May have different spatial dimensions if upsampling is used
            sample_weight: Optional sample weights (unused)
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Get dimensions
        pred_h, pred_w = tf.shape(y_pred)[1], tf.shape(y_pred)[2]
        label_h, label_w = tf.shape(y_true)[1], tf.shape(y_true)[2]

        # Upsample embeddings to match label resolution if needed
        def resize_embeddings():
            return tf.image.resize(y_pred, [label_h, label_w], method='nearest')

        def no_resize():
            return y_pred

        embeddings = tf.cond(
            tf.logical_or(
                tf.not_equal(pred_h, label_h),
                tf.not_equal(pred_w, label_w)
            ),
            resize_embeddings,
            no_resize
        )

        # Convert labels to class indices
        class_labels = self._labels_to_classes(y_true)

        # Sample embeddings with stride
        sampled_embeddings, sampled_labels = self._sample_embeddings_with_stride(
            embeddings, class_labels
        )

        # Compute distance statistics (sampled or exact)
        if self.config.use_exact:
            inter_sum, inter_count, intra_sum, intra_count = self._compute_class_distances_exact(
                sampled_embeddings, sampled_labels
            )
        else:
            inter_sum, inter_count, intra_sum, intra_count = self._compute_class_distances(
                sampled_embeddings, sampled_labels
            )

        # Update accumulators
        self.inter_class_sum.assign_add(inter_sum)
        self.inter_class_count.assign_add(inter_count)
        self.intra_class_sum.assign_add(intra_sum)
        self.intra_class_count.assign_add(intra_count)

    def result(self) -> tf.Tensor:
        """
        Compute the separation ratio.

        Returns:
            separation_ratio: mean(inter-class dist) / mean(intra-class dist)
        """
        mean_inter = self.inter_class_sum / (self.inter_class_count + self.config.epsilon)
        mean_intra = self.intra_class_sum / (self.intra_class_count + self.config.epsilon)

        # Separation ratio
        ratio = mean_inter / (mean_intra + self.config.epsilon)

        return ratio

    def reset_state(self):
        """Reset metric state."""
        self.inter_class_sum.assign(0.0)
        self.inter_class_count.assign(0.0)
        self.intra_class_sum.assign(0.0)
        self.intra_class_count.assign(0.0)

    def get_config(self):
        """Return configuration for serialization."""
        base_config = super().get_config()
        config_dict = {
            'max_samples_per_class': self.config.max_samples_per_class,
            'max_pairs_per_class': self.config.max_pairs_per_class,
            'distance_metric': self.config.distance_metric,
            'stride': self.config.stride,
            'exclude_background_intra': self.config.exclude_background_intra,
            'use_exact': self.config.use_exact,
            'batch_size_for_exact': self.config.batch_size_for_exact,
            'epsilon': self.config.epsilon,
        }
        return {**base_config, **config_dict}

    @classmethod
    def from_config(cls, config):
        """Create instance from configuration."""
        sep_config = EmbeddingSeparationConfig(
            max_samples_per_class=config.pop('max_samples_per_class', 500),
            max_pairs_per_class=config.pop('max_pairs_per_class', 10000),
            distance_metric=config.pop('distance_metric', 'euclidean'),
            stride=config.pop('stride', 1),
            exclude_background_intra=config.pop('exclude_background_intra', True),
            use_exact=config.pop('use_exact', False),
            batch_size_for_exact=config.pop('batch_size_for_exact', 100),
            epsilon=config.pop('epsilon', 1e-8),
        )
        return cls(config=sep_config, **config)


class EmbeddingIntraClassDistance(keras.metrics.Metric):
    """
    Metric that tracks the mean intra-class (within-class) distance.

    Lower values indicate tighter clustering of same-class embeddings.
    Useful for monitoring whether embeddings are becoming more compact.
    """

    def __init__(
        self,
        config: Optional[EmbeddingSeparationConfig] = None,
        name: str = "embedding_intra_class_dist",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.config = config or EmbeddingSeparationConfig()
        self.separation_metric = EmbeddingSeparationRatio(config=self.config, name="_internal")

        self.intra_sum = self.add_weight(name="intra_sum", initializer="zeros")
        self.intra_count = self.add_weight(name="intra_count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.separation_metric.update_state(y_true, y_pred, sample_weight)
        # Copy the intra-class stats
        self.intra_sum.assign(self.separation_metric.intra_class_sum)
        self.intra_count.assign(self.separation_metric.intra_class_count)

    def result(self):
        return self.intra_sum / (self.intra_count + self.config.epsilon)

    def reset_state(self):
        self.separation_metric.reset_state()
        self.intra_sum.assign(0.0)
        self.intra_count.assign(0.0)


class EmbeddingInterClassDistance(keras.metrics.Metric):
    """
    Metric that tracks the mean inter-class (between-class) distance.

    Higher values indicate better separation between different classes.
    Useful for monitoring whether embeddings are becoming more discriminative.
    """

    def __init__(
        self,
        config: Optional[EmbeddingSeparationConfig] = None,
        name: str = "embedding_inter_class_dist",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.config = config or EmbeddingSeparationConfig()
        self.separation_metric = EmbeddingSeparationRatio(config=self.config, name="_internal")

        self.inter_sum = self.add_weight(name="inter_sum", initializer="zeros")
        self.inter_count = self.add_weight(name="inter_count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.separation_metric.update_state(y_true, y_pred, sample_weight)
        # Copy the inter-class stats
        self.inter_sum.assign(self.separation_metric.inter_class_sum)
        self.inter_count.assign(self.separation_metric.inter_class_count)

    def result(self):
        return self.inter_sum / (self.inter_count + self.config.epsilon)

    def reset_state(self):
        self.separation_metric.reset_state()
        self.inter_sum.assign(0.0)
        self.inter_count.assign(0.0)


# Convenience function
def create_embedding_separation_metric(
    max_samples_per_class: int = 500,
    distance_metric: str = "euclidean",
    stride: int = 1,
    exclude_background_intra: bool = True,
    use_exact: bool = False,
    batch_size_for_exact: int = 100,
    name: str = "embedding_separation_ratio"
) -> EmbeddingSeparationRatio:
    """
    Create an embedding separation ratio metric.

    Args:
        max_samples_per_class: Maximum pixels to sample per class for distance computation.
            More samples = more accurate estimate but slower computation.
            Only used when use_exact=False.
        distance_metric: Distance metric to use ('euclidean', 'cosine', 'manhattan').
            Should match the metric used in your triplet loss.
        stride: Stride for sampling pixels. Use this to handle upsampling:
            - If embeddings are upsampled 4x (e.g., 64x64 -> 256x256), use stride=4
            - This avoids counting duplicate embeddings from upsampling
            - Set to 1 if embeddings are at native resolution
        exclude_background_intra: If True (default), exclude background-to-background pairs
            from intra-class distance computation. This focuses the metric on foreground
            cluster tightness while still measuring foreground-vs-background separation.
        use_exact: If True, compute ALL pairwise distances via looping instead of sampling.
            This gives exact statistics but takes longer. Default is False (sampling).
        batch_size_for_exact: Number of anchor pixels to process per iteration when
            use_exact=True. Larger batches are faster but use more memory. Default is 100.
        name: Name for the metric.

    Returns:
        EmbeddingSeparationRatio metric instance.

    Example:
        # For embeddings upsampled 4x to match 256x256 masks (sampling mode)
        metric = create_embedding_separation_metric(
            max_samples_per_class=500,
            distance_metric='euclidean',
            stride=4  # Match upsampling factor
        )

        # For exact computation (slower but more accurate)
        metric_exact = create_embedding_separation_metric(
            distance_metric='euclidean',
            stride=4,
            use_exact=True,
            batch_size_for_exact=200
        )

        model.compile(
            optimizer='adam',
            loss=pixel_triplet_loss,
            metrics=[metric]
        )
    """
    config = EmbeddingSeparationConfig(
        max_samples_per_class=max_samples_per_class,
        distance_metric=distance_metric,
        stride=stride,
        exclude_background_intra=exclude_background_intra,
        use_exact=use_exact,
        batch_size_for_exact=batch_size_for_exact,
    )
    return EmbeddingSeparationRatio(config=config, name=name)


def create_embedding_metrics_suite(
    max_samples_per_class: int = 500,
    distance_metric: str = "euclidean",
    stride: int = 1,
    exclude_background_intra: bool = True,
    use_exact: bool = False,
    batch_size_for_exact: int = 100,
) -> Dict[str, keras.metrics.Metric]:
    """
    Create a suite of embedding quality metrics.

    Returns a dictionary with:
    - 'separation_ratio': Inter/intra class distance ratio (higher = better)
    - 'intra_class_dist': Mean within-class distance (lower = tighter clusters)
    - 'inter_class_dist': Mean between-class distance (higher = better separation)

    Args:
        max_samples_per_class: Maximum pixels to sample per class.
            Only used when use_exact=False.
        distance_metric: Distance metric ('euclidean', 'cosine', 'manhattan').
        stride: Stride for sampling (use upsampling factor if applicable).
        exclude_background_intra: If True (default), exclude background-to-background pairs
            from intra-class distance. Focuses metric on foreground cluster quality.
        use_exact: If True, compute ALL pairwise distances via looping instead of sampling.
        batch_size_for_exact: Number of anchor pixels per iteration when use_exact=True.

    Returns:
        Dictionary of metric instances.

    Example:
        metrics = create_embedding_metrics_suite(stride=4)
        model.compile(
            optimizer='adam',
            loss=pixel_triplet_loss,
            metrics=list(metrics.values())
        )
    """
    config = EmbeddingSeparationConfig(
        max_samples_per_class=max_samples_per_class,
        distance_metric=distance_metric,
        stride=stride,
        exclude_background_intra=exclude_background_intra,
        use_exact=use_exact,
        batch_size_for_exact=batch_size_for_exact,
    )

    return {
        'separation_ratio': EmbeddingSeparationRatio(config=config),
        'intra_class_dist': EmbeddingIntraClassDistance(config=config),
        'inter_class_dist': EmbeddingInterClassDistance(config=config),
    }

