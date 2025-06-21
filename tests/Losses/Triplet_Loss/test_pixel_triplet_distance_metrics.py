"""
Tests for distance metric implementations in pixel triplet loss.

This module tests the newly implemented cosine and Manhattan distance metrics
to ensure they work correctly and produce expected results.
"""

import pytest
import numpy as np
import tensorflow as tf
import keras

from src.DeepLearningUtils.Losses.Triplet_Loss.pixel_triplet_loss_keras import (
    PixelTripletLoss,
    PixelTripletConfig,
    create_pixel_triplet_loss,
)
from tests.testing_utilities import assert_arrays_equal_with_nans


class TestDistanceMetricImplementations:
    """Test the actual distance metric implementations."""

    @pytest.fixture
    def known_embeddings(self):
        """Create embeddings with known geometric relationships."""
        embeddings = tf.constant([
            [1.0, 0.0, 0.0],   # Unit vector along x-axis
            [0.0, 1.0, 0.0],   # Unit vector along y-axis
            [-1.0, 0.0, 0.0],  # Opposite to first vector
            [2.0, 0.0, 0.0],   # Same direction as first, different magnitude
            [0.0, 0.0, 0.0],   # Zero vector
        ], dtype=tf.float32)
        return embeddings

    def test_euclidean_distance_implementation(self, keras_float32_policy, known_embeddings):
        """Test that Euclidean distance implementation matches expected values."""
        config = PixelTripletConfig(distance_metric="euclidean")
        loss_fn = PixelTripletLoss(config=config)
        
        distances = loss_fn._compute_pairwise_distances(known_embeddings)
        
        print(f"\nEuclidean distances:")
        print(f"[1,0,0] to [0,1,0]: {distances[0, 1].numpy():.4f}")  # Should be sqrt(2) ≈ 1.414
        print(f"[1,0,0] to [-1,0,0]: {distances[0, 2].numpy():.4f}")  # Should be 2.0
        print(f"[1,0,0] to [2,0,0]: {distances[0, 3].numpy():.4f}")   # Should be 1.0
        print(f"[1,0,0] to [0,0,0]: {distances[0, 4].numpy():.4f}")   # Should be 1.0
        
        # Check expected values
        assert abs(distances[0, 1].numpy() - np.sqrt(2)) < 1e-6
        assert abs(distances[0, 2].numpy() - 2.0) < 1e-6
        assert abs(distances[0, 3].numpy() - 1.0) < 1e-6
        assert abs(distances[0, 4].numpy() - 1.0) < 1e-6
        
        # Matrix should be symmetric
        assert abs(distances[0, 1].numpy() - distances[1, 0].numpy()) < 1e-6
        
        # Diagonal should be zero
        assert abs(distances[0, 0].numpy()) < 1e-6

    def test_cosine_distance_implementation(self, keras_float32_policy, known_embeddings):
        """Test that cosine distance implementation matches expected values."""
        config = PixelTripletConfig(distance_metric="cosine")
        loss_fn = PixelTripletLoss(config=config)
        
        distances = loss_fn._compute_pairwise_distances(known_embeddings)
        
        print(f"\nCosine distances:")
        print(f"[1,0,0] to [0,1,0]: {distances[0, 1].numpy():.4f}")  # Should be 1.0 (orthogonal)
        print(f"[1,0,0] to [-1,0,0]: {distances[0, 2].numpy():.4f}")  # Should be 2.0 (opposite)
        print(f"[1,0,0] to [2,0,0]: {distances[0, 3].numpy():.4f}")   # Should be 0.0 (same direction)
        print(f"[1,0,0] to [0,0,0]: {distances[0, 4].numpy():.4f}")   # Undefined (zero vector)
        
        # Check expected values (except zero vector case)
        assert abs(distances[0, 1].numpy() - 1.0) < 1e-6  # Orthogonal vectors
        assert abs(distances[0, 2].numpy() - 2.0) < 1e-6  # Opposite vectors
        assert abs(distances[0, 3].numpy() - 0.0) < 1e-6  # Same direction
        
        # Matrix should be symmetric
        assert abs(distances[0, 1].numpy() - distances[1, 0].numpy()) < 1e-6
        
        # Diagonal should be zero
        assert abs(distances[0, 0].numpy()) < 1e-6

    def test_manhattan_distance_implementation(self, keras_float32_policy, known_embeddings):
        """Test that Manhattan distance implementation matches expected values."""
        config = PixelTripletConfig(distance_metric="manhattan")
        loss_fn = PixelTripletLoss(config=config)
        
        distances = loss_fn._compute_pairwise_distances(known_embeddings)
        
        print(f"\nManhattan distances:")
        print(f"[1,0,0] to [0,1,0]: {distances[0, 1].numpy():.4f}")  # Should be |1-0| + |0-1| + |0-0| = 2.0
        print(f"[1,0,0] to [-1,0,0]: {distances[0, 2].numpy():.4f}")  # Should be |1-(-1)| + |0-0| + |0-0| = 2.0
        print(f"[1,0,0] to [2,0,0]: {distances[0, 3].numpy():.4f}")   # Should be |1-2| + |0-0| + |0-0| = 1.0
        print(f"[1,0,0] to [0,0,0]: {distances[0, 4].numpy():.4f}")   # Should be |1-0| + |0-0| + |0-0| = 1.0
        
        # Check expected values
        assert abs(distances[0, 1].numpy() - 2.0) < 1e-6
        assert abs(distances[0, 2].numpy() - 2.0) < 1e-6
        assert abs(distances[0, 3].numpy() - 1.0) < 1e-6
        assert abs(distances[0, 4].numpy() - 1.0) < 1e-6
        
        # Matrix should be symmetric
        assert abs(distances[0, 1].numpy() - distances[1, 0].numpy()) < 1e-6
        
        # Diagonal should be zero
        assert abs(distances[0, 0].numpy()) < 1e-6


class TestDistanceMetricEffects:
    """Test how different distance metrics affect triplet loss results."""

    @pytest.fixture
    def simple_segmentation_data(self):
        """Create simple segmentation data for testing different distance metrics."""
        batch_size = 1
        h, w = 6, 6
        feature_dim = 3
        num_whiskers = 2
        
        # Create embeddings with clear separations that behave differently for each metric
        embeddings = tf.zeros((batch_size, h, w, feature_dim), dtype=tf.float32)
        
        # Class 1: [1, 0, 0] (unit vector along x-axis)
        embeddings = tf.tensor_scatter_nd_update(
            embeddings,
            [[0, i, j, 0] for i in range(2) for j in range(2)],
            [1.0] * 4
        )
        
        # Class 2: [0, 1, 0] (unit vector along y-axis, orthogonal to class 1)
        embeddings = tf.tensor_scatter_nd_update(
            embeddings,
            [[0, i, j, 1] for i in range(3, 5) for j in range(2)],
            [1.0] * 4
        )
        
        # Background: [0, 0, 0] (zero vector)
        
        # Create corresponding labels
        labels = tf.zeros((batch_size, h, w, num_whiskers), dtype=tf.float32)
        
        # Whisker 0
        labels = tf.tensor_scatter_nd_update(labels,
                                           [[0, i, j, 0] for i in range(2) for j in range(2)],
                                           [1.0] * 4)
        
        # Whisker 1
        labels = tf.tensor_scatter_nd_update(labels,
                                           [[0, i, j, 1] for i in range(3, 5) for j in range(2)],
                                           [1.0] * 4)
        
        return embeddings, labels

    def test_euclidean_metric_loss(self, keras_float32_policy, simple_segmentation_data):
        """Test triplet loss with Euclidean distance metric."""
        embeddings, labels = simple_segmentation_data
        
        config = PixelTripletConfig(
            distance_metric="euclidean",
            margin=0.5,
            background_pixels=10,
            whisker_pixels=5,
            triplet_strategy="hard"
        )
        loss_fn = PixelTripletLoss(config=config)
        
        loss = loss_fn(labels, embeddings)
        
        print(f"\nEuclidean loss with margin 0.5: {loss.numpy():.4f}")
        
        assert not tf.math.is_nan(loss)
        assert not tf.math.is_inf(loss)
        assert loss.numpy() >= 0.0

    def test_cosine_metric_loss(self, keras_float32_policy, simple_segmentation_data):
        """Test triplet loss with cosine distance metric."""
        embeddings, labels = simple_segmentation_data
        
        config = PixelTripletConfig(
            distance_metric="cosine",
            margin=0.3,  # Smaller margin for cosine since max distance is 2.0
            background_pixels=10,
            whisker_pixels=5,
            triplet_strategy="hard"
        )
        loss_fn = PixelTripletLoss(config=config)
        
        loss = loss_fn(labels, embeddings)
        
        print(f"\nCosine loss with margin 0.3: {loss.numpy():.4f}")
        
        assert not tf.math.is_nan(loss)
        assert not tf.math.is_inf(loss)
        assert loss.numpy() >= 0.0

    def test_manhattan_metric_loss(self, keras_float32_policy, simple_segmentation_data):
        """Test triplet loss with Manhattan distance metric."""
        embeddings, labels = simple_segmentation_data
        
        config = PixelTripletConfig(
            distance_metric="manhattan",
            margin=1.0,  # Larger margin for Manhattan since distances are typically larger
            background_pixels=10,
            whisker_pixels=5,
            triplet_strategy="hard"
        )
        loss_fn = PixelTripletLoss(config=config)
        
        loss = loss_fn(labels, embeddings)
        
        print(f"\nManhattan loss with margin 1.0: {loss.numpy():.4f}")
        
        assert not tf.math.is_nan(loss)
        assert not tf.math.is_inf(loss)
        assert loss.numpy() >= 0.0

    def test_distance_metric_comparison(self, keras_float32_policy, simple_segmentation_data):
        """Compare all three distance metrics with the same data."""
        embeddings, labels = simple_segmentation_data
        
        # Test all metrics with appropriately scaled margins
        metrics_configs = [
            ("euclidean", 0.5),
            ("cosine", 0.3),
            ("manhattan", 1.0),
        ]
        
        loss_values = {}
        
        print(f"\nDistance metric comparison:")
        
        for metric, margin in metrics_configs:
            config = PixelTripletConfig(
                distance_metric=metric,
                margin=margin,
                background_pixels=10,
                whisker_pixels=5,
                triplet_strategy="hard"
            )
            loss_fn = PixelTripletLoss(config=config)
            
            loss = loss_fn(labels, embeddings)
            loss_values[metric] = loss.numpy()
            
            print(f"  {metric:10} (margin {margin:3.1f}): loss = {loss.numpy():.4f}")
            
            assert not tf.math.is_nan(loss)
            assert not tf.math.is_inf(loss)
            assert loss.numpy() >= 0.0
        
        # All metrics should produce reasonable loss values
        for metric, loss_val in loss_values.items():
            assert 0.0 <= loss_val <= 100.0, f"{metric} loss out of reasonable range: {loss_val}"


class TestDistanceMetricEdgeCases:
    """Test edge cases for different distance metrics."""

    def test_identical_embeddings_all_metrics(self, keras_float32_policy):
        """Test all distance metrics with identical embeddings."""
        batch_size = 1
        h, w = 4, 4
        feature_dim = 3
        num_whiskers = 2
        
        # Create identical embeddings everywhere
        embeddings = tf.ones((batch_size, h, w, feature_dim), dtype=tf.float32)
        labels = tf.zeros((batch_size, h, w, num_whiskers), dtype=tf.float32)
        
        # Add some labels
        labels = tf.tensor_scatter_nd_update(labels,
                                           [[0, i, j, 0] for i in range(2) for j in range(2)],
                                           [1.0] * 4)
        labels = tf.tensor_scatter_nd_update(labels,
                                           [[0, i, j, 1] for i in range(2, 4) for j in range(2)],
                                           [1.0] * 4)
        
        metrics = ["euclidean", "cosine", "manhattan"]
        margins = [0.5, 0.3, 1.0]
        
        print(f"\nIdentical embeddings test:")
        
        for metric, margin in zip(metrics, margins):
            config = PixelTripletConfig(
                distance_metric=metric,
                margin=margin,
                background_pixels=5,
                whisker_pixels=3,
                triplet_strategy="hard"
            )
            loss_fn = PixelTripletLoss(config=config)
            
            loss = loss_fn(labels, embeddings)
            
            print(f"  {metric:10}: loss = {loss.numpy():.4f}")
            
            # With identical embeddings, all distances should be 0
            # So loss should be equal to the margin (since d(a,p) = d(a,n) = 0)
            assert not tf.math.is_nan(loss)
            assert not tf.math.is_inf(loss)
            assert loss.numpy() >= 0.0

    def test_zero_embeddings_handling(self, keras_float32_policy):
        """Test how different metrics handle zero embeddings."""
        batch_size = 1
        h, w = 4, 4
        feature_dim = 3
        num_whiskers = 2
        
        # Create zero embeddings everywhere
        embeddings = tf.zeros((batch_size, h, w, feature_dim), dtype=tf.float32)
        labels = tf.zeros((batch_size, h, w, num_whiskers), dtype=tf.float32)
        
        # Add some labels
        labels = tf.tensor_scatter_nd_update(labels,
                                           [[0, i, j, 0] for i in range(2) for j in range(2)],
                                           [1.0] * 4)
        labels = tf.tensor_scatter_nd_update(labels,
                                           [[0, i, j, 1] for i in range(2, 4) for j in range(2)],
                                           [1.0] * 4)
        
        metrics = ["euclidean", "cosine", "manhattan"]
        margins = [0.5, 0.3, 1.0]
        
        print(f"\nZero embeddings test:")
        
        for metric, margin in zip(metrics, margins):
            config = PixelTripletConfig(
                distance_metric=metric,
                margin=margin,
                background_pixels=5,
                whisker_pixels=3,
                triplet_strategy="hard"
            )
            loss_fn = PixelTripletLoss(config=config)
            
            loss = loss_fn(labels, embeddings)
            
            print(f"  {metric:10}: loss = {loss.numpy():.4f}")
            
            # All should handle zero embeddings gracefully
            assert not tf.math.is_nan(loss)
            assert not tf.math.is_inf(loss)
            assert loss.numpy() >= 0.0

    def test_large_magnitude_embeddings(self, keras_float32_policy):
        """Test distance metrics with large magnitude embeddings."""
        batch_size = 1
        h, w = 4, 4
        feature_dim = 3
        num_whiskers = 2
        
        # Create large magnitude embeddings
        embeddings = tf.zeros((batch_size, h, w, feature_dim), dtype=tf.float32)
        
        # Class 1: [100, 0, 0]
        embeddings = tf.tensor_scatter_nd_update(
            embeddings,
            [[0, i, j, 0] for i in range(2) for j in range(2)],
            [100.0] * 4
        )
        
        # Class 2: [0, 100, 0]
        embeddings = tf.tensor_scatter_nd_update(
            embeddings,
            [[0, i, j, 1] for i in range(2, 4) for j in range(2)],
            [100.0] * 4
        )
        
        labels = tf.zeros((batch_size, h, w, num_whiskers), dtype=tf.float32)
        labels = tf.tensor_scatter_nd_update(labels,
                                           [[0, i, j, 0] for i in range(2) for j in range(2)],
                                           [1.0] * 4)
        labels = tf.tensor_scatter_nd_update(labels,
                                           [[0, i, j, 1] for i in range(2, 4) for j in range(2)],
                                           [1.0] * 4)
        
        metrics = ["euclidean", "cosine", "manhattan"]
        margins = [10.0, 0.5, 50.0]  # Adjusted margins for large values
        
        print(f"\nLarge magnitude embeddings test:")
        
        for metric, margin in zip(metrics, margins):
            config = PixelTripletConfig(
                distance_metric=metric,
                margin=margin,
                background_pixels=5,
                whisker_pixels=3,
                triplet_strategy="hard"
            )
            loss_fn = PixelTripletLoss(config=config)
            
            loss = loss_fn(labels, embeddings)
            
            print(f"  {metric:10} (margin {margin:4.1f}): loss = {loss.numpy():.4f}")
            
            assert not tf.math.is_nan(loss)
            assert not tf.math.is_inf(loss)
            assert loss.numpy() >= 0.0


class TestDistanceMetricConsistency:
    """Test that the new implementations are consistent with mathematical definitions."""

    def test_distance_properties(self, keras_float32_policy):
        """Test that all distance metrics satisfy basic distance properties."""
        embeddings = tf.random.normal((5, 4), dtype=tf.float32)
        
        metrics = ["euclidean", "cosine", "manhattan"]
        
        for metric in metrics:
            config = PixelTripletConfig(distance_metric=metric)
            loss_fn = PixelTripletLoss(config=config)
            
            distances = loss_fn._compute_pairwise_distances(embeddings)
            
            print(f"\nTesting {metric} distance properties:")
            
            # 1. Non-negativity: d(x,y) >= 0
            assert tf.reduce_all(distances >= 0.0), f"{metric}: distances should be non-negative"
            
            # 2. Identity: d(x,x) = 0
            diagonal = tf.linalg.diag_part(distances)
            assert tf.reduce_all(tf.abs(diagonal) < 1e-6), f"{metric}: diagonal should be zero"
            
            # 3. Symmetry: d(x,y) = d(y,x)
            assert tf.reduce_all(tf.abs(distances - tf.transpose(distances)) < 1e-6), f"{metric}: should be symmetric"
            
            print(f"  ✓ All distance properties satisfied for {metric}")

    def test_metric_specific_properties(self, keras_float32_policy):
        """Test properties specific to each distance metric."""
        # Create specific test vectors
        vec_a = tf.constant([[1.0, 0.0]], dtype=tf.float32)
        vec_b = tf.constant([[0.0, 1.0]], dtype=tf.float32)
        vec_c = tf.constant([[-1.0, 0.0]], dtype=tf.float32)
        vec_d = tf.constant([[2.0, 0.0]], dtype=tf.float32)
        
        test_embeddings = tf.concat([vec_a, vec_b, vec_c, vec_d], axis=0)
        
        # Test Euclidean properties
        config_euclidean = PixelTripletConfig(distance_metric="euclidean")
        loss_fn_euclidean = PixelTripletLoss(config=config_euclidean)
        euclidean_dist = loss_fn_euclidean._compute_pairwise_distances(test_embeddings)
        
        # Euclidean: d([1,0], [0,1]) = sqrt(2)
        assert abs(euclidean_dist[0, 1].numpy() - np.sqrt(2)) < 1e-6
        
        # Test Cosine properties
        config_cosine = PixelTripletConfig(distance_metric="cosine")
        loss_fn_cosine = PixelTripletLoss(config=config_cosine)
        cosine_dist = loss_fn_cosine._compute_pairwise_distances(test_embeddings)
        
        # Cosine: d([1,0], [2,0]) = 0 (same direction)
        assert abs(cosine_dist[0, 3].numpy() - 0.0) < 1e-6
        # Cosine: d([1,0], [-1,0]) = 2.0 (opposite direction)
        assert abs(cosine_dist[0, 2].numpy() - 2.0) < 1e-6
        
        # Test Manhattan properties
        config_manhattan = PixelTripletConfig(distance_metric="manhattan")
        loss_fn_manhattan = PixelTripletLoss(config=config_manhattan)
        manhattan_dist = loss_fn_manhattan._compute_pairwise_distances(test_embeddings)
        
        # Manhattan: d([1,0], [0,1]) = |1-0| + |0-1| = 2
        assert abs(manhattan_dist[0, 1].numpy() - 2.0) < 1e-6
        
        print(f"\n✓ All metric-specific properties verified")