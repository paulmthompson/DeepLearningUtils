"""
Tests for pixel-based triplet loss with multiple classes (>2).

This module tests edge cases and potential issues that arise when using 
pixel triplet loss with multiple whisker classes and various class distributions.
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


class TestMultiClassPixelTripletLoss:
    """Test pixel triplet loss with multiple classes."""

    @pytest.fixture
    def multiclass_segmentation_data(self):
        """Create segmentation data with multiple whisker classes."""
        batch_size = 2
        h, w = 8, 8
        feature_dim = 4
        num_whiskers = 5  # 5 different whiskers
        
        # Create embeddings with distinct patterns for each whisker
        embeddings = tf.zeros((batch_size, h, w, feature_dim), dtype=tf.float32)
        
        # Batch 1: Different spatial regions for different whiskers
        # Top-left: whisker 0 (class 1)
        embeddings = tf.tensor_scatter_nd_update(
            embeddings,
            [[0, i, j, 0] for i in range(2) for j in range(2)],
            [1.0] * 4
        )
        
        # Top-right: whisker 1 (class 2)  
        embeddings = tf.tensor_scatter_nd_update(
            embeddings,
            [[0, i, j, 1] for i in range(2) for j in range(4, 6)],
            [1.0] * 4
        )
        
        # Middle-left: whisker 2 (class 3)
        embeddings = tf.tensor_scatter_nd_update(
            embeddings,
            [[0, i, j, 2] for i in range(3, 5) for j in range(2)],
            [1.0] * 4
        )
        
        # Middle-right: whisker 3 (class 4)
        embeddings = tf.tensor_scatter_nd_update(
            embeddings,
            [[0, i, j, 3] for i in range(3, 5) for j in range(4, 6)],
            [1.0] * 4
        )
        
        # Bottom: whisker 4 (class 5)
        embeddings = tf.tensor_scatter_nd_update(
            embeddings,
            [[0, i, j, 0] for i in range(6, 8) for j in range(2, 6)] +
            [[0, i, j, 1] for i in range(6, 8) for j in range(2, 6)],
            [0.5] * 16
        )
        
        # Batch 2: Different pattern - create indices and values separately
        batch2_indices = []
        batch2_values = []
        for i in range(h):
            for j in range(w):
                for k in range(feature_dim):
                    batch2_indices.append([1, i, j, k])
                    batch2_values.append(0.1 * (i + j + k))
        
        embeddings = tf.tensor_scatter_nd_update(
            embeddings,
            batch2_indices,
            batch2_values
        )
        
        # Create corresponding labels
        labels = tf.zeros((batch_size, h, w, num_whiskers), dtype=tf.float32)
        
        # Batch 1 labels
        # Top-left: whisker 0
        labels = tf.tensor_scatter_nd_update(
            labels,
            [[0, i, j, 0] for i in range(2) for j in range(2)],
            [1.0] * 4
        )
        
        # Top-right: whisker 1
        labels = tf.tensor_scatter_nd_update(
            labels,
            [[0, i, j, 1] for i in range(2) for j in range(4, 6)],
            [1.0] * 4
        )
        
        # Middle-left: whisker 2
        labels = tf.tensor_scatter_nd_update(
            labels,
            [[0, i, j, 2] for i in range(3, 5) for j in range(2)],
            [1.0] * 4
        )
        
        # Middle-right: whisker 3
        labels = tf.tensor_scatter_nd_update(
            labels,
            [[0, i, j, 3] for i in range(3, 5) for j in range(4, 6)],
            [1.0] * 4
        )
        
        # Bottom: whisker 4
        labels = tf.tensor_scatter_nd_update(
            labels,
            [[0, i, j, 4] for i in range(6, 8) for j in range(2, 6)],
            [1.0] * 8
        )
        
        # Batch 2: Sparse whisker placement
        labels = tf.tensor_scatter_nd_update(
            labels,
            [[1, 1, 1, 0], [1, 2, 3, 1], [1, 4, 5, 2], [1, 6, 1, 3], [1, 7, 7, 4]],
            [1.0] * 5
        )
        
        return embeddings, labels

    def test_five_class_basic_functionality(self, keras_float32_policy, multiclass_segmentation_data):
        """Test basic functionality with 5 classes (background + 4 whiskers)."""
        embeddings, labels = multiclass_segmentation_data
        
        config = PixelTripletConfig(
            background_pixels=20,
            whisker_pixels=10,
            margin=1.0
        )
        loss_fn = PixelTripletLoss(config=config)
        
        loss = loss_fn(labels, embeddings)
        
        # Should work without errors
        assert loss.shape == ()
        assert not tf.math.is_nan(loss)
        assert not tf.math.is_inf(loss)
        assert loss.numpy() >= 0.0

    def test_class_distribution_analysis(self, keras_float32_policy, multiclass_segmentation_data):
        """Test that all classes are properly identified and sampled."""
        embeddings, labels = multiclass_segmentation_data
        
        config = PixelTripletConfig(background_pixels=10, whisker_pixels=5)
        loss_fn = PixelTripletLoss(config=config)
        
        # Test the helper functions directly
        y_true_resized = loss_fn._resize_labels(labels, 8, 8)
        class_labels = loss_fn._labels_to_classes(y_true_resized)
        
        # Check that we have multiple classes
        unique_classes = tf.unique(tf.reshape(class_labels, [-1]))[0]
        
        # Should have background (0) and multiple whisker classes (1, 2, 3, 4, 5)
        assert len(unique_classes) >= 3  # At least background + 2 whiskers
        assert 0 in unique_classes.numpy()  # Background should be present
        
        # Test sampling
        sampled_embeddings, sampled_labels = loss_fn._sample_pixels_per_class_simple(
            embeddings, class_labels
        )
        
        unique_sampled = tf.unique(sampled_labels)[0]
        
        # Should have sampled from multiple classes
        assert len(unique_sampled) >= 2
        assert sampled_embeddings.shape[0] > 0

    def test_class_imbalance_scenario(self, keras_float32_policy):
        """Test with highly imbalanced classes."""
        batch_size = 1
        h, w = 10, 10
        feature_dim = 3
        num_whiskers = 4
        
        # Create embeddings
        embeddings = tf.random.normal((batch_size, h, w, feature_dim))
        
        # Create highly imbalanced labels
        labels = tf.zeros((batch_size, h, w, num_whiskers), dtype=tf.float32)
        
        # Whisker 0: 1 pixel
        labels = tf.tensor_scatter_nd_update(labels, [[0, 0, 0, 0]], [1.0])
        
        # Whisker 1: 2 pixels  
        labels = tf.tensor_scatter_nd_update(labels, [[0, 1, 1, 1], [0, 1, 2, 1]], [1.0, 1.0])
        
        # Whisker 2: 5 pixels
        labels = tf.tensor_scatter_nd_update(labels, 
                                           [[0, i, 3, 2] for i in range(5)], 
                                           [1.0] * 5)
        
        # Whisker 3: 20 pixels
        labels = tf.tensor_scatter_nd_update(labels,
                                           [[0, i, j, 3] for i in range(5, 9) for j in range(5)],
                                           [1.0] * 20)
        
        # Most pixels are background (72 pixels)
        
        config = PixelTripletConfig(
            background_pixels=30,
            whisker_pixels=10,  # More than available for some classes
            margin=1.0
        )
        loss_fn = PixelTripletLoss(config=config)
        
        loss = loss_fn(labels, embeddings)
        
        # Should handle imbalance gracefully
        assert loss.shape == ()
        assert not tf.math.is_nan(loss)
        assert not tf.math.is_inf(loss)

    def test_missing_class_ids(self, keras_float32_policy):
        """Test with non-contiguous class IDs (e.g., classes 0, 1, 3, 5 but no 2, 4)."""
        batch_size = 1
        h, w = 6, 6
        feature_dim = 3
        num_whiskers = 6  # But we won't use all of them
        
        embeddings = tf.random.normal((batch_size, h, w, feature_dim))
        labels = tf.zeros((batch_size, h, w, num_whiskers), dtype=tf.float32)
        
        # Use only whiskers 0, 2, 4 (skip 1, 3, 5)
        # This creates classes 0 (background), 1 (whisker 0), 3 (whisker 2), 5 (whisker 4)
        
        # Whisker 0 -> class 1
        labels = tf.tensor_scatter_nd_update(labels,
                                           [[0, i, 0, 0] for i in range(3)],
                                           [1.0] * 3)
        
        # Whisker 2 -> class 3
        labels = tf.tensor_scatter_nd_update(labels,
                                           [[0, i, 2, 2] for i in range(3)],
                                           [1.0] * 3)
        
        # Whisker 4 -> class 5
        labels = tf.tensor_scatter_nd_update(labels,
                                           [[0, i, 4, 4] for i in range(3)],
                                           [1.0] * 3)
        
        config = PixelTripletConfig(background_pixels=10, whisker_pixels=5)
        loss_fn = PixelTripletLoss(config=config)
        
        loss = loss_fn(labels, embeddings)
        
        # Should handle non-contiguous class IDs
        assert loss.shape == ()
        assert not tf.math.is_nan(loss)
        assert not tf.math.is_inf(loss)

    def test_many_classes_scenario(self, keras_float32_policy):
        """Test with many classes (10+ whiskers)."""
        batch_size = 1
        h, w = 20, 20
        feature_dim = 8
        num_whiskers = 12
        
        embeddings = tf.random.normal((batch_size, h, w, feature_dim))
        labels = tf.zeros((batch_size, h, w, num_whiskers), dtype=tf.float32)
        
        # Create a grid pattern with different whiskers
        for whisker_id in range(num_whiskers):
            # Each whisker gets a 2x2 region
            start_i = (whisker_id // 6) * 3
            start_j = (whisker_id % 6) * 3
            
            if start_i + 2 < h and start_j + 2 < w:
                labels = tf.tensor_scatter_nd_update(
                    labels,
                    [[0, start_i + di, start_j + dj, whisker_id] 
                     for di in range(2) for dj in range(2)],
                    [1.0] * 4
                )
        
        config = PixelTripletConfig(
            background_pixels=50,
            whisker_pixels=10,
            margin=2.0  # Larger margin for many classes
        )
        loss_fn = PixelTripletLoss(config=config)
        
        loss = loss_fn(labels, embeddings)
        
        # Should handle many classes
        assert loss.shape == ()
        assert not tf.math.is_nan(loss)
        assert not tf.math.is_inf(loss)

    def test_triplet_formation_with_multiple_classes(self, keras_float32_policy):
        """Test that valid triplets can be formed with multiple classes."""
        # Create a controlled scenario with known class distributions
        batch_size = 1
        h, w = 6, 6
        feature_dim = 4
        num_whiskers = 3
        
        # Create embeddings with clear separations
        embeddings = tf.zeros((batch_size, h, w, feature_dim), dtype=tf.float32)
        
        # Class 1 (whisker 0): embeddings = [1, 0, 0, 0]
        embeddings = tf.tensor_scatter_nd_update(
            embeddings,
            [[0, 0, j, 0] for j in range(2)],  # Top row, left 2 pixels
            [1.0] * 2
        )
        
        # Class 2 (whisker 1): embeddings = [0, 1, 0, 0]
        embeddings = tf.tensor_scatter_nd_update(
            embeddings,
            [[0, 1, j, 1] for j in range(2)],  # Second row, left 2 pixels
            [1.0] * 2
        )
        
        # Class 3 (whisker 2): embeddings = [0, 0, 1, 0]
        embeddings = tf.tensor_scatter_nd_update(
            embeddings,
            [[0, 2, j, 2] for j in range(2)],  # Third row, left 2 pixels
            [1.0] * 2
        )
        
        # Background: embeddings = [0, 0, 0, 0] (default)
        
        # Create corresponding labels
        labels = tf.zeros((batch_size, h, w, num_whiskers), dtype=tf.float32)
        
        # Whisker 0
        labels = tf.tensor_scatter_nd_update(labels,
                                           [[0, 0, j, 0] for j in range(2)],
                                           [1.0] * 2)
        
        # Whisker 1
        labels = tf.tensor_scatter_nd_update(labels,
                                           [[0, 1, j, 1] for j in range(2)],
                                           [1.0] * 2)
        
        # Whisker 2
        labels = tf.tensor_scatter_nd_update(labels,
                                           [[0, 2, j, 2] for j in range(2)],
                                           [1.0] * 2)
        
        config = PixelTripletConfig(
            background_pixels=10,
            whisker_pixels=5,
            margin=0.5,
            triplet_strategy="hard"
        )
        loss_fn = PixelTripletLoss(config=config)
        
        # Test that the sampling captures all classes
        y_true_resized = loss_fn._resize_labels(labels, h, w)
        class_labels = loss_fn._labels_to_classes(y_true_resized)
        
        sampled_embeddings, sampled_labels = loss_fn._sample_pixels_per_class_simple(
            embeddings, class_labels
        )
        
        unique_classes = tf.unique(sampled_labels)[0]
        
        # Should sample from multiple classes including background
        assert len(unique_classes) >= 3  # Background + at least 2 whiskers
        assert 0 in unique_classes.numpy()  # Background
        
        # Compute loss
        loss = loss_fn(labels, embeddings)
        
        # With well-separated embeddings, loss should be reasonable
        assert loss.shape == ()
        assert not tf.math.is_nan(loss)
        assert not tf.math.is_inf(loss)
        assert loss.numpy() >= 0.0

    def test_different_strategies_multiclass(self, keras_float32_policy, multiclass_segmentation_data):
        """Test different triplet strategies with multiple classes."""
        embeddings, labels = multiclass_segmentation_data
        
        strategies = ["hard", "semi_hard", "all"]
        
        for strategy in strategies:
            config = PixelTripletConfig(
                background_pixels=15,
                whisker_pixels=8,
                triplet_strategy=strategy,
                margin=1.5
            )
            loss_fn = PixelTripletLoss(config=config)
            
            loss = loss_fn(labels, embeddings)
            
            # All strategies should work with multiple classes
            assert loss.shape == ()
            assert not tf.math.is_nan(loss), f"NaN loss with strategy {strategy}"
            assert not tf.math.is_inf(loss), f"Inf loss with strategy {strategy}"
            assert loss.numpy() >= 0.0, f"Negative loss with strategy {strategy}"

    def test_sampling_fairness_multiclass(self, keras_float32_policy):
        """Test that sampling is fair across multiple classes."""
        batch_size = 1
        h, w = 12, 12
        feature_dim = 3
        num_whiskers = 4
        
        embeddings = tf.random.normal((batch_size, h, w, feature_dim))
        labels = tf.zeros((batch_size, h, w, num_whiskers), dtype=tf.float32)
        
        # Create equal-sized regions for each whisker (3x3 each)
        for whisker_id in range(num_whiskers):
            start_i = (whisker_id // 2) * 6
            start_j = (whisker_id % 2) * 6
            
            labels = tf.tensor_scatter_nd_update(
                labels,
                [[0, start_i + di, start_j + dj, whisker_id] 
                 for di in range(3) for dj in range(3)],
                [1.0] * 9
            )
        
        config = PixelTripletConfig(
            background_pixels=20,
            whisker_pixels=5,  # Less than available (9) for each class
            margin=1.0
        )
        loss_fn = PixelTripletLoss(config=config)
        
        # Run multiple times to check sampling consistency
        class_counts = {}
        
        for _ in range(10):
            y_true_resized = loss_fn._resize_labels(labels, h, w)
            class_labels = loss_fn._labels_to_classes(y_true_resized)
            
            sampled_embeddings, sampled_labels = loss_fn._sample_pixels_per_class_simple(
                embeddings, class_labels
            )
            
            # Count samples per class
            for class_id in tf.unique(sampled_labels)[0].numpy():
                if class_id not in class_counts:
                    class_counts[class_id] = 0
                class_counts[class_id] += tf.reduce_sum(
                    tf.cast(tf.equal(sampled_labels, class_id), tf.int32)
                ).numpy()
        
        # Should have sampled from all classes
        assert 0 in class_counts  # Background
        whisker_classes = [k for k in class_counts.keys() if k > 0]
        assert len(whisker_classes) >= 3  # Multiple whisker classes
        
        # Sampling should be roughly balanced for whisker classes
        # (background might be more due to higher pixel count)
        whisker_counts = [class_counts[k] for k in whisker_classes]
        if len(whisker_counts) > 1:
            # Standard deviation should be reasonable relative to mean
            mean_count = np.mean(whisker_counts)
            std_count = np.std(whisker_counts)
            assert std_count / mean_count < 1.0  # Coefficient of variation < 100%

    def test_large_margin_multiclass(self, keras_float32_policy, multiclass_segmentation_data):
        """Test with very large margin that might make all triplets 'easy'."""
        embeddings, labels = multiclass_segmentation_data
        
        config = PixelTripletConfig(
            background_pixels=10,
            whisker_pixels=5,
            margin=100.0,  # Very large margin
            triplet_strategy="hard"
        )
        loss_fn = PixelTripletLoss(config=config)
        
        loss = loss_fn(labels, embeddings)
        
        # Large margin should result in higher loss but still be valid
        assert loss.shape == ()
        assert not tf.math.is_nan(loss)
        assert not tf.math.is_inf(loss)
        assert loss.numpy() >= 0.0
        # With large margin, loss will likely be > 0
        assert loss.numpy() > 50.0  # Should be substantial with margin=100

    def test_loss_scaling_with_class_count(self, keras_float32_policy):
        """Test how loss values change as the number of classes increases."""
        batch_size = 1
        h, w = 12, 12
        feature_dim = 4
        
        loss_values = []
        class_counts = [2, 4, 6, 8, 12]  # Different numbers of whisker classes
        
        for num_classes in class_counts:
            embeddings = tf.random.normal((batch_size, h, w, feature_dim))
            labels = tf.zeros((batch_size, h, w, num_classes), dtype=tf.float32)
            
            # Create distinct spatial regions for each class
            pixels_per_class = (h * w) // (num_classes + 4)  # Leave room for background
            
            for class_id in range(num_classes):
                # Create a small square region for each class
                start_i = (class_id // 4) * 3
                start_j = (class_id % 4) * 3
                
                if start_i + 2 < h and start_j + 2 < w:
                    labels = tf.tensor_scatter_nd_update(
                        labels,
                        [[0, start_i + di, start_j + dj, class_id] 
                         for di in range(2) for dj in range(2)],
                        [1.0] * 4
                    )
            
            config = PixelTripletConfig(
                background_pixels=20,
                whisker_pixels=8,
                margin=1.0,
                triplet_strategy="hard"
            )
            loss_fn = PixelTripletLoss(config=config)
            
            loss = loss_fn(labels, embeddings)
            loss_values.append(loss.numpy())
            
            # Each loss should be valid
            assert not tf.math.is_nan(loss)
            assert not tf.math.is_inf(loss)
            assert loss.numpy() >= 0.0
        
        # Print loss scaling for analysis
        print(f"\nLoss scaling with class count:")
        for i, (num_classes, loss_val) in enumerate(zip(class_counts, loss_values)):
            print(f"  {num_classes} classes: loss = {loss_val:.4f}")
        
        # With more classes, we generally expect:
        # - More negative examples available for each anchor
        # - Potentially better triplet formation
        # - Loss values should remain stable (not explode or vanish)
        
        # Check that losses are in a reasonable range
        assert all(0.0 <= loss_val <= 1000.0 for loss_val in loss_values), "Loss values out of reasonable range"
        
        # Loss should not monotonically increase dramatically with more classes
        max_loss = max(loss_values)
        min_loss = min(loss_values)
        ratio = max_loss / (min_loss + 1e-8)
        assert ratio < 100.0, f"Loss range too large: {ratio:.2f}x variation"


class TestMultiClassEdgeCases:
    """Test edge cases specific to multiclass scenarios."""

    def test_overlapping_labels_handling(self, keras_float32_policy):
        """Test what happens when labels overlap (multiple whiskers per pixel)."""
        batch_size = 1
        h, w = 4, 4
        feature_dim = 3
        num_whiskers = 3
        
        embeddings = tf.random.normal((batch_size, h, w, feature_dim))
        labels = tf.zeros((batch_size, h, w, num_whiskers), dtype=tf.float32)
        
        # Create overlapping labels (pixel has multiple whiskers)
        # This shouldn't happen in real whisker data, but we should handle it gracefully
        
        # Pixel (1,1) has both whisker 0 and whisker 1
        labels = tf.tensor_scatter_nd_update(labels,
                                           [[0, 1, 1, 0], [0, 1, 1, 1]],
                                           [1.0, 1.0])
        
        # Pixel (2,2) has all three whiskers
        labels = tf.tensor_scatter_nd_update(labels,
                                           [[0, 2, 2, 0], [0, 2, 2, 1], [0, 2, 2, 2]],
                                           [1.0, 1.0, 1.0])
        
        # Some normal pixels
        labels = tf.tensor_scatter_nd_update(labels,
                                           [[0, 0, 0, 0], [0, 3, 3, 1]],
                                           [1.0, 1.0])
        
        config = PixelTripletConfig(background_pixels=5, whisker_pixels=5)
        loss_fn = PixelTripletLoss(config=config)
        
        # Should handle overlapping labels (argmax will pick one)
        loss = loss_fn(labels, embeddings)
        
        assert loss.shape == ()
        assert not tf.math.is_nan(loss)
        assert not tf.math.is_inf(loss)

    def test_single_pixel_classes(self, keras_float32_policy):
        """Test with classes that have only single pixels."""
        batch_size = 1
        h, w = 6, 6
        feature_dim = 4
        num_whiskers = 5
        
        embeddings = tf.random.normal((batch_size, h, w, feature_dim))
        labels = tf.zeros((batch_size, h, w, num_whiskers), dtype=tf.float32)
        
        # Each whisker has only 1 pixel
        for whisker_id in range(num_whiskers):
            labels = tf.tensor_scatter_nd_update(labels,
                                               [[0, whisker_id, whisker_id, whisker_id]],
                                               [1.0])
        
        config = PixelTripletConfig(
            background_pixels=15,
            whisker_pixels=2,  # More than available for most classes
            margin=1.0
        )
        loss_fn = PixelTripletLoss(config=config)
        
        loss = loss_fn(labels, embeddings)
        
        # Should handle single-pixel classes
        assert loss.shape == ()
        assert not tf.math.is_nan(loss)
        assert not tf.math.is_inf(loss)

    def test_no_background_pixels(self, keras_float32_policy):
        """Test when every pixel belongs to some whisker (no background)."""
        batch_size = 1
        h, w = 4, 4
        feature_dim = 3
        num_whiskers = 4
        
        embeddings = tf.random.normal((batch_size, h, w, feature_dim))
        labels = tf.zeros((batch_size, h, w, num_whiskers), dtype=tf.float32)
        
        # Fill entire image with whiskers (no background)
        for i in range(h):
            for j in range(w):
                whisker_id = (i * w + j) % num_whiskers
                labels = tf.tensor_scatter_nd_update(labels,
                                                   [[0, i, j, whisker_id]],
                                                   [1.0])
        
        config = PixelTripletConfig(
            background_pixels=10,  # Won't find any
            whisker_pixels=5,
            margin=1.0
        )
        loss_fn = PixelTripletLoss(config=config)
        
        loss = loss_fn(labels, embeddings)
        
        # Should handle no background case
        assert loss.shape == ()
        assert not tf.math.is_nan(loss)
        assert not tf.math.is_inf(loss)

    def test_extremely_large_margin(self, keras_float32_policy):
        """Test with very large margin that might cause numerical issues."""
        batch_size = 1
        h, w = 6, 6
        feature_dim = 3
        num_whiskers = 3
        
        # Create simple test data
        embeddings = tf.random.normal((batch_size, h, w, feature_dim))
        labels = tf.zeros((batch_size, h, w, num_whiskers), dtype=tf.float32)
        
        # Add some whisker pixels
        for whisker_id in range(num_whiskers):
            labels = tf.tensor_scatter_nd_update(labels,
                                               [[0, whisker_id, whisker_id, whisker_id]],
                                               [1.0])
        
        config = PixelTripletConfig(
            background_pixels=10,
            whisker_pixels=5,
            margin=1000.0,  # Extremely large margin
            triplet_strategy="hard"
        )
        loss_fn = PixelTripletLoss(config=config)
        
        loss = loss_fn(labels, embeddings)
        
        # Should still be valid even with extreme margin
        assert loss.shape == ()
        assert not tf.math.is_nan(loss)
        assert not tf.math.is_inf(loss)
        assert loss.numpy() >= 0.0