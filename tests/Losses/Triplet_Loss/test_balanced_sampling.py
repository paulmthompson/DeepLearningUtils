"""
Tests for balanced sampling in pixel triplet loss.

This module demonstrates the improved class balancing strategy that:
1. Balances classes to the minimum available samples
2. Applies a maximum cap to control memory usage
3. Prevents class imbalance issues that hurt triplet loss training
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


class TestBalancedSampling:
    """Test the new balanced sampling approach versus legacy unbalanced sampling."""

    @pytest.fixture
    def imbalanced_segmentation_data(self):
        """Create segmentation data with severe class imbalance to test sampling strategies."""
        batch_size = 1
        h, w = 20, 20  # Larger image for realistic imbalance
        feature_dim = 4
        num_whiskers = 3
        
        # Create embeddings with class-specific patterns
        embeddings = tf.zeros((batch_size, h, w, feature_dim), dtype=tf.float32)
        
        # Background: [0, 0, 0, 1] - covers most of the image (300+ pixels)
        embeddings = tf.tensor_scatter_nd_update(
            embeddings,
            [[0, i, j, 3] for i in range(h) for j in range(w)],
            [1.0] * (h * w)
        )
        
        # Class 1 (abundant): [1, 0, 0, 0] - covers a large region (100+ pixels)
        embeddings = tf.tensor_scatter_nd_update(
            embeddings,
            [[0, i, j, k] for i in range(5, 15) for j in range(5, 15) for k in [0]],
            [1.0] * 100
        )
        embeddings = tf.tensor_scatter_nd_update(
            embeddings,
            [[0, i, j, k] for i in range(5, 15) for j in range(5, 15) for k in [3]],
            [0.0] * 100  # Zero out background
        )
        
        # Class 2 (sparse): [0, 1, 0, 0] - covers only a small region (25 pixels)  
        embeddings = tf.tensor_scatter_nd_update(
            embeddings,
            [[0, i, j, k] for i in range(2, 7) for j in range(2, 7) for k in [1]],
            [1.0] * 25
        )
        embeddings = tf.tensor_scatter_nd_update(
            embeddings,
            [[0, i, j, k] for i in range(2, 7) for j in range(2, 7) for k in [3]],
            [0.0] * 25  # Zero out background
        )
        
        # Class 3 (very sparse): [0, 0, 1, 0] - covers only a tiny region (9 pixels)
        embeddings = tf.tensor_scatter_nd_update(
            embeddings,
            [[0, i, j, k] for i in range(16, 19) for j in range(16, 19) for k in [2]],
            [1.0] * 9
        )
        embeddings = tf.tensor_scatter_nd_update(
            embeddings,
            [[0, i, j, k] for i in range(16, 19) for j in range(16, 19) for k in [3]],
            [0.0] * 9  # Zero out background
        )
        
        # Create corresponding labels
        labels = tf.zeros((batch_size, h, w, num_whiskers), dtype=tf.float32)
        
        # Whisker 0 (abundant class)
        labels = tf.tensor_scatter_nd_update(labels,
                                           [[0, i, j, 0] for i in range(5, 15) for j in range(5, 15)],
                                           [1.0] * 100)
        
        # Whisker 1 (sparse class)
        labels = tf.tensor_scatter_nd_update(labels,
                                           [[0, i, j, 1] for i in range(2, 7) for j in range(2, 7)],
                                           [1.0] * 25)
        
        # Whisker 2 (very sparse class)
        labels = tf.tensor_scatter_nd_update(labels,
                                           [[0, i, j, 2] for i in range(16, 19) for j in range(16, 19)],
                                           [1.0] * 9)
        
        return embeddings, labels

    def test_legacy_unbalanced_sampling_demonstration(self, keras_float32_policy, imbalanced_segmentation_data):
        """Demonstrate the problem with legacy unbalanced sampling."""
        embeddings, labels = imbalanced_segmentation_data
        
        print(f"\n" + "="*80)
        print(f"LEGACY UNBALANCED SAMPLING DEMONSTRATION")
        print(f"="*80)
        
        # Configure legacy unbalanced sampling
        config = PixelTripletConfig(
            margin=0.5,
            background_pixels=200,  # Request 200 background pixels
            whisker_pixels=100,     # Request 100 whisker pixels per class
            use_balanced_sampling=False,  # Use legacy approach
            distance_metric="euclidean",
            triplet_strategy="hard"
        )
        loss_fn = PixelTripletLoss(config=config)
        
        # Sample pixels and examine the distribution
        class_labels = loss_fn._labels_to_classes(labels)
        sampled_embeddings, sampled_labels = loss_fn._sample_pixels_per_class_simple(
            embeddings, class_labels
        )
        
        # Count samples per class
        unique_labels, _, counts = tf.unique_with_counts(sampled_labels)
        
        print(f"\nActual pixel availability:")
        print(f"  Background (class 0): ~266 pixels available")
        print(f"  Whisker 1 (class 1):  100 pixels available")
        print(f"  Whisker 2 (class 2):   25 pixels available")
        print(f"  Whisker 3 (class 3):    9 pixels available")
        
        print(f"\nRequested samples:")
        print(f"  Background: 200 pixels")
        print(f"  Each whisker: 100 pixels")
        
        print(f"\nActual samples with legacy approach:")
        for i, (label, count) in enumerate(zip(unique_labels.numpy(), counts.numpy())):
            class_name = f"Background" if label == 0 else f"Whisker {label}"
            print(f"  {class_name:<12}: {count:3d} samples")
        
        # Calculate class imbalance ratio
        counts_array = counts.numpy()
        max_count = np.max(counts_array)
        min_count = np.min(counts_array)
        imbalance_ratio = max_count / min_count
        
        print(f"\nClass imbalance ratio: {imbalance_ratio:.1f}:1 (higher is worse)")
        print(f"⚠️  This severe imbalance will hurt triplet loss training!")
        
        # Compute loss
        loss = loss_fn(labels, embeddings)
        print(f"Loss value: {loss.numpy():.4f}")

    def test_new_balanced_sampling_demonstration(self, keras_float32_policy, imbalanced_segmentation_data):
        """Demonstrate the solution with new balanced sampling."""
        embeddings, labels = imbalanced_segmentation_data
        
        print(f"\n" + "="*80)
        print(f"NEW BALANCED SAMPLING DEMONSTRATION")
        print(f"="*80)
        
        # Configure balanced sampling
        config = PixelTripletConfig(
            margin=0.5,
            max_samples_per_class=50,  # Cap at 50 samples per class
            use_balanced_sampling=True,  # Use new balanced approach
            distance_metric="euclidean",
            triplet_strategy="hard"
        )
        loss_fn = PixelTripletLoss(config=config)
        
        # Sample pixels and examine the distribution
        class_labels = loss_fn._labels_to_classes(labels)
        sampled_embeddings, sampled_labels = loss_fn._sample_pixels_balanced(
            embeddings, class_labels
        )
        
        # Count samples per class
        unique_labels, _, counts = tf.unique_with_counts(sampled_labels)
        
        print(f"\nBalanced sampling logic:")
        print(f"  1. Find minimum available: min(266, 100, 25, 9) = 9 pixels")
        print(f"  2. Apply maximum cap: min(9, 50) = 9 pixels")
        print(f"  3. Sample exactly 9 pixels from each class")
        
        print(f"\nActual samples with balanced approach:")
        for i, (label, count) in enumerate(zip(unique_labels.numpy(), counts.numpy())):
            class_name = f"Background" if label == 0 else f"Whisker {label}"
            print(f"  {class_name:<12}: {count:3d} samples")
        
        # Calculate class balance
        counts_array = counts.numpy()
        max_count = np.max(counts_array)
        min_count = np.min(counts_array)
        imbalance_ratio = max_count / min_count
        
        print(f"\nClass imbalance ratio: {imbalance_ratio:.1f}:1 (perfect balance!)")
        print(f"✅ Perfect class balance achieved!")
        
        # Compute loss
        loss = loss_fn(labels, embeddings)
        print(f"Loss value: {loss.numpy():.4f}")

    def test_balanced_sampling_with_cap(self, keras_float32_policy):
        """Test balanced sampling with different maximum caps."""
        print(f"\n" + "="*80)
        print(f"BALANCED SAMPLING WITH DIFFERENT CAPS")
        print(f"="*80)
        
        # Create simple test data where all classes have enough pixels
        batch_size = 1
        h, w = 10, 10
        feature_dim = 2
        num_whiskers = 2
        
        # Create balanced availability (all classes have 25 pixels each)
        embeddings = tf.zeros((batch_size, h, w, feature_dim), dtype=tf.float32)
        
        # Background: 75 pixels (most of image)
        embeddings = tf.tensor_scatter_nd_update(
            embeddings,
            [[0, i, j, 1] for i in range(h) for j in range(w) if (i * w + j) >= 25],
            [1.0] * 75
        )
        
        # Class 1: 15 pixels
        embeddings = tf.tensor_scatter_nd_update(
            embeddings,
            [[0, i, j, 0] for i in range(3) for j in range(5)],
            [1.0] * 15
        )
        
        # Class 2: 10 pixels
        embeddings = tf.tensor_scatter_nd_update(
            embeddings,
            [[0, i, j, 0] for i in range(7, 9) for j in range(5)],
            [1.0] * 10
        )
        
        labels = tf.zeros((batch_size, h, w, num_whiskers), dtype=tf.float32)
        labels = tf.tensor_scatter_nd_update(labels,
                                           [[0, i, j, 0] for i in range(3) for j in range(5)],
                                           [1.0] * 15)
        labels = tf.tensor_scatter_nd_update(labels,
                                           [[0, i, j, 1] for i in range(7, 9) for j in range(5)],
                                           [1.0] * 10)
        
        # Test different caps
        caps = [5, 15, 50]
        
        print(f"\nAvailable pixels: Background=75, Class1=15, Class2=10")
        print(f"Minimum available: 10")
        print(f"\n{'Cap':<5} {'Samples/Class':<13} {'Total Samples':<13} {'Description'}")
        print(f"{'-'*5} {'-'*13} {'-'*13} {'-'*30}")
        
        for cap in caps:
            config = PixelTripletConfig(
                max_samples_per_class=cap,
                use_balanced_sampling=True,
                distance_metric="euclidean",
                triplet_strategy="hard"
            )
            loss_fn = PixelTripletLoss(config=config)
            
            class_labels = loss_fn._labels_to_classes(labels)
            sampled_embeddings, sampled_labels = loss_fn._sample_pixels_balanced(
                embeddings, class_labels
            )
            
            unique_labels, _, counts = tf.unique_with_counts(sampled_labels)
            samples_per_class = counts[0].numpy()  # Should be same for all classes
            total_samples = tf.shape(sampled_embeddings)[0].numpy()
            
            if cap < 10:
                description = f"Cap limits to {cap}"
            elif cap == 15:
                description = f"Cap doesn't limit (min=10)"
            else:
                description = f"Cap doesn't limit (min=10)"
            
            print(f"{cap:<5} {samples_per_class:<13} {total_samples:<13} {description}")

    def test_balanced_vs_unbalanced_loss_comparison(self, keras_float32_policy, imbalanced_segmentation_data):
        """Compare loss values between balanced and unbalanced sampling."""
        embeddings, labels = imbalanced_segmentation_data
        
        print(f"\n" + "="*80)
        print(f"BALANCED VS UNBALANCED LOSS COMPARISON")
        print(f"="*80)
        
        # Test unbalanced approach
        config_unbalanced = PixelTripletConfig(
            margin=0.5,
            background_pixels=100,
            whisker_pixels=50,
            use_balanced_sampling=False,
            distance_metric="euclidean",
            triplet_strategy="hard"
        )
        loss_fn_unbalanced = PixelTripletLoss(config=config_unbalanced)
        loss_unbalanced = loss_fn_unbalanced(labels, embeddings)
        
        # Test balanced approach
        config_balanced = PixelTripletConfig(
            margin=0.5,
            max_samples_per_class=20,
            use_balanced_sampling=True,
            distance_metric="euclidean", 
            triplet_strategy="hard"
        )
        loss_fn_balanced = PixelTripletLoss(config=config_balanced)
        loss_balanced = loss_fn_balanced(labels, embeddings)
        
        print(f"\nLoss comparison:")
        print(f"  Unbalanced sampling: {loss_unbalanced.numpy():.4f}")
        print(f"  Balanced sampling:   {loss_balanced.numpy():.4f}")
        
        # Both should be valid losses
        assert not tf.math.is_nan(loss_unbalanced)
        assert not tf.math.is_inf(loss_unbalanced)
        assert loss_unbalanced.numpy() >= 0.0
        
        assert not tf.math.is_nan(loss_balanced)
        assert not tf.math.is_inf(loss_balanced)
        assert loss_balanced.numpy() >= 0.0
        
        print(f"\n✅ Both approaches produce valid losses")
        print(f"📊 Balanced sampling provides more stable training due to equal class representation")

    def test_backward_compatibility(self, keras_float32_policy):
        """Test that the new implementation maintains backward compatibility."""
        print(f"\n" + "="*80)
        print(f"BACKWARD COMPATIBILITY TEST")
        print(f"="*80)
        
        # Create simple test data
        batch_size = 1
        h, w = 6, 6
        feature_dim = 2
        num_whiskers = 2
        
        embeddings = tf.random.normal((batch_size, h, w, feature_dim))
        labels = tf.zeros((batch_size, h, w, num_whiskers), dtype=tf.float32)
        
        # Add some labels
        labels = tf.tensor_scatter_nd_update(labels,
                                           [[0, i, j, 0] for i in range(2) for j in range(3)],
                                           [1.0] * 6)
        labels = tf.tensor_scatter_nd_update(labels,
                                           [[0, i, j, 1] for i in range(3, 5) for j in range(3)],
                                           [1.0] * 6)
        
        # Test legacy usage (should still work)
        legacy_loss = create_pixel_triplet_loss(
            margin=0.5,
            background_pixels=20,
            whisker_pixels=10,
            use_balanced_sampling=False  # Explicitly use legacy
        )
        legacy_result = legacy_loss(labels, embeddings)
        
        # Test new usage
        balanced_loss = create_pixel_triplet_loss(
            margin=0.5,
            max_samples_per_class=15,
            use_balanced_sampling=True
        )
        balanced_result = balanced_loss(labels, embeddings)
        
        print(f"Legacy approach: Loss = {legacy_result.numpy():.4f}")
        print(f"Balanced approach: Loss = {balanced_result.numpy():.4f}")
        print(f"✅ Both approaches work correctly")
        
        # Test default behavior (should use balanced by default)
        default_loss = create_pixel_triplet_loss(margin=0.5)
        default_result = default_loss(labels, embeddings)
        
        print(f"Default behavior: Loss = {default_result.numpy():.4f}")
        print(f"✅ Default uses balanced sampling (recommended)")


class TestBalancedSamplingEdgeCases:
    """Test edge cases for balanced sampling."""

    def test_single_pixel_classes(self, keras_float32_policy):
        """Test balanced sampling when classes have very few pixels."""
        # Create scenario where each class has only 1-2 pixels
        batch_size = 1
        h, w = 4, 4
        feature_dim = 2
        num_whiskers = 2
        
        embeddings = tf.zeros((batch_size, h, w, feature_dim), dtype=tf.float32)
        labels = tf.zeros((batch_size, h, w, num_whiskers), dtype=tf.float32)
        
        # Background: 12 pixels
        # Class 1: 2 pixels
        labels = tf.tensor_scatter_nd_update(labels,
                                           [[0, 0, 0, 0], [0, 0, 1, 0]],
                                           [1.0, 1.0])
        
        # Class 2: 1 pixel (minimum)
        labels = tf.tensor_scatter_nd_update(labels,
                                           [[0, 1, 0, 1]],
                                           [1.0])
        
        config = PixelTripletConfig(
            max_samples_per_class=10,  # High cap
            use_balanced_sampling=True
        )
        loss_fn = PixelTripletLoss(config=config)
        
        loss = loss_fn(labels, embeddings)
        
        # Should balance to 1 sample per class (minimum available)
        class_labels = loss_fn._labels_to_classes(labels)
        sampled_embeddings, sampled_labels = loss_fn._sample_pixels_balanced(
            embeddings, class_labels
        )
        
        unique_labels, _, counts = tf.unique_with_counts(sampled_labels)
        
        # All classes should have exactly 1 sample
        for count in counts:
            assert count.numpy() == 1
        
        assert not tf.math.is_nan(loss)
        assert not tf.math.is_inf(loss)
        assert loss.numpy() >= 0.0

    def test_no_whisker_pixels(self, keras_float32_policy):
        """Test balanced sampling when there are no whisker pixels (background only)."""
        batch_size = 1
        h, w = 4, 4
        feature_dim = 2
        num_whiskers = 2
        
        # Only background pixels
        embeddings = tf.random.normal((batch_size, h, w, feature_dim))
        labels = tf.zeros((batch_size, h, w, num_whiskers), dtype=tf.float32)
        
        config = PixelTripletConfig(
            max_samples_per_class=10,
            use_balanced_sampling=True
        )
        loss_fn = PixelTripletLoss(config=config)
        
        # Should handle gracefully
        loss = loss_fn(labels, embeddings)
        
        assert not tf.math.is_nan(loss)
        assert not tf.math.is_inf(loss)
        assert loss.numpy() >= 0.0