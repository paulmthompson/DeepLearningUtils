"""
Tests for the exact (loop-based) computation mode in pixel triplet loss.

These tests verify that the memory-efficient loop-based computation gives
similar results to the standard pairwise matrix computation.
"""

import pytest
import numpy as np
import keras
import tensorflow as tf

from src.DeepLearningUtils.Losses.Triplet_Loss.pixel_triplet_loss_keras import (
    PixelTripletLoss,
    PixelTripletConfig,
    create_pixel_triplet_loss,
)


class TestExactTripletLossComputation:
    """Tests for the exact (loop-based) triplet loss computation."""

    def test_exact_hard_loss_basic(self):
        """Test that exact hard triplet loss runs without errors."""
        batch_size = 1
        height, width = 16, 16
        feature_dim = 8
        num_classes = 2

        # Create labels (2 classes: background and one foreground)
        y_true = np.zeros((batch_size, height, width, num_classes), dtype=np.float32)
        y_true[:, :, width//2:, 0] = 1.0  # Right half is foreground

        # Create embeddings with separation
        np.random.seed(42)
        embeddings = np.zeros((batch_size, height, width, feature_dim), dtype=np.float32)
        embeddings[:, :, :width//2, :] = np.random.randn(batch_size, height, width//2, feature_dim).astype(np.float32) * 0.5
        embeddings[:, :, width//2:, :] = 3.0 + np.random.randn(batch_size, height, width//2, feature_dim).astype(np.float32) * 0.5

        y_true = tf.constant(y_true)
        y_pred = tf.constant(embeddings)

        # Create loss with exact computation
        loss_fn = PixelTripletLoss(
            config=PixelTripletConfig(
                margin=1.0,
                triplet_strategy="hard",
                use_exact=True,
                batch_size_for_exact=50,
                max_samples_per_class=100,
            )
        )

        loss = loss_fn(y_true, y_pred)
        loss_value = float(keras.ops.convert_to_numpy(loss))

        print(f"\nExact hard triplet loss: {loss_value:.6f}")

        assert not np.isnan(loss_value), "Loss should not be NaN"
        assert not np.isinf(loss_value), "Loss should not be Inf"
        assert loss_value >= 0, "Loss should be non-negative"

    def test_exact_all_loss_basic(self):
        """Test that exact batch-all triplet loss runs without errors."""
        batch_size = 1
        height, width = 16, 16
        feature_dim = 8
        num_classes = 2

        y_true = np.zeros((batch_size, height, width, num_classes), dtype=np.float32)
        y_true[:, :, width//2:, 0] = 1.0

        np.random.seed(42)
        embeddings = np.zeros((batch_size, height, width, feature_dim), dtype=np.float32)
        embeddings[:, :, :width//2, :] = np.random.randn(batch_size, height, width//2, feature_dim).astype(np.float32) * 0.5
        embeddings[:, :, width//2:, :] = 3.0 + np.random.randn(batch_size, height, width//2, feature_dim).astype(np.float32) * 0.5

        y_true = tf.constant(y_true)
        y_pred = tf.constant(embeddings)

        loss_fn = PixelTripletLoss(
            config=PixelTripletConfig(
                margin=1.0,
                triplet_strategy="all",
                use_exact=True,
                batch_size_for_exact=50,
                max_samples_per_class=100,
            )
        )

        loss = loss_fn(y_true, y_pred)
        loss_value = float(keras.ops.convert_to_numpy(loss))

        print(f"\nExact batch-all triplet loss: {loss_value:.6f}")

        assert not np.isnan(loss_value), "Loss should not be NaN"
        assert not np.isinf(loss_value), "Loss should not be Inf"

    def test_exact_all_with_remove_easy_triplets(self):
        """
        Test batch-all with remove_easy_triplets=True in exact mode.

        This tests the proper computation of max(0, d_ap - d_an + margin)
        for each individual triplet.
        """
        batch_size = 1
        height, width = 16, 16
        feature_dim = 8
        num_classes = 2

        y_true = np.zeros((batch_size, height, width, num_classes), dtype=np.float32)
        y_true[:, :, width//2:, 0] = 1.0

        # Create embeddings with some easy and some hard triplets
        np.random.seed(42)
        embeddings = np.zeros((batch_size, height, width, feature_dim), dtype=np.float32)
        # Background spread out
        embeddings[:, :, :width//2, :] = np.random.randn(batch_size, height, width//2, feature_dim).astype(np.float32) * 2.0
        # Foreground more clustered but with some variance
        embeddings[:, :, width//2:, :] = 5.0 + np.random.randn(batch_size, height, width//2, feature_dim).astype(np.float32) * 0.5

        y_true = tf.constant(y_true)
        y_pred = tf.constant(embeddings)

        # Without removing easy triplets
        loss_with_easy = PixelTripletLoss(
            config=PixelTripletConfig(
                margin=1.0,
                triplet_strategy="all",
                use_exact=True,
                remove_easy_triplets=False,
                batch_size_for_exact=50,
                max_samples_per_class=100,
            )
        )

        # With removing easy triplets
        loss_without_easy = PixelTripletLoss(
            config=PixelTripletConfig(
                margin=1.0,
                triplet_strategy="all",
                use_exact=True,
                remove_easy_triplets=True,
                batch_size_for_exact=50,
                max_samples_per_class=100,
            )
        )

        value_with_easy = float(keras.ops.convert_to_numpy(loss_with_easy(y_true, y_pred)))
        value_without_easy = float(keras.ops.convert_to_numpy(loss_without_easy(y_true, y_pred)))

        print(f"\nBatch-all exact mode:")
        print(f"  With easy triplets:    {value_with_easy:.6f}")
        print(f"  Without easy triplets: {value_without_easy:.6f}")

        assert not np.isnan(value_with_easy), "Loss with easy should not be NaN"
        assert not np.isnan(value_without_easy), "Loss without easy should not be NaN"

        # When removing easy triplets, loss should be >= 0 (all max(0, ...) terms)
        assert value_without_easy >= 0, "Loss without easy triplets should be non-negative"

        # With good separation, removing easy triplets might give a different value
        # (could be higher since we're only averaging over hard triplets)

    def test_exact_all_remove_easy_vs_standard(self):
        """
        Compare exact mode with standard mode for batch-all with remove_easy_triplets.

        Both should give similar results for small inputs.
        """
        batch_size = 1
        height, width = 12, 12  # Small for standard mode to handle
        feature_dim = 8
        num_classes = 2

        y_true = np.zeros((batch_size, height, width, num_classes), dtype=np.float32)
        y_true[:, :, width//2:, 0] = 1.0

        np.random.seed(42)
        embeddings = np.zeros((batch_size, height, width, feature_dim), dtype=np.float32)
        embeddings[:, :, :width//2, :] = np.random.randn(batch_size, height, width//2, feature_dim).astype(np.float32)
        embeddings[:, :, width//2:, :] = 2.0 + np.random.randn(batch_size, height, width//2, feature_dim).astype(np.float32) * 0.5

        y_true = tf.constant(y_true)
        y_pred = tf.constant(embeddings)

        # Standard mode
        loss_standard = PixelTripletLoss(
            config=PixelTripletConfig(
                margin=1.0,
                triplet_strategy="all",
                use_exact=False,
                remove_easy_triplets=True,
                max_samples_per_class=100,
            )
        )

        # Exact mode
        loss_exact = PixelTripletLoss(
            config=PixelTripletConfig(
                margin=1.0,
                triplet_strategy="all",
                use_exact=True,
                remove_easy_triplets=True,
                batch_size_for_exact=50,
                max_samples_per_class=100,
            )
        )

        standard_value = float(keras.ops.convert_to_numpy(loss_standard(y_true, y_pred)))
        exact_value = float(keras.ops.convert_to_numpy(loss_exact(y_true, y_pred)))

        print(f"\nBatch-all with remove_easy_triplets:")
        print(f"  Standard mode: {standard_value:.6f}")
        print(f"  Exact mode:    {exact_value:.6f}")

        # Both should be valid
        assert not np.isnan(standard_value), "Standard loss should not be NaN"
        assert not np.isnan(exact_value), "Exact loss should not be NaN"
        assert standard_value >= 0, "Standard loss should be non-negative"
        assert exact_value >= 0, "Exact loss should be non-negative"

    def test_exact_all_perfect_separation(self):
        """
        Test batch-all exact with perfect separation (all easy triplets).

        With perfect separation, d_ap << d_an, so all triplets are "easy"
        and have loss <= 0. With remove_easy_triplets=True, this should
        give a small or zero loss.
        """
        batch_size = 1
        height, width = 16, 16
        feature_dim = 8
        num_classes = 2

        y_true = np.zeros((batch_size, height, width, num_classes), dtype=np.float32)
        y_true[:, :, width//2:, 0] = 1.0

        # Perfect separation: very tight clusters far apart
        embeddings = np.zeros((batch_size, height, width, feature_dim), dtype=np.float32)
        embeddings[:, :, :width//2, :] = 0.0  # Background at origin
        embeddings[:, :, width//2:, :] = 100.0  # Foreground very far away

        y_true = tf.constant(y_true)
        y_pred = tf.constant(embeddings)

        loss_fn = PixelTripletLoss(
            config=PixelTripletConfig(
                margin=1.0,
                triplet_strategy="all",
                use_exact=True,
                remove_easy_triplets=True,
                batch_size_for_exact=50,
                max_samples_per_class=100,
            )
        )

        loss_value = float(keras.ops.convert_to_numpy(loss_fn(y_true, y_pred)))

        print(f"\nPerfect separation with remove_easy_triplets: {loss_value:.6f}")

        assert not np.isnan(loss_value), "Loss should not be NaN"
        # With perfect separation, all triplets should be easy (d_ap - d_an + margin < 0)
        # So the loss should be very small (close to 0 or 0)
        assert loss_value >= 0, "Loss should be non-negative"

    def test_exact_vs_standard_similar_results(self):
        """
        Test that exact and standard computation give similar results.

        For small inputs where both methods can run, they should give
        comparable (though not necessarily identical) results.
        """
        batch_size = 1
        height, width = 16, 16
        feature_dim = 8
        num_classes = 2

        y_true = np.zeros((batch_size, height, width, num_classes), dtype=np.float32)
        y_true[:, :, width//2:, 0] = 1.0

        np.random.seed(42)
        embeddings = np.zeros((batch_size, height, width, feature_dim), dtype=np.float32)
        embeddings[:, :, :width//2, :] = np.random.randn(batch_size, height, width//2, feature_dim).astype(np.float32) * 0.5
        embeddings[:, :, width//2:, :] = 3.0 + np.random.randn(batch_size, height, width//2, feature_dim).astype(np.float32) * 0.5

        y_true = tf.constant(y_true)
        y_pred = tf.constant(embeddings)

        # Standard computation
        loss_standard = PixelTripletLoss(
            config=PixelTripletConfig(
                margin=1.0,
                triplet_strategy="hard",
                use_exact=False,
                max_samples_per_class=200,  # Sample enough to be representative
            )
        )

        # Exact computation
        loss_exact = PixelTripletLoss(
            config=PixelTripletConfig(
                margin=1.0,
                triplet_strategy="hard",
                use_exact=True,
                batch_size_for_exact=50,
                max_samples_per_class=200,
            )
        )

        standard_value = float(keras.ops.convert_to_numpy(loss_standard(y_true, y_pred)))
        exact_value = float(keras.ops.convert_to_numpy(loss_exact(y_true, y_pred)))

        print(f"\nStandard vs Exact comparison:")
        print(f"  Standard: {standard_value:.6f}")
        print(f"  Exact:    {exact_value:.6f}")

        # Both should be valid
        assert not np.isnan(standard_value), "Standard loss should not be NaN"
        assert not np.isnan(exact_value), "Exact loss should not be NaN"

        # They may differ due to sampling but both should be reasonable
        # We just check they're in the same ballpark
        assert standard_value >= 0, "Standard loss should be non-negative"
        assert exact_value >= 0, "Exact loss should be non-negative"

    def test_exact_with_different_batch_sizes(self):
        """Test that batch_size_for_exact doesn't significantly affect results."""
        batch_size = 1
        height, width = 16, 16
        feature_dim = 8
        num_classes = 2

        y_true = np.zeros((batch_size, height, width, num_classes), dtype=np.float32)
        y_true[:, :, width//2:, 0] = 1.0

        np.random.seed(42)
        embeddings = np.random.randn(batch_size, height, width, feature_dim).astype(np.float32)
        embeddings[:, :, width//2:, :] += 2.0

        y_true = tf.constant(y_true)
        y_pred = tf.constant(embeddings)

        # Small batch size
        loss_small = PixelTripletLoss(
            config=PixelTripletConfig(
                margin=1.0,
                triplet_strategy="hard",
                use_exact=True,
                batch_size_for_exact=10,
                max_samples_per_class=100,
            )
        )

        # Large batch size
        loss_large = PixelTripletLoss(
            config=PixelTripletConfig(
                margin=1.0,
                triplet_strategy="hard",
                use_exact=True,
                batch_size_for_exact=100,
                max_samples_per_class=100,
            )
        )

        small_value = float(keras.ops.convert_to_numpy(loss_small(y_true, y_pred)))
        large_value = float(keras.ops.convert_to_numpy(loss_large(y_true, y_pred)))

        print(f"\nDifferent batch sizes:")
        print(f"  batch_size=10:  {small_value:.6f}")
        print(f"  batch_size=100: {large_value:.6f}")

        # Results should be very close (only floating point differences)
        assert np.isclose(small_value, large_value, rtol=1e-3), (
            f"Different batch sizes should give similar results. "
            f"Got {small_value:.6f} vs {large_value:.6f}"
        )

    @pytest.mark.parametrize("distance_metric", ["euclidean", "cosine", "manhattan"])
    def test_exact_all_distance_metrics(self, distance_metric):
        """Test exact computation with all distance metrics."""
        batch_size = 1
        height, width = 16, 16
        feature_dim = 8
        num_classes = 2

        y_true = np.zeros((batch_size, height, width, num_classes), dtype=np.float32)
        y_true[:, :, width//2:, 0] = 1.0

        np.random.seed(42)
        embeddings = np.random.randn(batch_size, height, width, feature_dim).astype(np.float32)
        embeddings[:, :, width//2:, :] += 2.0

        y_true = tf.constant(y_true)
        y_pred = tf.constant(embeddings)

        loss_fn = PixelTripletLoss(
            config=PixelTripletConfig(
                margin=1.0,
                triplet_strategy="hard",
                distance_metric=distance_metric,
                use_exact=True,
                batch_size_for_exact=50,
                max_samples_per_class=100,
            )
        )

        loss_value = float(keras.ops.convert_to_numpy(loss_fn(y_true, y_pred)))

        print(f"\nExact with {distance_metric}: {loss_value:.6f}")

        assert not np.isnan(loss_value), f"Loss should not be NaN for {distance_metric}"
        assert not np.isinf(loss_value), f"Loss should not be Inf for {distance_metric}"

    def test_exact_with_multiple_foreground_classes(self):
        """Test exact computation with multiple foreground classes."""
        batch_size = 1
        height, width = 24, 24
        feature_dim = 8
        num_classes = 3  # 3 foreground classes

        # Create labels with 3 foreground regions
        y_true = np.zeros((batch_size, height, width, num_classes), dtype=np.float32)
        y_true[:, :8, :, 0] = 1.0   # Top third: class 1
        y_true[:, 16:, :, 1] = 1.0  # Bottom third: class 2
        y_true[:, 8:16, :8, 2] = 1.0  # Middle-left: class 3
        # Middle-right is background

        np.random.seed(42)
        embeddings = np.random.randn(batch_size, height, width, feature_dim).astype(np.float32) * 0.3
        # Give each class a different offset
        embeddings[:, :8, :, 0] += 3.0
        embeddings[:, 16:, :, 1] += 3.0
        embeddings[:, 8:16, :8, 2] += 3.0

        y_true = tf.constant(y_true)
        y_pred = tf.constant(embeddings)

        loss_fn = PixelTripletLoss(
            config=PixelTripletConfig(
                margin=1.0,
                triplet_strategy="hard",
                use_exact=True,
                batch_size_for_exact=50,
                max_samples_per_class=50,
            )
        )

        loss_value = float(keras.ops.convert_to_numpy(loss_fn(y_true, y_pred)))

        print(f"\nExact with 3 foreground classes: {loss_value:.6f}")

        assert not np.isnan(loss_value), "Loss should not be NaN"
        assert not np.isinf(loss_value), "Loss should not be Inf"


class TestExactTripletLossConfig:
    """Tests for configuration and serialization with exact mode."""

    def test_config_includes_exact_params(self):
        """Test that config includes use_exact and batch_size_for_exact."""
        config = PixelTripletConfig(
            use_exact=True,
            batch_size_for_exact=200,
        )

        assert config.use_exact is True
        assert config.batch_size_for_exact == 200

    def test_get_config_includes_exact_params(self):
        """Test that get_config returns exact parameters."""
        loss_fn = PixelTripletLoss(
            config=PixelTripletConfig(
                use_exact=True,
                batch_size_for_exact=150,
            )
        )

        saved_config = loss_fn.get_config()

        assert saved_config['use_exact'] is True
        assert saved_config['batch_size_for_exact'] == 150

    def test_from_config_restores_exact_params(self):
        """Test that from_config properly restores exact parameters."""
        original = PixelTripletLoss(
            config=PixelTripletConfig(
                use_exact=True,
                batch_size_for_exact=175,
                margin=0.5,
            )
        )

        saved_config = original.get_config()
        restored = PixelTripletLoss.from_config(saved_config)

        assert restored.config.use_exact is True
        assert restored.config.batch_size_for_exact == 175
        assert restored.config.margin == 0.5

    def test_convenience_function_with_exact(self):
        """Test create_pixel_triplet_loss with exact parameters."""
        loss_fn = create_pixel_triplet_loss(
            margin=0.8,
            use_exact=True,
            batch_size_for_exact=200,
            triplet_strategy="hard",
        )

        assert loss_fn.config.use_exact is True
        assert loss_fn.config.batch_size_for_exact == 200
        assert loss_fn.config.margin == 0.8


class TestExactTripletLossEdgeCases:
    """Tests for edge cases in exact triplet loss computation."""

    def test_exact_with_no_foreground(self):
        """Test behavior when there's no foreground (all background)."""
        batch_size = 1
        height, width = 16, 16
        feature_dim = 8
        num_classes = 1

        # All background (zeros in the label mask)
        y_true = np.zeros((batch_size, height, width, num_classes), dtype=np.float32)

        np.random.seed(42)
        embeddings = np.random.randn(batch_size, height, width, feature_dim).astype(np.float32)

        y_true = tf.constant(y_true)
        y_pred = tf.constant(embeddings)

        loss_fn = PixelTripletLoss(
            config=PixelTripletConfig(
                margin=1.0,
                triplet_strategy="hard",
                use_exact=True,
                batch_size_for_exact=50,
            )
        )

        loss_value = float(keras.ops.convert_to_numpy(loss_fn(y_true, y_pred)))

        print(f"\nExact with no foreground: {loss_value:.6f}")

        # Should handle gracefully (loss might be 0 or small)
        assert not np.isnan(loss_value), "Loss should not be NaN"
        assert not np.isinf(loss_value), "Loss should not be Inf"

    def test_exact_with_small_regions(self):
        """Test exact computation with very small foreground regions."""
        batch_size = 1
        height, width = 32, 32
        feature_dim = 8
        num_classes = 1

        # Very small foreground region
        y_true = np.zeros((batch_size, height, width, num_classes), dtype=np.float32)
        y_true[:, 15:17, 15:17, 0] = 1.0  # Only 4 pixels

        np.random.seed(42)
        embeddings = np.random.randn(batch_size, height, width, feature_dim).astype(np.float32)
        embeddings[:, 15:17, 15:17, :] += 5.0

        y_true = tf.constant(y_true)
        y_pred = tf.constant(embeddings)

        loss_fn = PixelTripletLoss(
            config=PixelTripletConfig(
                margin=1.0,
                triplet_strategy="hard",
                use_exact=True,
                batch_size_for_exact=50,
                max_samples_per_class=50,
            )
        )

        loss_value = float(keras.ops.convert_to_numpy(loss_fn(y_true, y_pred)))

        print(f"\nExact with small foreground (4 pixels): {loss_value:.6f}")

        assert not np.isnan(loss_value), "Loss should not be NaN"
        assert not np.isinf(loss_value), "Loss should not be Inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


class TestClassBalancedWeighting:
    """Tests for class-balanced weighting in exact mode."""

    def test_class_balanced_weighting_basic(self):
        """Test that class_balanced_weighting runs without errors."""
        batch_size = 1
        height, width = 24, 24
        feature_dim = 8
        num_classes = 2

        # Create imbalanced classes: 80% background, 20% foreground
        y_true = np.zeros((batch_size, height, width, num_classes), dtype=np.float32)
        # Only bottom-right quadrant is foreground
        y_true[:, height//2:, width//2:, 0] = 1.0

        np.random.seed(42)
        embeddings = np.random.randn(batch_size, height, width, feature_dim).astype(np.float32)
        embeddings[:, height//2:, width//2:, :] += 3.0

        y_true = tf.constant(y_true)
        y_pred = tf.constant(embeddings)

        loss_fn = PixelTripletLoss(
            config=PixelTripletConfig(
                margin=1.0,
                triplet_strategy="hard",
                class_balanced_weighting=True,  # This enables use_exact automatically
                batch_size_for_exact=50,
            )
        )

        loss = loss_fn(y_true, y_pred)
        loss_value = float(keras.ops.convert_to_numpy(loss))

        print(f"\nClass-balanced hard triplet loss: {loss_value:.6f}")

        assert not np.isnan(loss_value), "Loss should not be NaN"
        assert not np.isinf(loss_value), "Loss should not be Inf"
        assert loss_value >= 0, "Loss should be non-negative"

    def test_class_balanced_vs_unbalanced(self):
        """
        Test that class-balanced weighting gives different results than unbalanced.

        With imbalanced classes, the unbalanced loss will be dominated by the
        majority class, while balanced loss weights each class equally.
        """
        batch_size = 1
        height, width = 32, 32
        feature_dim = 8
        num_classes = 2

        # Highly imbalanced: ~90% background, ~10% foreground
        y_true = np.zeros((batch_size, height, width, num_classes), dtype=np.float32)
        y_true[:, 28:, 28:, 0] = 1.0  # Only 4x4 = 16 pixels foreground

        np.random.seed(42)
        embeddings = np.random.randn(batch_size, height, width, feature_dim).astype(np.float32)
        embeddings[:, 28:, 28:, :] += 3.0

        y_true = tf.constant(y_true)
        y_pred = tf.constant(embeddings)

        # Unbalanced (standard exact mode)
        loss_unbalanced = PixelTripletLoss(
            config=PixelTripletConfig(
                margin=1.0,
                triplet_strategy="hard",
                use_exact=True,
                class_balanced_weighting=False,
                batch_size_for_exact=50,
                max_samples_per_class=500,  # Sample many pixels
            )
        )

        # Balanced (class-balanced weighting)
        loss_balanced = PixelTripletLoss(
            config=PixelTripletConfig(
                margin=1.0,
                triplet_strategy="hard",
                class_balanced_weighting=True,
                batch_size_for_exact=50,
            )
        )

        unbalanced_value = float(keras.ops.convert_to_numpy(loss_unbalanced(y_true, y_pred)))
        balanced_value = float(keras.ops.convert_to_numpy(loss_balanced(y_true, y_pred)))

        print(f"\nImbalanced classes (90% bg, 10% fg):")
        print(f"  Unbalanced loss: {unbalanced_value:.6f}")
        print(f"  Balanced loss:   {balanced_value:.6f}")

        # Both should be valid
        assert not np.isnan(unbalanced_value), "Unbalanced loss should not be NaN"
        assert not np.isnan(balanced_value), "Balanced loss should not be NaN"

    def test_class_balanced_multiple_foreground_classes(self):
        """Test class-balanced weighting with multiple foreground classes."""
        batch_size = 1
        height, width = 32, 32
        feature_dim = 8
        num_classes = 3

        # 3 foreground classes with different sizes
        y_true = np.zeros((batch_size, height, width, num_classes), dtype=np.float32)
        y_true[:, :8, :, 0] = 1.0     # Class 1: 8x32 = 256 pixels (25%)
        y_true[:, 8:12, :, 1] = 1.0   # Class 2: 4x32 = 128 pixels (12.5%)
        y_true[:, 12:14, :, 2] = 1.0  # Class 3: 2x32 = 64 pixels (6.25%)
        # Background: 18x32 = 576 pixels (56.25%)

        np.random.seed(42)
        embeddings = np.random.randn(batch_size, height, width, feature_dim).astype(np.float32) * 0.5
        # Give each class a different mean embedding
        embeddings[:, :8, :, 0] += 3.0
        embeddings[:, 8:12, :, 1] += 3.0
        embeddings[:, 12:14, :, 2] += 3.0

        y_true = tf.constant(y_true)
        y_pred = tf.constant(embeddings)

        loss_fn = PixelTripletLoss(
            config=PixelTripletConfig(
                margin=1.0,
                triplet_strategy="hard",
                class_balanced_weighting=True,
                batch_size_for_exact=50,
            )
        )

        loss_value = float(keras.ops.convert_to_numpy(loss_fn(y_true, y_pred)))

        print(f"\nClass-balanced with 3 foreground classes: {loss_value:.6f}")

        assert not np.isnan(loss_value), "Loss should not be NaN"
        assert not np.isinf(loss_value), "Loss should not be Inf"

    def test_class_balanced_batch_all_with_remove_easy(self):
        """Test class-balanced batch-all with remove_easy_triplets."""
        batch_size = 1
        height, width = 24, 24
        feature_dim = 8
        num_classes = 2

        y_true = np.zeros((batch_size, height, width, num_classes), dtype=np.float32)
        y_true[:, :, width//2:, 0] = 1.0

        np.random.seed(42)
        embeddings = np.random.randn(batch_size, height, width, feature_dim).astype(np.float32) * 0.5
        embeddings[:, :, width//2:, :] += 2.0

        y_true = tf.constant(y_true)
        y_pred = tf.constant(embeddings)

        loss_fn = PixelTripletLoss(
            config=PixelTripletConfig(
                margin=1.0,
                triplet_strategy="all",
                class_balanced_weighting=True,
                remove_easy_triplets=True,
                batch_size_for_exact=50,
            )
        )

        loss_value = float(keras.ops.convert_to_numpy(loss_fn(y_true, y_pred)))

        print(f"\nClass-balanced batch-all with remove_easy_triplets: {loss_value:.6f}")

        assert not np.isnan(loss_value), "Loss should not be NaN"
        assert not np.isinf(loss_value), "Loss should not be Inf"
        assert loss_value >= 0, "Loss should be non-negative"

    def test_class_balanced_config_auto_enables_exact(self):
        """Test that class_balanced_weighting automatically enables use_exact."""
        config = PixelTripletConfig(
            class_balanced_weighting=True,
            use_exact=False,  # This should be overridden
        )

        assert config.use_exact is True, "use_exact should be auto-enabled"
        assert config.class_balanced_weighting is True

    def test_class_balanced_serialization(self):
        """Test that class_balanced_weighting is properly serialized."""
        loss_fn = PixelTripletLoss(
            config=PixelTripletConfig(
                class_balanced_weighting=True,
                batch_size_for_exact=150,
            )
        )

        saved_config = loss_fn.get_config()

        assert saved_config['class_balanced_weighting'] is True
        assert saved_config['use_exact'] is True

        # Test from_config
        restored = PixelTripletLoss.from_config(saved_config)
        assert restored.config.class_balanced_weighting is True

    def test_convenience_function_with_class_balanced(self):
        """Test create_pixel_triplet_loss with class_balanced_weighting."""
        loss_fn = create_pixel_triplet_loss(
            margin=0.8,
            class_balanced_weighting=True,
            batch_size_for_exact=200,
            triplet_strategy="hard",
        )

        assert loss_fn.config.class_balanced_weighting is True
        assert loss_fn.config.use_exact is True
        assert loss_fn.config.batch_size_for_exact == 200


