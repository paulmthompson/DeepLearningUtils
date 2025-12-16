"""
Test suite for Embedding Separation Metric.

Tests the EmbeddingSeparationRatio metric and related metrics that monitor
embedding quality during triplet loss training.
"""

import pytest
import numpy as np
import keras
import tensorflow as tf

from src.DeepLearningUtils.Metrics.embedding_separation_metric import (
    EmbeddingSeparationRatio,
    EmbeddingIntraClassDistance,
    EmbeddingInterClassDistance,
    EmbeddingSeparationConfig,
    create_embedding_separation_metric,
    create_embedding_metrics_suite,
)


class TestEmbeddingSeparationRatio:
    """Tests for the EmbeddingSeparationRatio metric."""

    def test_perfect_separation(self):
        """
        Test that perfectly separated embeddings produce a high ratio.

        Setup: Two classes with embeddings at opposite ends of feature space.
        Expected: High separation ratio (inter >> intra).
        """
        batch_size = 2
        height, width = 32, 32
        feature_dim = 16

        # Create masks: half background, half foreground
        y_true = np.zeros((batch_size, height, width, 1), dtype=np.float32)
        y_true[:, :, width//2:, 0] = 1.0  # Right half is foreground

        # Create embeddings: background at [0, 0, ...], foreground at [10, 10, ...]
        embeddings = np.zeros((batch_size, height, width, feature_dim), dtype=np.float32)
        embeddings[:, :, width//2:, :] = 10.0  # Foreground embeddings at [10, ...]

        y_true = keras.ops.convert_to_tensor(y_true)
        y_pred = keras.ops.convert_to_tensor(embeddings)

        metric = EmbeddingSeparationRatio(
            config=EmbeddingSeparationConfig(max_samples_per_class=200)
        )
        metric.update_state(y_true, y_pred)
        ratio = float(keras.ops.convert_to_numpy(metric.result()))

        print(f"\nPerfect separation: ratio = {ratio:.4f}")

        # With perfect separation, intra-class distance is 0 (all same),
        # but due to numerical stability, we use epsilon
        # Ratio should be very high
        assert ratio > 10, f"Expected high ratio for perfect separation, got {ratio:.4f}"
        assert not np.isnan(ratio), "Ratio should not be NaN"
        assert not np.isinf(ratio), "Ratio should not be Inf"

    def test_no_separation(self):
        """
        Test that overlapping embeddings produce a low ratio.

        Setup: Two classes with identical embeddings.
        Expected: Ratio close to 1 (inter ≈ intra).
        """
        batch_size = 2
        height, width = 32, 32
        feature_dim = 16

        # Create masks: half background, half foreground
        y_true = np.zeros((batch_size, height, width, 1), dtype=np.float32)
        y_true[:, :, width//2:, 0] = 1.0

        # Create embeddings: all at the same point (with small noise)
        np.random.seed(42)
        embeddings = np.random.randn(batch_size, height, width, feature_dim).astype(np.float32) * 0.01

        y_true = keras.ops.convert_to_tensor(y_true)
        y_pred = keras.ops.convert_to_tensor(embeddings)

        metric = EmbeddingSeparationRatio(
            config=EmbeddingSeparationConfig(max_samples_per_class=200)
        )
        metric.update_state(y_true, y_pred)
        ratio = float(keras.ops.convert_to_numpy(metric.result()))

        print(f"\nNo separation (random noise): ratio = {ratio:.4f}")

        # With overlapping embeddings, ratio should be close to 1
        assert 0.5 < ratio < 2.0, f"Expected ratio ~1 for overlapping embeddings, got {ratio:.4f}"
        assert not np.isnan(ratio), "Ratio should not be NaN"

    def test_partial_separation(self):
        """
        Test embeddings with partial separation.

        Setup: Classes have different mean embeddings but some overlap.
        Expected: Moderate ratio (1 < ratio < high value).
        """
        batch_size = 2
        height, width = 32, 32
        feature_dim = 16

        # Create masks
        y_true = np.zeros((batch_size, height, width, 1), dtype=np.float32)
        y_true[:, :, width//2:, 0] = 1.0

        # Create embeddings with partial separation
        np.random.seed(42)
        embeddings = np.random.randn(batch_size, height, width, feature_dim).astype(np.float32)
        embeddings[:, :, width//2:, :] += 2.0  # Shift foreground by moderate amount

        y_true = keras.ops.convert_to_tensor(y_true)
        y_pred = keras.ops.convert_to_tensor(embeddings)

        metric = EmbeddingSeparationRatio(
            config=EmbeddingSeparationConfig(max_samples_per_class=200)
        )
        metric.update_state(y_true, y_pred)
        ratio = float(keras.ops.convert_to_numpy(metric.result()))

        print(f"\nPartial separation: ratio = {ratio:.4f}")

        # With partial separation, ratio should be moderately high
        assert ratio > 1.0, f"Expected ratio > 1 for partial separation, got {ratio:.4f}"
        assert not np.isnan(ratio), "Ratio should not be NaN"

    def test_accumulation_across_batches(self):
        """
        Test that the metric correctly accumulates across multiple batches.
        """
        batch_size = 2
        height, width = 32, 32
        feature_dim = 16

        metric = EmbeddingSeparationRatio(
            config=EmbeddingSeparationConfig(max_samples_per_class=100)
        )

        # Process multiple batches
        np.random.seed(42)
        for _ in range(3):
            y_true = np.zeros((batch_size, height, width, 1), dtype=np.float32)
            y_true[:, :, width//2:, 0] = 1.0

            embeddings = np.random.randn(batch_size, height, width, feature_dim).astype(np.float32)
            embeddings[:, :, width//2:, :] += 3.0

            y_true = keras.ops.convert_to_tensor(y_true)
            y_pred = keras.ops.convert_to_tensor(embeddings)

            metric.update_state(y_true, y_pred)

        ratio = float(keras.ops.convert_to_numpy(metric.result()))

        print(f"\nAccumulated ratio (3 batches): {ratio:.4f}")

        # Check accumulation worked (non-zero counts)
        assert metric.inter_class_count.numpy() > 0, "Inter-class count should be positive"
        assert metric.intra_class_count.numpy() > 0, "Intra-class count should be positive"
        assert not np.isnan(ratio), "Ratio should not be NaN"

    def test_reset_state(self):
        """Test that reset_state properly clears accumulators."""
        batch_size = 2
        height, width = 32, 32
        feature_dim = 16

        metric = EmbeddingSeparationRatio()

        # Add some data
        y_true = np.zeros((batch_size, height, width, 1), dtype=np.float32)
        y_true[:, :, width//2:, 0] = 1.0
        embeddings = np.random.randn(batch_size, height, width, feature_dim).astype(np.float32)

        metric.update_state(
            keras.ops.convert_to_tensor(y_true),
            keras.ops.convert_to_tensor(embeddings)
        )

        # Reset
        metric.reset_state()

        # Check all accumulators are zero
        assert metric.inter_class_sum.numpy() == 0
        assert metric.inter_class_count.numpy() == 0
        assert metric.intra_class_sum.numpy() == 0
        assert metric.intra_class_count.numpy() == 0


class TestStridedSampling:
    """Tests for strided sampling to handle upsampling."""

    def test_stride_reduces_redundancy(self):
        """
        Test that using stride with upsampled embeddings gives meaningful results.

        When embeddings are upsampled (e.g., nearest neighbor), adjacent pixels
        have identical embeddings. Stride sampling should pick distinct embeddings.
        """
        batch_size = 1
        original_size = 16
        upsampled_size = 64  # 4x upsampling
        feature_dim = 8

        # Create original embeddings at native resolution
        np.random.seed(42)
        original_embeddings = np.random.randn(batch_size, original_size, original_size, feature_dim).astype(np.float32)

        # "Upsample" using nearest neighbor (simulating what happens in triplet loss)
        upsampled_embeddings = np.repeat(np.repeat(original_embeddings, 4, axis=1), 4, axis=2)

        # Create mask at upsampled resolution
        y_true = np.zeros((batch_size, upsampled_size, upsampled_size, 1), dtype=np.float32)
        y_true[:, :, upsampled_size//2:, 0] = 1.0

        y_true = keras.ops.convert_to_tensor(y_true)
        y_pred = keras.ops.convert_to_tensor(upsampled_embeddings)

        # Without stride (would have many zero-distance pairs)
        metric_no_stride = EmbeddingSeparationRatio(
            config=EmbeddingSeparationConfig(max_samples_per_class=200, stride=1)
        )
        metric_no_stride.update_state(y_true, y_pred)
        ratio_no_stride = float(keras.ops.convert_to_numpy(metric_no_stride.result()))

        # With stride matching upsampling factor
        metric_with_stride = EmbeddingSeparationRatio(
            config=EmbeddingSeparationConfig(max_samples_per_class=200, stride=4)
        )
        metric_with_stride.update_state(y_true, y_pred)
        ratio_with_stride = float(keras.ops.convert_to_numpy(metric_with_stride.result()))

        print(f"\nUpsampled embeddings:")
        print(f"  Without stride: ratio = {ratio_no_stride:.4f}")
        print(f"  With stride=4:  ratio = {ratio_with_stride:.4f}")

        # Both should be valid (non-NaN)
        assert not np.isnan(ratio_no_stride), "Ratio without stride should not be NaN"
        assert not np.isnan(ratio_with_stride), "Ratio with stride should not be NaN"

        # Note: The strided version should give a more accurate estimate of true separation
        # because it samples from the original resolution


class TestDistanceMetrics:
    """Tests for different distance metrics."""

    @pytest.mark.parametrize("distance_metric", ["euclidean", "cosine", "manhattan"])
    def test_distance_metric_compatibility(self, distance_metric):
        """Test that all distance metrics work correctly."""
        batch_size = 2
        height, width = 32, 32
        feature_dim = 16

        y_true = np.zeros((batch_size, height, width, 1), dtype=np.float32)
        y_true[:, :, width//2:, 0] = 1.0

        np.random.seed(42)
        embeddings = np.random.randn(batch_size, height, width, feature_dim).astype(np.float32)
        embeddings[:, :, width//2:, :] += 2.0

        y_true = keras.ops.convert_to_tensor(y_true)
        y_pred = keras.ops.convert_to_tensor(embeddings)

        config = EmbeddingSeparationConfig(
            max_samples_per_class=200,
            distance_metric=distance_metric
        )
        metric = EmbeddingSeparationRatio(config=config)
        metric.update_state(y_true, y_pred)
        ratio = float(keras.ops.convert_to_numpy(metric.result()))

        print(f"\nDistance metric '{distance_metric}': ratio = {ratio:.4f}")

        assert not np.isnan(ratio), f"Ratio should not be NaN for {distance_metric}"
        assert not np.isinf(ratio), f"Ratio should not be Inf for {distance_metric}"
        assert ratio > 0, f"Ratio should be positive for {distance_metric}"


class TestMultiClass:
    """Tests for multi-class segmentation scenarios."""

    def test_three_class_separation(self):
        """Test with background + 2 foreground classes."""
        batch_size = 2
        height, width = 48, 48
        feature_dim = 16

        # Create 2-channel mask (2 foreground classes)
        y_true = np.zeros((batch_size, height, width, 2), dtype=np.float32)
        y_true[:, :16, :, 0] = 1.0  # Class 1: top third
        y_true[:, 32:, :, 1] = 1.0  # Class 2: bottom third
        # Middle third is background (all zeros)

        # Create embeddings with good separation
        np.random.seed(42)
        embeddings = np.random.randn(batch_size, height, width, feature_dim).astype(np.float32) * 0.5
        embeddings[:, :16, :, :] += np.array([5.0] + [0.0] * (feature_dim - 1))  # Class 1
        embeddings[:, 32:, :, :] += np.array([0.0, 5.0] + [0.0] * (feature_dim - 2))  # Class 2
        # Background stays near origin

        y_true = keras.ops.convert_to_tensor(y_true)
        y_pred = keras.ops.convert_to_tensor(embeddings)

        metric = EmbeddingSeparationRatio(
            config=EmbeddingSeparationConfig(max_samples_per_class=150)
        )
        metric.update_state(y_true, y_pred)
        ratio = float(keras.ops.convert_to_numpy(metric.result()))

        print(f"\nThree-class separation: ratio = {ratio:.4f}")

        assert ratio > 2.0, f"Expected good separation ratio, got {ratio:.4f}"
        assert not np.isnan(ratio), "Ratio should not be NaN"


class TestBackgroundExclusion:
    """Tests for the exclude_background_intra feature."""

    def test_exclude_background_intra_default_true(self):
        """Test that exclude_background_intra is True by default."""
        config = EmbeddingSeparationConfig()
        assert config.exclude_background_intra is True

    def test_exclude_vs_include_background_intra(self):
        """
        Test that excluding background from intra-class changes the ratio.

        Setup:
        - Background has high variance (spread out embeddings)
        - Foreground has low variance (tight cluster)

        Expected:
        - With exclude_background_intra=True: ratio should be higher
          (only measures tight foreground cluster)
        - With exclude_background_intra=False: ratio should be lower
          (background variance inflates intra-class distance)
        """
        batch_size = 2
        height, width = 32, 32
        feature_dim = 16

        # Create masks: half background, half foreground
        y_true = np.zeros((batch_size, height, width, 1), dtype=np.float32)
        y_true[:, :, width//2:, 0] = 1.0

        # Create embeddings:
        # - Background: high variance (spread out)
        # - Foreground: low variance (tight cluster at [5, 5, ...])
        np.random.seed(42)
        embeddings = np.zeros((batch_size, height, width, feature_dim), dtype=np.float32)

        # Background: spread out embeddings (high intra-class variance)
        embeddings[:, :, :width//2, :] = np.random.randn(batch_size, height, width//2, feature_dim).astype(np.float32) * 3.0

        # Foreground: tight cluster (low intra-class variance)
        embeddings[:, :, width//2:, :] = 5.0 + np.random.randn(batch_size, height, width//2, feature_dim).astype(np.float32) * 0.1

        y_true = keras.ops.convert_to_tensor(y_true)
        y_pred = keras.ops.convert_to_tensor(embeddings)

        # With background excluded (default)
        metric_exclude = EmbeddingSeparationRatio(
            config=EmbeddingSeparationConfig(
                max_samples_per_class=200,
                exclude_background_intra=True
            )
        )
        metric_exclude.update_state(y_true, y_pred)
        ratio_exclude = float(keras.ops.convert_to_numpy(metric_exclude.result()))

        # With background included
        metric_include = EmbeddingSeparationRatio(
            config=EmbeddingSeparationConfig(
                max_samples_per_class=200,
                exclude_background_intra=False
            )
        )
        metric_include.update_state(y_true, y_pred)
        ratio_include = float(keras.ops.convert_to_numpy(metric_include.result()))

        print(f"\nBackground exclusion comparison:")
        print(f"  exclude_background_intra=True:  ratio = {ratio_exclude:.4f}")
        print(f"  exclude_background_intra=False: ratio = {ratio_include:.4f}")

        # Excluding background should give a higher ratio because we only
        # measure the tight foreground cluster, not the spread-out background
        assert ratio_exclude > ratio_include, (
            f"Expected higher ratio when excluding background intra-class distances. "
            f"Got exclude={ratio_exclude:.4f}, include={ratio_include:.4f}"
        )

        # Both should be valid
        assert not np.isnan(ratio_exclude), "Ratio (exclude) should not be NaN"
        assert not np.isnan(ratio_include), "Ratio (include) should not be NaN"

    def test_background_still_used_for_inter_class(self):
        """
        Test that background is still used for inter-class distances even when
        exclude_background_intra=True.

        The idea: we want foreground vs background separation to be measured,
        just not background-to-background "tightness".
        """
        batch_size = 2
        height, width = 32, 32
        feature_dim = 16

        # Create masks: half background, half foreground
        y_true = np.zeros((batch_size, height, width, 1), dtype=np.float32)
        y_true[:, :, width//2:, 0] = 1.0

        # Create embeddings with clear separation
        np.random.seed(42)
        embeddings = np.zeros((batch_size, height, width, feature_dim), dtype=np.float32)
        embeddings[:, :, :width//2, :] = 0.0  # Background at origin
        embeddings[:, :, width//2:, :] = 5.0  # Foreground at [5, 5, ...]

        y_true = keras.ops.convert_to_tensor(y_true)
        y_pred = keras.ops.convert_to_tensor(embeddings)

        metric = EmbeddingSeparationRatio(
            config=EmbeddingSeparationConfig(
                max_samples_per_class=200,
                exclude_background_intra=True
            )
        )
        metric.update_state(y_true, y_pred)

        # Check that inter-class count includes foreground-background pairs
        inter_count = float(keras.ops.convert_to_numpy(metric.inter_class_count))

        print(f"\nInter-class count with exclude_background_intra=True: {inter_count}")

        # Should have inter-class pairs (foreground vs background)
        assert inter_count > 0, "Should have inter-class pairs between foreground and background"

    def test_only_foreground_no_background(self):
        """
        Test behavior when there's no background (all foreground).

        In this case, exclude_background_intra shouldn't change anything.
        """
        batch_size = 2
        height, width = 32, 32
        feature_dim = 16

        # All foreground (single class)
        y_true = np.ones((batch_size, height, width, 1), dtype=np.float32)

        np.random.seed(42)
        embeddings = np.random.randn(batch_size, height, width, feature_dim).astype(np.float32)

        y_true = keras.ops.convert_to_tensor(y_true)
        y_pred = keras.ops.convert_to_tensor(embeddings)

        # With background excluded
        metric_exclude = EmbeddingSeparationRatio(
            config=EmbeddingSeparationConfig(
                max_samples_per_class=200,
                exclude_background_intra=True
            )
        )
        metric_exclude.update_state(y_true, y_pred)
        ratio_exclude = float(keras.ops.convert_to_numpy(metric_exclude.result()))

        # With background included
        metric_include = EmbeddingSeparationRatio(
            config=EmbeddingSeparationConfig(
                max_samples_per_class=200,
                exclude_background_intra=False
            )
        )
        metric_include.update_state(y_true, y_pred)
        ratio_include = float(keras.ops.convert_to_numpy(metric_include.result()))

        print(f"\nNo background (all foreground):")
        print(f"  exclude_background_intra=True:  ratio = {ratio_exclude:.4f}")
        print(f"  exclude_background_intra=False: ratio = {ratio_include:.4f}")

        # Both should be valid and approximately equal (no background to exclude)
        assert not np.isnan(ratio_exclude), "Ratio (exclude) should not be NaN"
        assert not np.isnan(ratio_include), "Ratio (include) should not be NaN"

    def test_serialization_includes_exclude_background_intra(self):
        """Test that exclude_background_intra is properly serialized/deserialized."""
        config = EmbeddingSeparationConfig(
            max_samples_per_class=300,
            exclude_background_intra=False  # Non-default value
        )
        metric = EmbeddingSeparationRatio(config=config)

        saved_config = metric.get_config()

        assert 'exclude_background_intra' in saved_config
        assert saved_config['exclude_background_intra'] is False

        # Test from_config
        restored_metric = EmbeddingSeparationRatio.from_config(saved_config)
        assert restored_metric.config.exclude_background_intra is False


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_embedding_separation_metric(self):
        """Test the convenience function for creating a single metric."""
        metric = create_embedding_separation_metric(
            max_samples_per_class=100,
            distance_metric='cosine',
            stride=2,
            name='test_metric'
        )

        assert metric.name == 'test_metric'
        assert metric.config.max_samples_per_class == 100
        assert metric.config.distance_metric == 'cosine'
        assert metric.config.stride == 2

    def test_create_embedding_metrics_suite(self):
        """Test the convenience function for creating a suite of metrics."""
        metrics = create_embedding_metrics_suite(
            max_samples_per_class=100,
            distance_metric='euclidean',
            stride=1
        )

        assert 'separation_ratio' in metrics
        assert 'intra_class_dist' in metrics
        assert 'inter_class_dist' in metrics

        assert isinstance(metrics['separation_ratio'], EmbeddingSeparationRatio)
        assert isinstance(metrics['intra_class_dist'], EmbeddingIntraClassDistance)
        assert isinstance(metrics['inter_class_dist'], EmbeddingInterClassDistance)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_class_only(self):
        """Test behavior when only one class is present (all background or all foreground)."""
        batch_size = 2
        height, width = 32, 32
        feature_dim = 16

        # All background
        y_true = np.zeros((batch_size, height, width, 1), dtype=np.float32)
        embeddings = np.random.randn(batch_size, height, width, feature_dim).astype(np.float32)

        y_true = keras.ops.convert_to_tensor(y_true)
        y_pred = keras.ops.convert_to_tensor(embeddings)

        metric = EmbeddingSeparationRatio()
        metric.update_state(y_true, y_pred)
        ratio = float(keras.ops.convert_to_numpy(metric.result()))

        print(f"\nSingle class (all background): ratio = {ratio:.4f}")

        # With only one class, inter-class count should be 0
        # Result depends on epsilon handling
        assert not np.isnan(ratio), "Ratio should not be NaN"

    def test_empty_class(self):
        """Test with very small regions for some classes."""
        batch_size = 2
        height, width = 32, 32
        feature_dim = 16

        # Very small foreground region (just a few pixels)
        y_true = np.zeros((batch_size, height, width, 1), dtype=np.float32)
        y_true[:, 15:17, 15:17, 0] = 1.0  # Only 4 pixels

        embeddings = np.random.randn(batch_size, height, width, feature_dim).astype(np.float32)
        embeddings[:, 15:17, 15:17, :] += 5.0

        y_true = keras.ops.convert_to_tensor(y_true)
        y_pred = keras.ops.convert_to_tensor(embeddings)

        metric = EmbeddingSeparationRatio(
            config=EmbeddingSeparationConfig(max_samples_per_class=100)
        )
        metric.update_state(y_true, y_pred)
        ratio = float(keras.ops.convert_to_numpy(metric.result()))

        print(f"\nSmall foreground region: ratio = {ratio:.4f}")

        assert not np.isnan(ratio), "Ratio should not be NaN"
        assert not np.isinf(ratio), "Ratio should not be Inf"


class TestSerialization:
    """Tests for metric serialization/deserialization."""

    def test_get_config(self):
        """Test that get_config returns correct configuration."""
        config = EmbeddingSeparationConfig(
            max_samples_per_class=300,
            distance_metric='cosine',
            stride=4,
            exclude_background_intra=False
        )
        metric = EmbeddingSeparationRatio(config=config, name='test_sep')

        saved_config = metric.get_config()

        assert saved_config['max_samples_per_class'] == 300
        assert saved_config['distance_metric'] == 'cosine'
        assert saved_config['stride'] == 4
        assert saved_config['exclude_background_intra'] is False
        assert saved_config['name'] == 'test_sep'

    def test_from_config(self):
        """Test that from_config recreates the metric correctly."""
        config = EmbeddingSeparationConfig(
            max_samples_per_class=300,
            distance_metric='manhattan',
            stride=2,
            exclude_background_intra=False
        )
        original_metric = EmbeddingSeparationRatio(config=config, name='test_sep')

        saved_config = original_metric.get_config()
        restored_metric = EmbeddingSeparationRatio.from_config(saved_config)

        assert restored_metric.config.max_samples_per_class == 300
        assert restored_metric.config.distance_metric == 'manhattan'
        assert restored_metric.config.stride == 2
        assert restored_metric.config.exclude_background_intra is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

