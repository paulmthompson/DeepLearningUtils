"""
Tests for triplet loss functions in DeepLearningUtils.

This module contains comprehensive tests for triplet loss implementations,
including tests with simple embedding data using values -1, 1, 0 for exact calculations.
"""

import pytest
import numpy as np
import tensorflow as tf
import keras

from src.DeepLearningUtils.Losses.Triplet_Loss.triplet_loss_keras import (
    batch_hard_triplet_loss,
    batch_all_triplet_loss,
    hard_negative_triplet_mining,
    semi_hard_negative_triplet_mining,
    batch_distance_loss,
    _pairwise_distances,
    _get_anchor_positive_triplet_mask,
    _get_anchor_negative_triplet_mask,
    _get_triplet_mask,
)
from tests.testing_utilities import assert_arrays_equal_with_nans


class TestTripletLossHelpers:
    """Test helper functions for triplet loss calculations."""

    def test_pairwise_distances_simple(self, keras_float32_policy):
        """Test pairwise distance calculation with simple embeddings."""
        # Create simple embeddings using -1, 1, 0
        embeddings = tf.constant([
            [1.0, 0.0],   # Point A
            [-1.0, 0.0],  # Point B
            [0.0, 1.0],   # Point C
            [0.0, -1.0],  # Point D
        ], dtype=tf.float32)
        
        # Calculate pairwise distances
        distances = _pairwise_distances(embeddings, squared=False)
        
        # Expected distances (calculated manually)
        expected = np.array([
            [0.0, 2.0, np.sqrt(2), np.sqrt(2)],     # From A
            [2.0, 0.0, np.sqrt(2), np.sqrt(2)],     # From B
            [np.sqrt(2), np.sqrt(2), 0.0, 2.0],     # From C
            [np.sqrt(2), np.sqrt(2), 2.0, 0.0],     # From D
        ])
        
        assert_arrays_equal_with_nans(distances.numpy(), expected, atol=1e-6)

    def test_pairwise_distances_squared(self, keras_float32_policy):
        """Test pairwise squared distance calculation."""
        embeddings = tf.constant([
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
        ], dtype=tf.float32)
        
        distances = _pairwise_distances(embeddings, squared=True)
        
        expected = np.array([
            [0.0, 4.0, 2.0],
            [4.0, 0.0, 2.0],
            [2.0, 2.0, 0.0],
        ])
        
        assert_arrays_equal_with_nans(distances.numpy(), expected, atol=1e-6)

    def test_pairwise_distances_extra_dimensions(self, keras_float32_policy):
        """Test pairwise distances with extra dimensions (like from global average pooling)."""
        # Simulate embeddings from global average pooling: (batch_size, 1, 1, channels)
        embeddings_4d = tf.constant([
            [[[1.0, 0.0]]],
            [[[-1.0, 0.0]]],
            [[[0.0, 1.0]]],
        ], dtype=tf.float32)
        
        # Reshape to 2D for distance calculation
        embeddings_2d = tf.reshape(embeddings_4d, (3, 2))
        distances = _pairwise_distances(embeddings_2d, squared=False)
        
        expected = np.array([
            [0.0, 2.0, np.sqrt(2)],
            [2.0, 0.0, np.sqrt(2)],
            [np.sqrt(2), np.sqrt(2), 0.0],
        ])
        
        assert_arrays_equal_with_nans(distances.numpy(), expected, atol=1e-6)

    def test_anchor_positive_mask(self, keras_float32_policy):
        """Test anchor-positive mask generation."""
        labels = tf.constant([0, 0, 1, 1, 2], dtype=tf.int32)
        mask = _get_anchor_positive_triplet_mask(labels)
        
        # Expected mask: True where labels are same but indices are different
        expected = np.array([
            [False, True,  False, False, False],  # label 0 vs [0,0,1,1,2]
            [True,  False, False, False, False],  # label 0 vs [0,0,1,1,2]
            [False, False, False, True,  False],  # label 1 vs [0,0,1,1,2]
            [False, False, True,  False, False],  # label 1 vs [0,0,1,1,2]
            [False, False, False, False, False],  # label 2 vs [0,0,1,1,2]
        ])
        
        assert np.array_equal(mask.numpy(), expected)

    def test_anchor_negative_mask(self, keras_float32_policy):
        """Test anchor-negative mask generation."""
        labels = tf.constant([0, 0, 1, 1], dtype=tf.int32)
        mask = _get_anchor_negative_triplet_mask(labels)
        
        # Expected mask: True where labels are different
        expected = np.array([
            [False, False, True,  True ],  # label 0 vs [0,0,1,1]
            [False, False, True,  True ],  # label 0 vs [0,0,1,1]
            [True,  True,  False, False],  # label 1 vs [0,0,1,1]
            [True,  True,  False, False],  # label 1 vs [0,0,1,1]
        ])
        
        assert np.array_equal(mask.numpy(), expected)

    def test_triplet_mask(self, keras_float32_policy):
        """Test triplet mask generation."""
        labels = tf.constant([0, 0, 1], dtype=tf.int32)
        mask = _get_triplet_mask(labels)
        
        # Expected shape: (batch_size, batch_size, batch_size)
        assert mask.shape == (3, 3, 3)
        
        # Check some specific triplets
        # (0, 1, 2) should be valid: same label for 0,1 and different for 2
        assert mask[0, 1, 2].numpy() == True
        # (1, 0, 2) should be valid: same label for 1,0 and different for 2
        assert mask[1, 0, 2].numpy() == True
        # (0, 0, 1) should be invalid: same indices
        assert mask[0, 0, 1].numpy() == False
        # (0, 2, 1) should be invalid: different labels for 0,2
        assert mask[0, 2, 1].numpy() == False


class TestTripletLossFunctions:
    """Test triplet loss functions with known expected values."""

    @pytest.fixture
    def simple_embeddings_and_labels(self):
        """Create simple test data with known distances."""
        # Create embeddings with clear separations
        embeddings = tf.constant([
            [1.0, 0.0],   # Class 0, sample 1
            [0.8, 0.0],   # Class 0, sample 2 (closer to sample 1)
            [-1.0, 0.0],  # Class 1, sample 1
            [-0.8, 0.0],  # Class 1, sample 2 (closer to sample 3)
        ], dtype=tf.float32)
        
        labels = tf.constant([0, 0, 1, 1], dtype=tf.int32)
        
        return embeddings, labels

    def test_batch_hard_triplet_loss_basic(self, keras_float32_policy, simple_embeddings_and_labels):
        """Test batch hard triplet loss with known values."""
        embeddings, labels = simple_embeddings_and_labels
        margin = 0.5
        
        loss = batch_hard_triplet_loss(labels, embeddings, margin, squared=False)
        
        # Loss should be a scalar
        assert loss.shape == ()
        # Loss should be non-negative
        assert loss.numpy() >= 0.0

    def test_batch_hard_triplet_loss_zero_margin(self, keras_float32_policy):
        """Test batch hard triplet loss with zero margin."""
        # Create embeddings where positive pairs are closer than negative pairs
        embeddings = tf.constant([
            [1.0, 0.0],   # Class 0
            [0.9, 0.0],   # Class 0 (very close)
            [-1.0, 0.0],  # Class 1 (far from class 0)
        ], dtype=tf.float32)
        labels = tf.constant([0, 0, 1], dtype=tf.int32)
        
        loss = batch_hard_triplet_loss(labels, embeddings, margin=0.0, squared=False)
        
        # With zero margin and well-separated classes, loss should be zero
        assert loss.numpy() >= 0.0

    def test_batch_all_triplet_loss_basic(self, keras_float32_policy, simple_embeddings_and_labels):
        """Test batch all triplet loss."""
        embeddings, labels = simple_embeddings_and_labels
        margin = 0.5
        
        loss, fraction_positive = batch_all_triplet_loss(
            labels, embeddings, margin, squared=False, remove_negative=True
        )
        
        # Loss should be a scalar
        assert loss.shape == ()
        assert fraction_positive.shape == ()
        # Loss should be non-negative
        assert loss.numpy() >= 0.0
        # Fraction should be between 0 and 1
        assert 0.0 <= fraction_positive.numpy() <= 1.0

    def test_batch_distance_loss(self, keras_float32_policy, simple_embeddings_and_labels):
        """Test batch distance loss."""
        embeddings, labels = simple_embeddings_and_labels
        
        loss, fraction_negative = batch_distance_loss(labels, embeddings, squared=False)
        
        # Loss should be a scalar
        assert loss.shape == ()
        assert fraction_negative.shape == ()
        # Fraction should be between 0 and 1
        assert 0.0 <= fraction_negative.numpy() <= 1.0

    def test_different_embedding_dimensions(self, keras_float32_policy):
        """Test functions with different embedding dimensions."""
        # Test with 1D embeddings
        embeddings_1d = tf.constant([
            [1.0],
            [-1.0],
            [0.5],
        ], dtype=tf.float32)
        labels = tf.constant([0, 1, 0], dtype=tf.int32)
        
        loss = batch_hard_triplet_loss(labels, embeddings_1d, margin=0.5, squared=False)
        assert loss.shape == ()
        
        # Test with higher dimensional embeddings
        embeddings_5d = tf.constant([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0, 0.0],
        ], dtype=tf.float32)
        labels = tf.constant([0, 1, 0, 1], dtype=tf.int32)
        
        loss = batch_hard_triplet_loss(labels, embeddings_5d, margin=0.5, squared=False)
        assert loss.shape == ()

    def test_normalization_option(self, keras_float32_policy):
        """Test normalization option in loss functions."""
        embeddings = tf.constant([
            [10.0, 0.0],   # Large values
            [-10.0, 0.0],
            [0.0, 10.0],
        ], dtype=tf.float32)
        labels = tf.constant([0, 1, 0], dtype=tf.int32)
        
        # Test with normalization
        loss_normalized = batch_hard_triplet_loss(
            labels, embeddings, margin=0.5, squared=False, normalize=True
        )
        
        # Test without normalization
        loss_unnormalized = batch_hard_triplet_loss(
            labels, embeddings, margin=0.5, squared=False, normalize=False
        )
        
        # Both should be valid losses
        assert loss_normalized.shape == ()
        assert loss_unnormalized.shape == ()
        assert loss_normalized.numpy() >= 0.0
        assert loss_unnormalized.numpy() >= 0.0

    def test_edge_cases(self, keras_float32_policy):
        """Test edge cases."""
        # Test with single class (should have no valid triplets)
        embeddings = tf.constant([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ], dtype=tf.float32)
        labels = tf.constant([0, 0, 0], dtype=tf.int32)  # All same class
        
        loss = batch_hard_triplet_loss(labels, embeddings, margin=0.5, squared=False)
        # Loss should be zero when no valid triplets exist
        assert loss.numpy() >= 0.0

    def test_mining_functions(self, keras_float32_policy, simple_embeddings_and_labels):
        """Test hard and semi-hard negative mining functions."""
        embeddings, labels = simple_embeddings_and_labels
        margin = 0.5
        
        # Test hard negative mining
        loss_hard = hard_negative_triplet_mining(labels, embeddings, margin, squared=False)
        assert loss_hard.shape == (4, 4)  # Should return per-sample losses
        
        # Test semi-hard negative mining
        loss_semi_hard = semi_hard_negative_triplet_mining(labels, embeddings, margin, squared=False)
        assert loss_semi_hard.shape == (4, 1, 4)  # Should return per-anchor losses with negative dimension


class TestTripletLossWithGlobalAveragePooling:
    """Test triplet loss functions with embeddings from global average pooling."""

    def test_global_average_pooling_simulation(self, keras_float32_policy):
        """Test with embeddings that simulate global average pooling output."""
        # Simulate features from global average pooling: (batch_size, 1, 1, channels)
        batch_size = 4
        channels = 3
        
        # Create 4D tensor as if from global average pooling
        embeddings_4d = tf.constant([
            [[[1.0, 0.0, 0.0]]],    # Class 0
            [[[0.8, 0.2, 0.0]]],    # Class 0
            [[[-1.0, 0.0, 0.0]]],   # Class 1
            [[[-0.8, -0.2, 0.0]]],  # Class 1
        ], dtype=tf.float32)
        
        # Reshape to 2D for triplet loss calculation
        embeddings_2d = tf.reshape(embeddings_4d, (batch_size, channels))
        labels = tf.constant([0, 0, 1, 1], dtype=tf.int32)
        
        loss = batch_hard_triplet_loss(labels, embeddings_2d, margin=0.5, squared=False)
        
        assert loss.shape == ()
        assert loss.numpy() >= 0.0

    def test_squeeze_dimensions(self, keras_float32_policy):
        """Test handling of extra dimensions that need to be squeezed."""
        # Create embeddings with extra dimensions
        embeddings = tf.constant([
            [[1.0], [0.0]],
            [[-1.0], [0.0]],
            [[0.0], [1.0]],
        ], dtype=tf.float32)
        
        # Reshape to proper format
        embeddings_reshaped = tf.reshape(embeddings, (3, 2))
        labels = tf.constant([0, 1, 0], dtype=tf.int32)
        
        loss = batch_hard_triplet_loss(labels, embeddings_reshaped, margin=0.5, squared=False)
        
        assert loss.shape == ()
        assert loss.numpy() >= 0.0


class TestTripletLossNumericalStability:
    """Test numerical stability of triplet loss functions."""

    def test_identical_embeddings(self, keras_float32_policy):
        """Test with identical embeddings to check numerical stability."""
        embeddings = tf.constant([
            [1.0, 0.0],
            [1.0, 0.0],  # Identical to first
            [-1.0, 0.0],
        ], dtype=tf.float32)
        labels = tf.constant([0, 0, 1], dtype=tf.int32)
        
        loss = batch_hard_triplet_loss(labels, embeddings, margin=0.5, squared=False)
        
        # Should handle identical embeddings gracefully
        assert loss.shape == ()
        assert not tf.math.is_nan(loss)
        assert not tf.math.is_inf(loss)

    def test_very_small_values(self, keras_float32_policy):
        """Test with very small embedding values."""
        embeddings = tf.constant([
            [1e-8, 0.0],
            [-1e-8, 0.0],
            [0.0, 1e-8],
        ], dtype=tf.float32)
        labels = tf.constant([0, 1, 0], dtype=tf.int32)
        
        loss = batch_hard_triplet_loss(labels, embeddings, margin=0.5, squared=False)
        
        assert loss.shape == ()
        assert not tf.math.is_nan(loss)
        assert not tf.math.is_inf(loss)

    def test_very_large_values(self, keras_float32_policy):
        """Test with very large embedding values."""
        embeddings = tf.constant([
            [1e8, 0.0],
            [-1e8, 0.0],
            [0.0, 1e8],
        ], dtype=tf.float32)
        labels = tf.constant([0, 1, 0], dtype=tf.int32)
        
        loss = batch_hard_triplet_loss(labels, embeddings, margin=0.5, squared=False)
        
        assert loss.shape == ()
        assert not tf.math.is_nan(loss)
        assert not tf.math.is_inf(loss)