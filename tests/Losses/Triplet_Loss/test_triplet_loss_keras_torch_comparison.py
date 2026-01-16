"""
Tests comparing Keras/TensorFlow and PyTorch implementations of triplet loss.

This module verifies that the PyTorch triplet loss functions produce identical
results to the Keras/TensorFlow implementations.
"""

import pytest
import numpy as np


# Skip all tests if either TensorFlow or PyTorch is not available
tensorflow_available = False
torch_available = False

try:
    import tensorflow as tf
    import keras
    tensorflow_available = True
except ImportError:
    pass

try:
    import torch
    torch_available = True
except ImportError:
    pass


# Conditional imports
if tensorflow_available:
    from src.DeepLearningUtils.Losses.Triplet_Loss.triplet_loss_keras import (
        batch_hard_triplet_loss as keras_batch_hard_triplet_loss,
        batch_all_triplet_loss as keras_batch_all_triplet_loss,
        hard_negative_triplet_mining as keras_hard_negative_triplet_mining,
        batch_distance_loss as keras_batch_distance_loss,
        _pairwise_distances as keras_pairwise_distances,
        _get_anchor_positive_triplet_mask as keras_get_anchor_positive_triplet_mask,
        _get_anchor_negative_triplet_mask as keras_get_anchor_negative_triplet_mask,
        _get_triplet_mask as keras_get_triplet_mask,
    )

if torch_available:
    from src.DeepLearningUtils.Losses.Triplet_Loss.triplet_loss_torch import (
        batch_hard_triplet_loss as torch_batch_hard_triplet_loss,
        batch_all_triplet_loss as torch_batch_all_triplet_loss,
        hard_negative_triplet_mining as torch_hard_negative_triplet_mining,
        batch_distance_loss as torch_batch_distance_loss,
        _pairwise_distances as torch_pairwise_distances,
        _get_anchor_positive_triplet_mask as torch_get_anchor_positive_triplet_mask,
        _get_anchor_negative_triplet_mask as torch_get_anchor_negative_triplet_mask,
        _get_triplet_mask as torch_get_triplet_mask,
    )


requires_both = pytest.mark.skipif(
    not (tensorflow_available and torch_available),
    reason="Both TensorFlow and PyTorch are required for comparison tests"
)


@pytest.fixture
def keras_float32_policy():
    """Fixture to set Keras dtype policy to float32."""
    if tensorflow_available:
        original_policy = keras.mixed_precision.dtype_policy().name
        keras.mixed_precision.set_dtype_policy("float32")
        yield
        keras.mixed_precision.set_dtype_policy(original_policy)
    else:
        yield


@pytest.fixture
def simple_embeddings():
    """Create simple test embeddings using -1, 1, 0 for exact calculations."""
    embeddings_np = np.array([
        [1.0, 0.0],   # Point A
        [-1.0, 0.0],  # Point B
        [0.0, 1.0],   # Point C
        [0.0, -1.0],  # Point D
    ], dtype=np.float32)
    return embeddings_np


@pytest.fixture
def simple_labels():
    """Create simple test labels."""
    labels_np = np.array([0, 0, 1, 1], dtype=np.int32)
    return labels_np


@pytest.fixture
def random_embeddings():
    """Create random test embeddings."""
    np.random.seed(42)
    embeddings_np = np.random.randn(16, 8).astype(np.float32)
    return embeddings_np


@pytest.fixture
def random_labels():
    """Create random test labels with 4 classes."""
    np.random.seed(42)
    labels_np = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=np.int32)
    return labels_np


@requires_both
class TestPairwiseDistances:
    """Test pairwise distance calculations match between Keras and PyTorch."""
    
    def test_pairwise_distances_simple(self, keras_float32_policy, simple_embeddings):
        """Test pairwise distances with simple embeddings."""
        # Keras
        embeddings_tf = tf.constant(simple_embeddings)
        distances_keras = keras_pairwise_distances(embeddings_tf, squared=False).numpy()
        
        # PyTorch
        embeddings_torch = torch.tensor(simple_embeddings)
        distances_torch = torch_pairwise_distances(embeddings_torch, squared=False).numpy()
        
        np.testing.assert_allclose(distances_keras, distances_torch, rtol=1e-5, atol=1e-5)
    
    def test_pairwise_distances_squared(self, keras_float32_policy, simple_embeddings):
        """Test squared pairwise distances."""
        # Keras
        embeddings_tf = tf.constant(simple_embeddings)
        distances_keras = keras_pairwise_distances(embeddings_tf, squared=True).numpy()
        
        # PyTorch
        embeddings_torch = torch.tensor(simple_embeddings)
        distances_torch = torch_pairwise_distances(embeddings_torch, squared=True).numpy()
        
        np.testing.assert_allclose(distances_keras, distances_torch, rtol=1e-5, atol=1e-5)
    
    def test_pairwise_distances_random(self, keras_float32_policy, random_embeddings):
        """Test pairwise distances with random embeddings."""
        # Keras
        embeddings_tf = tf.constant(random_embeddings)
        distances_keras = keras_pairwise_distances(embeddings_tf, squared=False).numpy()
        
        # PyTorch
        embeddings_torch = torch.tensor(random_embeddings)
        distances_torch = torch_pairwise_distances(embeddings_torch, squared=False).numpy()
        
        np.testing.assert_allclose(distances_keras, distances_torch, rtol=1e-5, atol=1e-5)


@requires_both
class TestTripletMasks:
    """Test triplet mask generation matches between Keras and PyTorch."""
    
    def test_anchor_positive_mask(self, keras_float32_policy, simple_labels):
        """Test anchor-positive mask generation."""
        # Keras
        labels_tf = tf.constant(simple_labels)
        mask_keras = keras_get_anchor_positive_triplet_mask(labels_tf).numpy()
        
        # PyTorch
        labels_torch = torch.tensor(simple_labels)
        mask_torch = torch_get_anchor_positive_triplet_mask(labels_torch).numpy()
        
        np.testing.assert_array_equal(mask_keras, mask_torch)
    
    def test_anchor_negative_mask(self, keras_float32_policy, simple_labels):
        """Test anchor-negative mask generation."""
        # Keras
        labels_tf = tf.constant(simple_labels)
        mask_keras = keras_get_anchor_negative_triplet_mask(labels_tf).numpy()
        
        # PyTorch
        labels_torch = torch.tensor(simple_labels)
        mask_torch = torch_get_anchor_negative_triplet_mask(labels_torch).numpy()
        
        np.testing.assert_array_equal(mask_keras, mask_torch)
    
    def test_triplet_mask(self, keras_float32_policy, simple_labels):
        """Test 3D triplet mask generation."""
        # Keras
        labels_tf = tf.constant(simple_labels)
        mask_keras = keras_get_triplet_mask(labels_tf).numpy()
        
        # PyTorch
        labels_torch = torch.tensor(simple_labels)
        mask_torch = torch_get_triplet_mask(labels_torch).numpy()
        
        np.testing.assert_array_equal(mask_keras, mask_torch)
    
    def test_triplet_mask_random_labels(self, keras_float32_policy, random_labels):
        """Test 3D triplet mask with random labels."""
        # Keras
        labels_tf = tf.constant(random_labels)
        mask_keras = keras_get_triplet_mask(labels_tf).numpy()
        
        # PyTorch
        labels_torch = torch.tensor(random_labels)
        mask_torch = torch_get_triplet_mask(labels_torch).numpy()
        
        np.testing.assert_array_equal(mask_keras, mask_torch)


@requires_both
class TestBatchHardTripletLoss:
    """Test batch hard triplet loss matches between Keras and PyTorch."""
    
    def test_batch_hard_triplet_loss_simple(
        self, keras_float32_policy, simple_embeddings, simple_labels
    ):
        """Test batch hard triplet loss with simple embeddings."""
        margin = 0.5
        
        # Keras
        embeddings_tf = tf.constant(simple_embeddings)
        labels_tf = tf.constant(simple_labels)
        loss_keras = keras_batch_hard_triplet_loss(
            labels_tf, embeddings_tf, margin, squared=False
        ).numpy()
        
        # PyTorch
        embeddings_torch = torch.tensor(simple_embeddings)
        labels_torch = torch.tensor(simple_labels)
        loss_torch = torch_batch_hard_triplet_loss(
            labels_torch, embeddings_torch, margin, squared=False
        ).numpy()
        
        np.testing.assert_allclose(loss_keras, loss_torch, rtol=1e-5, atol=1e-5)
    
    def test_batch_hard_triplet_loss_random(
        self, keras_float32_policy, random_embeddings, random_labels
    ):
        """Test batch hard triplet loss with random embeddings."""
        margin = 1.0
        
        # Keras
        embeddings_tf = tf.constant(random_embeddings)
        labels_tf = tf.constant(random_labels)
        loss_keras = keras_batch_hard_triplet_loss(
            labels_tf, embeddings_tf, margin, squared=False
        ).numpy()
        
        # PyTorch
        embeddings_torch = torch.tensor(random_embeddings)
        labels_torch = torch.tensor(random_labels)
        loss_torch = torch_batch_hard_triplet_loss(
            labels_torch, embeddings_torch, margin, squared=False
        ).numpy()
        
        np.testing.assert_allclose(loss_keras, loss_torch, rtol=1e-4, atol=1e-4)
    
    def test_batch_hard_triplet_loss_squared(
        self, keras_float32_policy, random_embeddings, random_labels
    ):
        """Test batch hard triplet loss with squared distances."""
        margin = 1.0
        
        # Keras
        embeddings_tf = tf.constant(random_embeddings)
        labels_tf = tf.constant(random_labels)
        loss_keras = keras_batch_hard_triplet_loss(
            labels_tf, embeddings_tf, margin, squared=True
        ).numpy()
        
        # PyTorch
        embeddings_torch = torch.tensor(random_embeddings)
        labels_torch = torch.tensor(random_labels)
        loss_torch = torch_batch_hard_triplet_loss(
            labels_torch, embeddings_torch, margin, squared=True
        ).numpy()
        
        np.testing.assert_allclose(loss_keras, loss_torch, rtol=1e-4, atol=1e-4)
    
    def test_batch_hard_triplet_loss_normalized(
        self, keras_float32_policy, random_embeddings, random_labels
    ):
        """Test batch hard triplet loss with normalization."""
        margin = 0.5
        
        # Keras
        embeddings_tf = tf.constant(random_embeddings)
        labels_tf = tf.constant(random_labels)
        loss_keras = keras_batch_hard_triplet_loss(
            labels_tf, embeddings_tf, margin, squared=False, normalize=True
        ).numpy()
        
        # PyTorch
        embeddings_torch = torch.tensor(random_embeddings)
        labels_torch = torch.tensor(random_labels)
        loss_torch = torch_batch_hard_triplet_loss(
            labels_torch, embeddings_torch, margin, squared=False, normalize=True
        ).numpy()
        
        np.testing.assert_allclose(loss_keras, loss_torch, rtol=1e-4, atol=1e-4)
    
    def test_batch_hard_triplet_loss_zero_margin(
        self, keras_float32_policy, simple_embeddings, simple_labels
    ):
        """Test batch hard triplet loss with zero margin."""
        margin = 0.0
        
        # Keras
        embeddings_tf = tf.constant(simple_embeddings)
        labels_tf = tf.constant(simple_labels)
        loss_keras = keras_batch_hard_triplet_loss(
            labels_tf, embeddings_tf, margin, squared=False
        ).numpy()
        
        # PyTorch
        embeddings_torch = torch.tensor(simple_embeddings)
        labels_torch = torch.tensor(simple_labels)
        loss_torch = torch_batch_hard_triplet_loss(
            labels_torch, embeddings_torch, margin, squared=False
        ).numpy()
        
        np.testing.assert_allclose(loss_keras, loss_torch, rtol=1e-5, atol=1e-5)


@requires_both
class TestBatchAllTripletLoss:
    """Test batch all triplet loss matches between Keras and PyTorch."""
    
    def test_batch_all_triplet_loss_simple(
        self, keras_float32_policy, simple_embeddings, simple_labels
    ):
        """Test batch all triplet loss with simple embeddings."""
        margin = 0.5
        
        # Keras
        embeddings_tf = tf.constant(simple_embeddings)
        labels_tf = tf.constant(simple_labels)
        loss_keras, frac_keras = keras_batch_all_triplet_loss(
            labels_tf, embeddings_tf, margin, squared=False, remove_negative=False
        )
        loss_keras = loss_keras.numpy()
        frac_keras = frac_keras.numpy()
        
        # PyTorch
        embeddings_torch = torch.tensor(simple_embeddings)
        labels_torch = torch.tensor(simple_labels)
        loss_torch, frac_torch = torch_batch_all_triplet_loss(
            labels_torch, embeddings_torch, margin, squared=False, remove_negative=False
        )
        loss_torch = loss_torch.numpy()
        frac_torch = frac_torch.numpy()
        
        np.testing.assert_allclose(loss_keras, loss_torch, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(frac_keras, frac_torch, rtol=1e-5, atol=1e-5)
    
    def test_batch_all_triplet_loss_random(
        self, keras_float32_policy, random_embeddings, random_labels
    ):
        """Test batch all triplet loss with random embeddings."""
        margin = 1.0
        
        # Keras
        embeddings_tf = tf.constant(random_embeddings)
        labels_tf = tf.constant(random_labels)
        loss_keras, frac_keras = keras_batch_all_triplet_loss(
            labels_tf, embeddings_tf, margin, squared=False, remove_negative=False
        )
        loss_keras = loss_keras.numpy()
        frac_keras = frac_keras.numpy()
        
        # PyTorch
        embeddings_torch = torch.tensor(random_embeddings)
        labels_torch = torch.tensor(random_labels)
        loss_torch, frac_torch = torch_batch_all_triplet_loss(
            labels_torch, embeddings_torch, margin, squared=False, remove_negative=False
        )
        loss_torch = loss_torch.numpy()
        frac_torch = frac_torch.numpy()
        
        np.testing.assert_allclose(loss_keras, loss_torch, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(frac_keras, frac_torch, rtol=1e-4, atol=1e-4)
    
    def test_batch_all_triplet_loss_remove_negative(
        self, keras_float32_policy, random_embeddings, random_labels
    ):
        """Test batch all triplet loss with remove_negative=True."""
        margin = 1.0
        
        # Keras
        embeddings_tf = tf.constant(random_embeddings)
        labels_tf = tf.constant(random_labels)
        loss_keras, frac_keras = keras_batch_all_triplet_loss(
            labels_tf, embeddings_tf, margin, squared=False, remove_negative=True
        )
        loss_keras = loss_keras.numpy()
        frac_keras = frac_keras.numpy()
        
        # PyTorch
        embeddings_torch = torch.tensor(random_embeddings)
        labels_torch = torch.tensor(random_labels)
        loss_torch, frac_torch = torch_batch_all_triplet_loss(
            labels_torch, embeddings_torch, margin, squared=False, remove_negative=True
        )
        loss_torch = loss_torch.numpy()
        frac_torch = frac_torch.numpy()
        
        np.testing.assert_allclose(loss_keras, loss_torch, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(frac_keras, frac_torch, rtol=1e-4, atol=1e-4)


@requires_both
class TestBatchDistanceLoss:
    """Test batch distance loss matches between Keras and PyTorch."""
    
    def test_batch_distance_loss_simple(
        self, keras_float32_policy, simple_embeddings, simple_labels
    ):
        """Test batch distance loss with simple embeddings."""
        # Keras
        embeddings_tf = tf.constant(simple_embeddings)
        labels_tf = tf.constant(simple_labels)
        loss_keras, frac_keras = keras_batch_distance_loss(
            labels_tf, embeddings_tf, squared=False
        )
        loss_keras = loss_keras.numpy()
        frac_keras = frac_keras.numpy()
        
        # PyTorch
        embeddings_torch = torch.tensor(simple_embeddings)
        labels_torch = torch.tensor(simple_labels)
        loss_torch, frac_torch = torch_batch_distance_loss(
            labels_torch, embeddings_torch, squared=False
        )
        loss_torch = loss_torch.numpy()
        frac_torch = frac_torch.numpy()
        
        np.testing.assert_allclose(loss_keras, loss_torch, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(frac_keras, frac_torch, rtol=1e-5, atol=1e-5)
    
    def test_batch_distance_loss_random(
        self, keras_float32_policy, random_embeddings, random_labels
    ):
        """Test batch distance loss with random embeddings."""
        # Keras
        embeddings_tf = tf.constant(random_embeddings)
        labels_tf = tf.constant(random_labels)
        loss_keras, frac_keras = keras_batch_distance_loss(
            labels_tf, embeddings_tf, squared=False
        )
        loss_keras = loss_keras.numpy()
        frac_keras = frac_keras.numpy()
        
        # PyTorch
        embeddings_torch = torch.tensor(random_embeddings)
        labels_torch = torch.tensor(random_labels)
        loss_torch, frac_torch = torch_batch_distance_loss(
            labels_torch, embeddings_torch, squared=False
        )
        loss_torch = loss_torch.numpy()
        frac_torch = frac_torch.numpy()
        
        np.testing.assert_allclose(loss_keras, loss_torch, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(frac_keras, frac_torch, rtol=1e-4, atol=1e-4)


@requires_both
class TestHardNegativeMining:
    """Test hard negative mining matches between Keras and PyTorch."""
    
    def test_hard_negative_mining_simple(
        self, keras_float32_policy, simple_embeddings, simple_labels
    ):
        """Test hard negative mining with simple embeddings."""
        margin = 0.5
        
        # Keras
        embeddings_tf = tf.constant(simple_embeddings)
        labels_tf = tf.constant(simple_labels)
        loss_keras = keras_hard_negative_triplet_mining(
            labels_tf, embeddings_tf, margin, squared=False
        ).numpy()
        
        # PyTorch
        embeddings_torch = torch.tensor(simple_embeddings)
        labels_torch = torch.tensor(simple_labels)
        loss_torch = torch_hard_negative_triplet_mining(
            labels_torch, embeddings_torch, margin, squared=False
        ).numpy()
        
        np.testing.assert_allclose(loss_keras, loss_torch, rtol=1e-5, atol=1e-5)
    
    def test_hard_negative_mining_random(
        self, keras_float32_policy, random_embeddings, random_labels
    ):
        """Test hard negative mining with random embeddings."""
        margin = 1.0
        
        # Keras
        embeddings_tf = tf.constant(random_embeddings)
        labels_tf = tf.constant(random_labels)
        loss_keras = keras_hard_negative_triplet_mining(
            labels_tf, embeddings_tf, margin, squared=False
        ).numpy()
        
        # PyTorch
        embeddings_torch = torch.tensor(random_embeddings)
        labels_torch = torch.tensor(random_labels)
        loss_torch = torch_hard_negative_triplet_mining(
            labels_torch, embeddings_torch, margin, squared=False
        ).numpy()
        
        np.testing.assert_allclose(loss_keras, loss_torch, rtol=1e-4, atol=1e-4)


@requires_both
class TestGradientComputation:
    """Test that gradients can be computed and are similar."""
    
    def test_batch_hard_gradient(
        self, keras_float32_policy, random_embeddings, random_labels
    ):
        """Test gradient computation for batch hard triplet loss."""
        margin = 1.0
        
        # Keras gradient
        embeddings_tf = tf.Variable(random_embeddings)
        labels_tf = tf.constant(random_labels)
        
        with tf.GradientTape() as tape:
            loss_keras = keras_batch_hard_triplet_loss(
                labels_tf, embeddings_tf, margin, squared=False
            )
        grad_keras = tape.gradient(loss_keras, embeddings_tf).numpy()
        
        # PyTorch gradient
        embeddings_torch = torch.tensor(random_embeddings, requires_grad=True)
        labels_torch = torch.tensor(random_labels)
        
        loss_torch = torch_batch_hard_triplet_loss(
            labels_torch, embeddings_torch, margin, squared=False
        )
        loss_torch.backward()
        grad_torch = embeddings_torch.grad.numpy()
        
        # Gradients should be similar (not necessarily identical due to numerical differences)
        np.testing.assert_allclose(grad_keras, grad_torch, rtol=1e-3, atol=1e-3)
    
    def test_batch_all_gradient(
        self, keras_float32_policy, random_embeddings, random_labels
    ):
        """Test gradient computation for batch all triplet loss."""
        margin = 1.0
        
        # Keras gradient
        embeddings_tf = tf.Variable(random_embeddings)
        labels_tf = tf.constant(random_labels)
        
        with tf.GradientTape() as tape:
            loss_keras, _ = keras_batch_all_triplet_loss(
                labels_tf, embeddings_tf, margin, squared=False, remove_negative=False
            )
        grad_keras = tape.gradient(loss_keras, embeddings_tf).numpy()
        
        # PyTorch gradient
        embeddings_torch = torch.tensor(random_embeddings, requires_grad=True)
        labels_torch = torch.tensor(random_labels)
        
        loss_torch, _ = torch_batch_all_triplet_loss(
            labels_torch, embeddings_torch, margin, squared=False, remove_negative=False
        )
        loss_torch.backward()
        grad_torch = embeddings_torch.grad.numpy()
        
        # Gradients should be similar
        np.testing.assert_allclose(grad_keras, grad_torch, rtol=1e-3, atol=1e-3)


@requires_both
class TestEdgeCases:
    """Test edge cases for triplet loss functions."""
    
    def test_single_class_labels(self, keras_float32_policy):
        """Test with all labels being the same class."""
        embeddings_np = np.random.randn(8, 4).astype(np.float32)
        labels_np = np.zeros(8, dtype=np.int32)  # All same class
        margin = 1.0
        
        # Keras
        embeddings_tf = tf.constant(embeddings_np)
        labels_tf = tf.constant(labels_np)
        loss_keras = keras_batch_hard_triplet_loss(
            labels_tf, embeddings_tf, margin, squared=False
        ).numpy()
        
        # PyTorch
        embeddings_torch = torch.tensor(embeddings_np)
        labels_torch = torch.tensor(labels_np)
        loss_torch = torch_batch_hard_triplet_loss(
            labels_torch, embeddings_torch, margin, squared=False
        ).numpy()
        
        np.testing.assert_allclose(loss_keras, loss_torch, rtol=1e-5, atol=1e-5)
    
    def test_small_batch(self, keras_float32_policy):
        """Test with very small batch size."""
        embeddings_np = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
        labels_np = np.array([0, 1], dtype=np.int32)
        margin = 0.5
        
        # Keras
        embeddings_tf = tf.constant(embeddings_np)
        labels_tf = tf.constant(labels_np)
        loss_keras = keras_batch_hard_triplet_loss(
            labels_tf, embeddings_tf, margin, squared=False
        ).numpy()
        
        # PyTorch
        embeddings_torch = torch.tensor(embeddings_np)
        labels_torch = torch.tensor(labels_np)
        loss_torch = torch_batch_hard_triplet_loss(
            labels_torch, embeddings_torch, margin, squared=False
        ).numpy()
        
        np.testing.assert_allclose(loss_keras, loss_torch, rtol=1e-5, atol=1e-5)
    
    def test_large_embedding_dimension(self, keras_float32_policy):
        """Test with large embedding dimension."""
        np.random.seed(42)
        embeddings_np = np.random.randn(8, 128).astype(np.float32)
        labels_np = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
        margin = 1.0
        
        # Keras
        embeddings_tf = tf.constant(embeddings_np)
        labels_tf = tf.constant(labels_np)
        loss_keras = keras_batch_hard_triplet_loss(
            labels_tf, embeddings_tf, margin, squared=False
        ).numpy()
        
        # PyTorch
        embeddings_torch = torch.tensor(embeddings_np)
        labels_torch = torch.tensor(labels_np)
        loss_torch = torch_batch_hard_triplet_loss(
            labels_torch, embeddings_torch, margin, squared=False
        ).numpy()
        
        np.testing.assert_allclose(loss_keras, loss_torch, rtol=1e-4, atol=1e-4)
