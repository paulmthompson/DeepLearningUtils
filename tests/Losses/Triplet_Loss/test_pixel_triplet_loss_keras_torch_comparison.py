"""
Tests comparing Keras/TensorFlow and PyTorch implementations of pixel-based triplet loss.

This module verifies that the PyTorch PixelTripletLoss produces identical
results to the Keras/TensorFlow implementation.
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
    from src.DeepLearningUtils.Losses.Triplet_Loss.pixel_triplet_loss_keras import (
        PixelTripletLoss as KerasPixelTripletLoss,
        PixelTripletConfig as KerasPixelTripletConfig,
        create_pixel_triplet_loss as keras_create_pixel_triplet_loss,
        _get_anchor_positive_triplet_mask_exclude_background as keras_get_anchor_positive_mask_exclude_bg,
        _get_triplet_mask_exclude_background as keras_get_triplet_mask_exclude_bg,
    )

if torch_available:
    from src.DeepLearningUtils.Losses.Triplet_Loss.pixel_triplet_loss_torch import (
        PixelTripletLoss as TorchPixelTripletLoss,
        PixelTripletConfig as TorchPixelTripletConfig,
        create_pixel_triplet_loss as torch_create_pixel_triplet_loss,
        _get_anchor_positive_triplet_mask_exclude_background as torch_get_anchor_positive_mask_exclude_bg,
        _get_triplet_mask_exclude_background as torch_get_triplet_mask_exclude_bg,
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
def simple_pixel_embeddings():
    """Create simple pixel embeddings for testing.
    
    Shape: (batch_size=1, h=4, w=4, feature_dim=2)
    """
    np.random.seed(42)
    embeddings = np.random.randn(1, 4, 4, 2).astype(np.float32)
    return embeddings


@pytest.fixture
def simple_pixel_labels():
    """Create simple pixel labels for testing.
    
    Shape: (batch_size=1, h=4, w=4, num_classes=2)
    Binary masks where first channel is whisker 1, second is whisker 2.
    """
    labels = np.zeros((1, 4, 4, 2), dtype=np.float32)
    # Whisker 1: top-left 2x2
    labels[0, 0:2, 0:2, 0] = 1.0
    # Whisker 2: bottom-right 2x2
    labels[0, 2:4, 2:4, 1] = 1.0
    # Rest is background (all zeros)
    return labels


@pytest.fixture
def larger_pixel_embeddings():
    """Create larger pixel embeddings for more comprehensive testing.
    
    Shape: (batch_size=2, h=8, w=8, feature_dim=4)
    """
    np.random.seed(123)
    embeddings = np.random.randn(2, 8, 8, 4).astype(np.float32)
    return embeddings


@pytest.fixture
def larger_pixel_labels():
    """Create larger pixel labels with 3 whiskers.
    
    Shape: (batch_size=2, h=8, w=8, num_whiskers=3)
    """
    labels = np.zeros((2, 8, 8, 3), dtype=np.float32)
    
    for batch_idx in range(2):
        # Whisker 1: top-left region
        labels[batch_idx, 0:3, 0:3, 0] = 1.0
        # Whisker 2: top-right region
        labels[batch_idx, 0:3, 5:8, 1] = 1.0
        # Whisker 3: bottom-center region
        labels[batch_idx, 5:8, 3:6, 2] = 1.0
    
    return labels


@pytest.fixture
def flat_embeddings_and_labels():
    """Create flattened embeddings and labels for direct loss function testing."""
    np.random.seed(42)
    embeddings = np.random.randn(20, 4).astype(np.float32)
    # Labels: 0=background, 1=whisker1, 2=whisker2
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0, 1, 2, 2], dtype=np.int32)
    return embeddings, labels


@requires_both
class TestBackgroundExclusionMasks:
    """Test that background exclusion masks match between implementations."""
    
    def test_anchor_positive_mask_exclude_background(self, keras_float32_policy):
        """Test anchor-positive mask that excludes background."""
        # Labels: 0=background, 1,2=whiskers
        labels_np = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        
        # Keras
        labels_tf = tf.constant(labels_np)
        mask_keras = keras_get_anchor_positive_mask_exclude_bg(labels_tf).numpy()
        
        # PyTorch
        labels_torch = torch.tensor(labels_np)
        mask_torch = torch_get_anchor_positive_mask_exclude_bg(labels_torch).numpy()
        
        np.testing.assert_array_equal(mask_keras, mask_torch)
        
        # Verify that background (class 0) anchors have no valid positives
        # Background indices are 0, 1
        assert not mask_keras[0].any()  # No valid positives for background anchor
        assert not mask_keras[1].any()  # No valid positives for background anchor
        
        # Whisker anchors should have valid positives
        assert mask_keras[2, 3]  # Whisker 1 at index 2 has positive at index 3
        assert mask_keras[3, 2]  # Whisker 1 at index 3 has positive at index 2
        assert mask_keras[4, 5]  # Whisker 2 at index 4 has positive at index 5
        assert mask_keras[5, 4]  # Whisker 2 at index 5 has positive at index 4
    
    def test_triplet_mask_exclude_background(self, keras_float32_policy):
        """Test 3D triplet mask that excludes background positives."""
        labels_np = np.array([0, 1, 1, 2], dtype=np.int32)
        
        # Keras
        labels_tf = tf.constant(labels_np)
        mask_keras = keras_get_triplet_mask_exclude_bg(labels_tf).numpy()
        
        # PyTorch
        labels_torch = torch.tensor(labels_np)
        mask_torch = torch_get_triplet_mask_exclude_bg(labels_torch).numpy()
        
        np.testing.assert_array_equal(mask_keras, mask_torch)
        
        # Verify triplet mask properties
        # Valid triplet: (anchor=1, positive=2, negative=0) - whisker 1 anchor, whisker 1 positive, background negative
        assert mask_keras[1, 2, 0]  # Valid: same class (1), different negative class (0), anchor is not background
        assert mask_keras[1, 2, 3]  # Valid: same class (1), different negative class (2)
        
        # Invalid: background anchor
        assert not mask_keras[0, :, :].any()  # Background anchor should have no valid triplets


@requires_both
class TestDistanceMetrics:
    """Test that different distance metrics produce matching results."""
    
    @pytest.fixture
    def embeddings_and_labels(self, flat_embeddings_and_labels):
        return flat_embeddings_and_labels
    
    @pytest.mark.parametrize("distance_metric", ["euclidean", "cosine", "manhattan"])
    def test_distance_metric_matching(
        self, keras_float32_policy, embeddings_and_labels, distance_metric
    ):
        """Test that distance metric implementations match."""
        embeddings, labels = embeddings_and_labels
        
        # Create configs with specific distance metric
        config_keras = KerasPixelTripletConfig(
            margin=1.0,
            distance_metric=distance_metric,
            triplet_strategy="hard",
            use_balanced_sampling=False,
        )
        config_torch = TorchPixelTripletConfig(
            margin=1.0,
            distance_metric=distance_metric,
            triplet_strategy="hard",
            use_balanced_sampling=False,
        )
        
        loss_keras = KerasPixelTripletLoss(config=config_keras)
        loss_torch = TorchPixelTripletLoss(config=config_torch)
        
        # Test custom distance computation
        emb_tf = tf.constant(embeddings)
        emb_torch = torch.tensor(embeddings)
        
        dist_keras = loss_keras._compute_pairwise_distances(emb_tf).numpy()
        dist_torch = loss_torch._compute_pairwise_distances(emb_torch).numpy()
        
        np.testing.assert_allclose(dist_keras, dist_torch, rtol=1e-4, atol=1e-4)


@requires_both
class TestBatchHardTripletLossCustom:
    """Test custom batch hard triplet loss implementations match."""
    
    def test_batch_hard_triplet_loss_euclidean(
        self, keras_float32_policy, flat_embeddings_and_labels
    ):
        """Test batch hard triplet loss with euclidean distance."""
        embeddings, labels = flat_embeddings_and_labels
        
        config_keras = KerasPixelTripletConfig(
            margin=1.0,
            distance_metric="euclidean",
            triplet_strategy="hard",
        )
        config_torch = TorchPixelTripletConfig(
            margin=1.0,
            distance_metric="euclidean",
            triplet_strategy="hard",
        )
        
        loss_keras = KerasPixelTripletLoss(config=config_keras)
        loss_torch = TorchPixelTripletLoss(config=config_torch)
        
        emb_tf = tf.constant(embeddings)
        labels_tf = tf.constant(labels)
        emb_torch = torch.tensor(embeddings)
        labels_torch = torch.tensor(labels)
        
        result_keras = loss_keras._batch_hard_triplet_loss_custom(labels_tf, emb_tf).numpy()
        result_torch = loss_torch._batch_hard_triplet_loss_custom(labels_torch, emb_torch).numpy()
        
        np.testing.assert_allclose(result_keras, result_torch, rtol=1e-4, atol=1e-4)
    
    def test_batch_hard_triplet_loss_cosine(
        self, keras_float32_policy, flat_embeddings_and_labels
    ):
        """Test batch hard triplet loss with cosine distance."""
        embeddings, labels = flat_embeddings_and_labels
        
        config_keras = KerasPixelTripletConfig(
            margin=0.5,
            distance_metric="cosine",
            triplet_strategy="hard",
        )
        config_torch = TorchPixelTripletConfig(
            margin=0.5,
            distance_metric="cosine",
            triplet_strategy="hard",
        )
        
        loss_keras = KerasPixelTripletLoss(config=config_keras)
        loss_torch = TorchPixelTripletLoss(config=config_torch)
        
        emb_tf = tf.constant(embeddings)
        labels_tf = tf.constant(labels)
        emb_torch = torch.tensor(embeddings)
        labels_torch = torch.tensor(labels)
        
        result_keras = loss_keras._batch_hard_triplet_loss_custom(labels_tf, emb_tf).numpy()
        result_torch = loss_torch._batch_hard_triplet_loss_custom(labels_torch, emb_torch).numpy()
        
        np.testing.assert_allclose(result_keras, result_torch, rtol=1e-4, atol=1e-4)


@requires_both
class TestBatchAllTripletLossCustom:
    """Test custom batch all triplet loss implementations match."""
    
    def test_batch_all_triplet_loss_euclidean(
        self, keras_float32_policy, flat_embeddings_and_labels
    ):
        """Test batch all triplet loss with euclidean distance."""
        embeddings, labels = flat_embeddings_and_labels
        
        config_keras = KerasPixelTripletConfig(
            margin=1.0,
            distance_metric="euclidean",
            triplet_strategy="all",
            remove_easy_triplets=False,
        )
        config_torch = TorchPixelTripletConfig(
            margin=1.0,
            distance_metric="euclidean",
            triplet_strategy="all",
            remove_easy_triplets=False,
        )
        
        loss_keras = KerasPixelTripletLoss(config=config_keras)
        loss_torch = TorchPixelTripletLoss(config=config_torch)
        
        emb_tf = tf.constant(embeddings)
        labels_tf = tf.constant(labels)
        emb_torch = torch.tensor(embeddings)
        labels_torch = torch.tensor(labels)
        
        result_keras = loss_keras._batch_all_triplet_loss_custom(labels_tf, emb_tf).numpy()
        result_torch = loss_torch._batch_all_triplet_loss_custom(labels_torch, emb_torch).numpy()
        
        np.testing.assert_allclose(result_keras, result_torch, rtol=1e-4, atol=1e-4)
    
    def test_batch_all_triplet_loss_remove_easy(
        self, keras_float32_policy, flat_embeddings_and_labels
    ):
        """Test batch all triplet loss with remove_easy_triplets=True."""
        embeddings, labels = flat_embeddings_and_labels
        
        config_keras = KerasPixelTripletConfig(
            margin=1.0,
            distance_metric="euclidean",
            triplet_strategy="all",
            remove_easy_triplets=True,
        )
        config_torch = TorchPixelTripletConfig(
            margin=1.0,
            distance_metric="euclidean",
            triplet_strategy="all",
            remove_easy_triplets=True,
        )
        
        loss_keras = KerasPixelTripletLoss(config=config_keras)
        loss_torch = TorchPixelTripletLoss(config=config_torch)
        
        emb_tf = tf.constant(embeddings)
        labels_tf = tf.constant(labels)
        emb_torch = torch.tensor(embeddings)
        labels_torch = torch.tensor(labels)
        
        result_keras = loss_keras._batch_all_triplet_loss_custom(labels_tf, emb_tf).numpy()
        result_torch = loss_torch._batch_all_triplet_loss_custom(labels_torch, emb_torch).numpy()
        
        np.testing.assert_allclose(result_keras, result_torch, rtol=1e-4, atol=1e-4)


@requires_both
class TestLabelsToClasses:
    """Test label conversion matches between implementations."""
    
    def test_labels_to_classes(self, keras_float32_policy, simple_pixel_labels):
        """Test conversion of multi-channel labels to class indices."""
        config_keras = KerasPixelTripletConfig()
        config_torch = TorchPixelTripletConfig()
        
        loss_keras = KerasPixelTripletLoss(config=config_keras)
        loss_torch = TorchPixelTripletLoss(config=config_torch)
        
        labels_tf = tf.constant(simple_pixel_labels)
        labels_torch = torch.tensor(simple_pixel_labels)
        
        classes_keras = loss_keras._labels_to_classes(labels_tf).numpy()
        classes_torch = loss_torch._labels_to_classes(labels_torch).numpy()
        
        np.testing.assert_array_equal(classes_keras, classes_torch)
        
        # Verify class values
        # Background should be 0, whisker 1 should be 1, whisker 2 should be 2
        assert classes_keras[0, 0, 0] == 1  # Top-left is whisker 1
        assert classes_keras[0, 2, 2] == 2  # Bottom-right is whisker 2
        assert classes_keras[0, 0, 3] == 0  # Top-right corner is background


@requires_both
class TestPixelTripletLossEndToEnd:
    """End-to-end tests for the full PixelTripletLoss forward pass."""
    
    def test_forward_pass_hard_strategy(
        self, keras_float32_policy, simple_pixel_embeddings, simple_pixel_labels
    ):
        """Test forward pass with hard triplet strategy."""
        # Set seed for reproducible sampling
        np.random.seed(42)
        tf.random.set_seed(42)
        torch.manual_seed(42)
        
        config_keras = KerasPixelTripletConfig(
            margin=1.0,
            triplet_strategy="hard",
            use_balanced_sampling=False,
            background_pixels=5,
            whisker_pixels=5,
        )
        config_torch = TorchPixelTripletConfig(
            margin=1.0,
            triplet_strategy="hard",
            use_balanced_sampling=False,
            background_pixels=5,
            whisker_pixels=5,
        )
        
        loss_keras = KerasPixelTripletLoss(config=config_keras)
        loss_torch = TorchPixelTripletLoss(config=config_torch)
        
        y_true_tf = tf.constant(simple_pixel_labels)
        y_pred_tf = tf.constant(simple_pixel_embeddings)
        y_true_torch = torch.tensor(simple_pixel_labels)
        y_pred_torch = torch.tensor(simple_pixel_embeddings)
        
        # Note: Due to random sampling, exact values may differ
        # Test that both produce valid losses
        result_keras = loss_keras(y_true_tf, y_pred_tf).numpy()
        result_torch = loss_torch(y_true_torch, y_pred_torch).numpy()
        
        # Both should produce non-negative losses
        assert result_keras >= 0
        assert result_torch >= 0
    
    def test_forward_pass_all_strategy(
        self, keras_float32_policy, simple_pixel_embeddings, simple_pixel_labels
    ):
        """Test forward pass with all triplet strategy."""
        np.random.seed(42)
        tf.random.set_seed(42)
        torch.manual_seed(42)
        
        config_keras = KerasPixelTripletConfig(
            margin=1.0,
            triplet_strategy="all",
            use_balanced_sampling=False,
            background_pixels=5,
            whisker_pixels=5,
        )
        config_torch = TorchPixelTripletConfig(
            margin=1.0,
            triplet_strategy="all",
            use_balanced_sampling=False,
            background_pixels=5,
            whisker_pixels=5,
        )
        
        loss_keras = KerasPixelTripletLoss(config=config_keras)
        loss_torch = TorchPixelTripletLoss(config=config_torch)
        
        y_true_tf = tf.constant(simple_pixel_labels)
        y_pred_tf = tf.constant(simple_pixel_embeddings)
        y_true_torch = torch.tensor(simple_pixel_labels)
        y_pred_torch = torch.tensor(simple_pixel_embeddings)
        
        result_keras = loss_keras(y_true_tf, y_pred_tf).numpy()
        result_torch = loss_torch(y_true_torch, y_pred_torch).numpy()
        
        assert result_keras >= 0
        assert result_torch >= 0


@requires_both
class TestExactMode:
    """Test exact (loop-based) computation mode."""
    
    def test_exact_hard_triplet_loss(
        self, keras_float32_policy, flat_embeddings_and_labels
    ):
        """Test exact hard triplet loss computation."""
        embeddings, labels = flat_embeddings_and_labels
        
        config_keras = KerasPixelTripletConfig(
            margin=1.0,
            distance_metric="euclidean",
            triplet_strategy="hard",
            use_exact=True,
            batch_size_for_exact=5,
        )
        config_torch = TorchPixelTripletConfig(
            margin=1.0,
            distance_metric="euclidean",
            triplet_strategy="hard",
            use_exact=True,
            batch_size_for_exact=5,
        )
        
        loss_keras = KerasPixelTripletLoss(config=config_keras)
        loss_torch = TorchPixelTripletLoss(config=config_torch)
        
        emb_tf = tf.constant(embeddings)
        labels_tf = tf.constant(labels)
        emb_torch = torch.tensor(embeddings)
        labels_torch = torch.tensor(labels)
        
        result_keras = loss_keras._batch_hard_triplet_loss_exact(labels_tf, emb_tf).numpy()
        result_torch = loss_torch._batch_hard_triplet_loss_exact(labels_torch, emb_torch).numpy()
        
        np.testing.assert_allclose(result_keras, result_torch, rtol=1e-4, atol=1e-4)
    
    def test_exact_all_triplet_loss_no_remove_easy(
        self, keras_float32_policy, flat_embeddings_and_labels
    ):
        """Test exact all triplet loss without removing easy triplets."""
        embeddings, labels = flat_embeddings_and_labels
        
        config_keras = KerasPixelTripletConfig(
            margin=1.0,
            distance_metric="euclidean",
            triplet_strategy="all",
            use_exact=True,
            batch_size_for_exact=5,
            remove_easy_triplets=False,
        )
        config_torch = TorchPixelTripletConfig(
            margin=1.0,
            distance_metric="euclidean",
            triplet_strategy="all",
            use_exact=True,
            batch_size_for_exact=5,
            remove_easy_triplets=False,
        )
        
        loss_keras = KerasPixelTripletLoss(config=config_keras)
        loss_torch = TorchPixelTripletLoss(config=config_torch)
        
        emb_tf = tf.constant(embeddings)
        labels_tf = tf.constant(labels)
        emb_torch = torch.tensor(embeddings)
        labels_torch = torch.tensor(labels)
        
        result_keras = loss_keras._batch_all_triplet_loss_exact(labels_tf, emb_tf).numpy()
        result_torch = loss_torch._batch_all_triplet_loss_exact(labels_torch, emb_torch).numpy()
        
        np.testing.assert_allclose(result_keras, result_torch, rtol=1e-4, atol=1e-4)


@requires_both
class TestCreatePixelTripletLoss:
    """Test the convenience function for creating pixel triplet loss."""
    
    def test_create_with_defaults(self, keras_float32_policy):
        """Test creating loss with default parameters."""
        loss_keras = keras_create_pixel_triplet_loss()
        loss_torch = torch_create_pixel_triplet_loss()
        
        assert loss_keras.config.margin == loss_torch.config.margin
        assert loss_keras.config.distance_metric == loss_torch.config.distance_metric
        assert loss_keras.config.triplet_strategy == loss_torch.config.triplet_strategy
    
    def test_create_with_custom_params(self, keras_float32_policy):
        """Test creating loss with custom parameters."""
        loss_keras = keras_create_pixel_triplet_loss(
            margin=2.0,
            distance_metric="cosine",
            triplet_strategy="all",
            remove_easy_triplets=True,
        )
        loss_torch = torch_create_pixel_triplet_loss(
            margin=2.0,
            distance_metric="cosine",
            triplet_strategy="all",
            remove_easy_triplets=True,
        )
        
        assert loss_keras.config.margin == 2.0
        assert loss_torch.config.margin == 2.0
        assert loss_keras.config.distance_metric == "cosine"
        assert loss_torch.config.distance_metric == "cosine"


@requires_both
class TestResizeOperations:
    """Test resize operations match between implementations."""
    
    def test_resize_embeddings(self, keras_float32_policy):
        """Test embedding resize operation."""
        np.random.seed(42)
        embeddings = np.random.randn(2, 4, 4, 8).astype(np.float32)
        
        config_keras = KerasPixelTripletConfig()
        config_torch = TorchPixelTripletConfig()
        
        loss_keras = KerasPixelTripletLoss(config=config_keras)
        loss_torch = TorchPixelTripletLoss(config=config_torch)
        
        emb_tf = tf.constant(embeddings)
        emb_torch = torch.tensor(embeddings)
        
        # Resize from 4x4 to 8x8
        resized_keras = loss_keras._resize_embeddings(emb_tf, 8, 8).numpy()
        resized_torch = loss_torch._resize_embeddings(emb_torch, 8, 8).numpy()
        
        np.testing.assert_allclose(resized_keras, resized_torch, rtol=1e-5, atol=1e-5)
        
        # Check shape
        assert resized_keras.shape == (2, 8, 8, 8)
        assert resized_torch.shape == (2, 8, 8, 8)
    
    def test_resize_embeddings_no_op(self, keras_float32_policy):
        """Test that resize is a no-op when dimensions match."""
        np.random.seed(42)
        embeddings = np.random.randn(2, 8, 8, 4).astype(np.float32)
        
        config_keras = KerasPixelTripletConfig()
        config_torch = TorchPixelTripletConfig()
        
        loss_keras = KerasPixelTripletLoss(config=config_keras)
        loss_torch = TorchPixelTripletLoss(config=config_torch)
        
        emb_tf = tf.constant(embeddings)
        emb_torch = torch.tensor(embeddings)
        
        # Resize to same dimensions (should be no-op)
        resized_keras = loss_keras._resize_embeddings(emb_tf, 8, 8).numpy()
        resized_torch = loss_torch._resize_embeddings(emb_torch, 8, 8).numpy()
        
        np.testing.assert_allclose(resized_keras, embeddings, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(resized_torch, embeddings, rtol=1e-5, atol=1e-5)


@requires_both
class TestGradientComputation:
    """Test that gradients can be computed for both implementations."""
    
    def test_keras_gradient(self, keras_float32_policy, simple_pixel_embeddings, simple_pixel_labels):
        """Test gradient computation in Keras."""
        config = KerasPixelTripletConfig(
            margin=1.0,
            triplet_strategy="hard",
            use_balanced_sampling=False,
            background_pixels=5,
            whisker_pixels=5,
        )
        loss_fn = KerasPixelTripletLoss(config=config)
        
        y_pred = tf.Variable(simple_pixel_embeddings)
        y_true = tf.constant(simple_pixel_labels)
        
        with tf.GradientTape() as tape:
            loss = loss_fn(y_true, y_pred)
        
        grad = tape.gradient(loss, y_pred)
        
        assert grad is not None
        assert grad.shape == simple_pixel_embeddings.shape
    
    def test_pytorch_gradient(self, simple_pixel_embeddings, simple_pixel_labels):
        """Test gradient computation in PyTorch."""
        config = TorchPixelTripletConfig(
            margin=1.0,
            triplet_strategy="hard",
            use_balanced_sampling=False,
            background_pixels=5,
            whisker_pixels=5,
        )
        loss_fn = TorchPixelTripletLoss(config=config)
        
        y_pred = torch.tensor(simple_pixel_embeddings, requires_grad=True)
        y_true = torch.tensor(simple_pixel_labels)
        
        loss = loss_fn(y_true, y_pred)
        loss.backward()
        
        assert y_pred.grad is not None
        assert y_pred.grad.shape == simple_pixel_embeddings.shape


@requires_both
class TestConfigValidation:
    """Test that configuration validation matches between implementations."""
    
    def test_invalid_margin(self, keras_float32_policy):
        """Test that invalid margin raises error in both implementations."""
        with pytest.raises(ValueError):
            KerasPixelTripletConfig(margin=-1.0)
        
        with pytest.raises(ValueError):
            TorchPixelTripletConfig(margin=-1.0)
    
    def test_invalid_distance_metric(self, keras_float32_policy):
        """Test that invalid distance metric raises error."""
        with pytest.raises(ValueError):
            KerasPixelTripletConfig(distance_metric="invalid")
        
        with pytest.raises(ValueError):
            TorchPixelTripletConfig(distance_metric="invalid")
    
    def test_invalid_triplet_strategy(self, keras_float32_policy):
        """Test that invalid triplet strategy raises error."""
        with pytest.raises(ValueError):
            KerasPixelTripletConfig(triplet_strategy="invalid")
        
        with pytest.raises(ValueError):
            TorchPixelTripletConfig(triplet_strategy="invalid")
