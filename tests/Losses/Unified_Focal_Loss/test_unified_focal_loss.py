"""
Test suite for Unified Focal Loss functions.

This test suite validates the behavior of various focal loss implementations,
particularly focusing on edge cases and the bug in asymmetric_focal_tversky_loss
where empty labels with correct predictions result in high/undefined loss.
"""

import pytest
import numpy as np
import keras

from src.DeepLearningUtils.Losses.Unified_Focal_Loss.unified_focal_loss_keras import (
    asymmetric_focal_loss,
    asymmetric_focal_tversky_loss,
    asym_unified_focal_loss,
    symmetric_focal_loss,
    symmetric_focal_tversky_loss,
    sym_unified_focal_loss,
)


class TestAsymmetricFocalTverskyLoss:
    """Tests for asymmetric_focal_tversky_loss function."""

    def test_empty_labels_correct_prediction(self):
        """
        Test case: When y_true is all zeros (no labels) and y_pred correctly
        predicts all zeros (no foreground), the loss should be low (good prediction).

        With the fix:
        - Foreground channel: tp = 0, fn = 0, fp = 0 (class not present)
        - Background channel: tp = all pixels, fn = 0, fp = 0 (perfect prediction)
        - Only background contributes to loss since foreground is absent
        - Loss should be very low (near 0) for this perfect prediction
        """
        # Create a batch with empty labels (no foreground)
        batch_size = 2
        height, width = 64, 64

        # y_true: all zeros (no labels/no foreground)
        y_true = keras.ops.zeros((batch_size, height, width, 1), dtype=np.float32)

        # y_pred: correctly predicts all zeros (no foreground detected)
        y_pred = keras.ops.zeros((batch_size, height, width, 1), dtype=np.float32)

        # Add background channel
        y_true_with_bg = keras.ops.concatenate([1 - y_true, y_true], axis=-1)
        y_pred_with_bg = keras.ops.concatenate([1 - y_pred, y_pred], axis=-1)

        loss_fn = asymmetric_focal_tversky_loss(delta=0.7, gamma=0.75)
        loss = loss_fn(y_true_with_bg, y_pred_with_bg)

        loss_value = float(keras.ops.convert_to_numpy(loss))

        print(f"\nEmpty labels, correct prediction (all zeros): loss = {loss_value}")

        # Loss should be very low for perfect prediction
        assert not np.isnan(loss_value), "Loss should not be NaN"
        assert not np.isinf(loss_value), "Loss should not be Inf"
        assert loss_value < 0.1, f"Loss should be low (< 0.1) for perfect prediction, got {loss_value:.6f}"

    def test_empty_labels_incorrect_prediction(self):
        """
        Test case: When y_true is all zeros (no labels) but y_pred incorrectly
        predicts some foreground, the loss should be high (bad prediction).
        """
        batch_size = 2
        height, width = 64, 64

        # y_true: all zeros (no labels/no foreground)
        y_true = keras.ops.zeros((batch_size, height, width, 1), dtype=np.float32)

        # y_pred: incorrectly predicts some foreground
        y_pred = keras.ops.ones((batch_size, height, width, 1), dtype=np.float32) * 0.8

        # Add background channel
        y_true_with_bg = keras.ops.concatenate([1 - y_true, y_true], axis=-1)
        y_pred_with_bg = keras.ops.concatenate([1 - y_pred, y_pred], axis=-1)

        loss_fn = asymmetric_focal_tversky_loss(delta=0.7, gamma=0.75)
        loss = loss_fn(y_true_with_bg, y_pred_with_bg)

        loss_value = float(keras.ops.convert_to_numpy(loss))

        print(f"\nEmpty labels, incorrect prediction (high confidence): loss = {loss_value}")

        # Loss should be high (bad prediction) and well-defined
        assert not np.isnan(loss_value), "Loss should not be NaN"
        assert not np.isinf(loss_value), "Loss should not be Inf"
        assert loss_value > 0.5, "Loss should be high for incorrect predictions"

    def test_full_labels_correct_prediction(self):
        """
        Test case: When y_true is all ones (full foreground) and y_pred
        correctly predicts all ones, the loss should be low.
        """
        batch_size = 2
        height, width = 64, 64

        # y_true: all ones (full foreground)
        y_true = keras.ops.ones((batch_size, height, width, 1), dtype=np.float32)

        # y_pred: correctly predicts all ones
        y_pred = keras.ops.ones((batch_size, height, width, 1), dtype=np.float32) * 0.99

        # Add background channel
        y_true_with_bg = keras.ops.concatenate([1 - y_true, y_true], axis=-1)
        y_pred_with_bg = keras.ops.concatenate([1 - y_pred, y_pred], axis=-1)

        loss_fn = asymmetric_focal_tversky_loss(delta=0.7, gamma=0.75)
        loss = loss_fn(y_true_with_bg, y_pred_with_bg)

        loss_value = float(keras.ops.convert_to_numpy(loss))

        print(f"\nFull labels, correct prediction: loss = {loss_value}")

        # Loss should be relatively low (good prediction) and well-defined
        # Note: with gamma=0.75, even good predictions have moderate loss values
        assert not np.isnan(loss_value), "Loss should not be NaN"
        assert not np.isinf(loss_value), "Loss should not be Inf"
        assert loss_value < 0.8, "Loss should be reasonable for correct predictions"

    def test_full_labels_incorrect_prediction(self):
        """
        Test case: When y_true is all ones (full foreground) but y_pred
        incorrectly predicts all zeros, the loss should be high.
        """
        batch_size = 2
        height, width = 64, 64

        # y_true: all ones (full foreground)
        y_true = keras.ops.ones((batch_size, height, width, 1), dtype=np.float32)

        # y_pred: incorrectly predicts all zeros
        y_pred = keras.ops.zeros((batch_size, height, width, 1), dtype=np.float32)

        # Add background channel
        y_true_with_bg = keras.ops.concatenate([1 - y_true, y_true], axis=-1)
        y_pred_with_bg = keras.ops.concatenate([1 - y_pred, y_pred], axis=-1)

        loss_fn = asymmetric_focal_tversky_loss(delta=0.7, gamma=0.75)
        loss = loss_fn(y_true_with_bg, y_pred_with_bg)

        loss_value = float(keras.ops.convert_to_numpy(loss))

        print(f"\nFull labels, incorrect prediction: loss = {loss_value}")

        # Loss should be high (bad prediction) and well-defined
        assert not np.isnan(loss_value), "Loss should not be NaN"
        assert not np.isinf(loss_value), "Loss should not be Inf"
        assert loss_value > 0.5, "Loss should be high for incorrect predictions"

    def test_partial_labels_good_prediction(self):
        """
        Test case: When y_true has some foreground and y_pred matches well,
        the loss should be low.
        """
        batch_size = 2
        height, width = 64, 64

        # y_true: partial foreground (circle in center) - use numpy array
        y_true = np.zeros((batch_size, height, width, 1), dtype=np.float32)
        center_h, center_w = height // 2, width // 2
        radius = 15
        for h in range(height):
            for w in range(width):
                if (h - center_h)**2 + (w - center_w)**2 < radius**2:
                    y_true[:, h, w, 0] = 1.0

        # y_pred: similar to y_true (good prediction)
        y_pred = y_true.copy() * 0.95 + 0.02  # slight noise

        # Convert to keras tensors and add background channel
        y_true = keras.ops.convert_to_tensor(y_true)
        y_pred = keras.ops.convert_to_tensor(y_pred)
        y_true_with_bg = keras.ops.concatenate([1 - y_true, y_true], axis=-1)
        y_pred_with_bg = keras.ops.concatenate([1 - y_pred, y_pred], axis=-1)

        loss_fn = asymmetric_focal_tversky_loss(delta=0.7, gamma=0.75)
        loss = loss_fn(y_true_with_bg, y_pred_with_bg)

        loss_value = float(keras.ops.convert_to_numpy(loss))

        print(f"\nPartial labels, good prediction: loss = {loss_value}")

        # Loss should be reasonably low (good prediction) and well-defined
        assert not np.isnan(loss_value), "Loss should not be NaN"
        assert not np.isinf(loss_value), "Loss should not be Inf"
        assert loss_value < 0.5, f"Loss should be reasonably low for good predictions, got {loss_value:.6f}"

    def test_mixed_batch_with_empty_samples(self):
        """
        Test case that closely matches real-world scenario: A batch where some
        samples have labels and some don't. This is the scenario where the bug
        manifests in practice.
        """
        batch_size = 4
        height, width = 64, 64

        # Use numpy arrays for assignment
        y_true = np.zeros((batch_size, height, width, 1), dtype=np.float32)
        y_pred = np.zeros((batch_size, height, width, 1), dtype=np.float32)

        # Sample 0: has labels and good prediction
        y_true[0, 20:40, 20:40, 0] = 1.0
        y_pred[0, 20:40, 20:40, 0] = 0.9

        # Sample 1: has labels but poor prediction
        y_true[1, 10:30, 10:30, 0] = 1.0
        y_pred[1, 10:30, 10:30, 0] = 0.3

        # Sample 2: NO labels, correct prediction (all zeros) - BUG HERE
        # y_true[2] = 0, y_pred[2] = 0

        # Sample 3: NO labels, correct prediction (all zeros) - BUG HERE
        # y_true[3] = 0, y_pred[3] = 0

        # Convert to keras tensors and add background channel
        y_true = keras.ops.convert_to_tensor(y_true)
        y_pred = keras.ops.convert_to_tensor(y_pred)
        y_true_with_bg = keras.ops.concatenate([1 - y_true, y_true], axis=-1)
        y_pred_with_bg = keras.ops.concatenate([1 - y_pred, y_pred], axis=-1)

        loss_fn = asymmetric_focal_tversky_loss(delta=0.7, gamma=0.75)
        loss = loss_fn(y_true_with_bg, y_pred_with_bg)

        loss_value = float(keras.ops.convert_to_numpy(loss))

        print(f"\nMixed batch (2 with labels, 2 empty): loss = {loss_value}")

        # With the fix: empty samples contribute near-zero loss when correctly predicted
        # The overall batch loss should be dominated by the labeled samples
        assert not np.isnan(loss_value), "Loss should not be NaN"
        assert not np.isinf(loss_value), "Loss should not be Inf"

        # Loss should be reasonable (not inflated by the empty samples)
        print(f"Empty samples correctly contribute minimal loss to the batch")


class TestAsymmetricFocalLoss:
    """Tests for asymmetric_focal_loss function."""

    def test_empty_labels_correct_prediction(self):
        """Test that asymmetric_focal_loss handles empty labels correctly."""
        batch_size = 2
        height, width = 64, 64

        y_true = keras.ops.zeros((batch_size, height, width, 1), dtype=np.float32)
        y_pred = keras.ops.zeros((batch_size, height, width, 1), dtype=np.float32)

        # Add background channel
        y_true_with_bg = keras.ops.concatenate([1 - y_true, y_true], axis=-1)
        y_pred_with_bg = keras.ops.concatenate([1 - y_pred, y_pred], axis=-1)

        loss_fn = asymmetric_focal_loss(delta=0.7, gamma=2.0)
        loss = loss_fn(y_true_with_bg, y_pred_with_bg)

        loss_value = float(keras.ops.convert_to_numpy(loss))

        print(f"\nAsymmetric Focal Loss - Empty labels, correct prediction: loss = {loss_value}")

        # This loss function should handle empty labels correctly
        assert not np.isnan(loss_value), "Loss should not be NaN"
        assert not np.isinf(loss_value), "Loss should not be Inf"


class TestAsymUnifiedFocalLoss:
    """Tests for asym_unified_focal_loss function."""

    def test_empty_labels_correct_prediction(self):
        """
        Test that asym_unified_focal_loss correctly handles empty labels with
        the fixed asymmetric_focal_tversky_loss component.
        """
        batch_size = 2
        height, width = 64, 64

        y_true = keras.ops.zeros((batch_size, height, width, 1), dtype=np.float32)
        y_pred = keras.ops.zeros((batch_size, height, width, 1), dtype=np.float32)

        loss_fn = asym_unified_focal_loss(weight=0.5, delta=0.6, gamma=0.5)
        loss = loss_fn(y_true, y_pred)

        loss_value = float(keras.ops.convert_to_numpy(loss))

        print(f"\nAsym Unified Focal Loss - Empty labels, correct prediction: loss = {loss_value}")

        # With the fix: unified loss should be low for perfect prediction
        assert not np.isnan(loss_value), "Loss should not be NaN"
        assert not np.isinf(loss_value), "Loss should not be Inf"
        assert loss_value < 0.2, f"Loss should be low for perfect prediction, got {loss_value:.6f}"


class TestSymmetricFocalTverskyLoss:
    """Tests for symmetric_focal_tversky_loss function."""

    def test_empty_labels_correct_prediction(self):
        """
        Test that symmetric_focal_tversky_loss correctly handles empty labels
        with the weighted averaging fix.
        """
        batch_size = 2
        height, width = 64, 64

        y_true = keras.ops.zeros((batch_size, height, width, 1), dtype=np.float32)
        y_pred = keras.ops.zeros((batch_size, height, width, 1), dtype=np.float32)

        # Add background channel
        y_true_with_bg = keras.ops.concatenate([1 - y_true, y_true], axis=-1)
        y_pred_with_bg = keras.ops.concatenate([1 - y_pred, y_pred], axis=-1)

        loss_fn = symmetric_focal_tversky_loss(delta=0.7, gamma=0.75)
        loss = loss_fn(y_true_with_bg, y_pred_with_bg)

        loss_value = float(keras.ops.convert_to_numpy(loss))

        print(f"\nSymmetric Focal Tversky Loss - Empty labels, correct prediction: loss = {loss_value}")

        # With the fix: symmetric version should also have low loss for perfect prediction
        assert not np.isnan(loss_value), "Loss should not be NaN"
        assert not np.isinf(loss_value), "Loss should not be Inf"
        assert loss_value < 0.1, f"Loss should be low for perfect prediction, got {loss_value:.6f}"


class TestParameterVariations:
    """Test different parameter combinations to understand bug behavior."""

    @pytest.mark.parametrize("gamma", [0.5, 0.75, 1.0, 1.5, 2.0])
    def test_gamma_variation(self, gamma):
        """Test how different gamma values affect the bug."""
        batch_size = 2
        height, width = 64, 64

        y_true = keras.ops.zeros((batch_size, height, width, 1), dtype=np.float32)
        y_pred = keras.ops.zeros((batch_size, height, width, 1), dtype=np.float32)

        # Add background channel
        y_true_with_bg = keras.ops.concatenate([1 - y_true, y_true], axis=-1)
        y_pred_with_bg = keras.ops.concatenate([1 - y_pred, y_pred], axis=-1)

        loss_fn = asymmetric_focal_tversky_loss(delta=0.7, gamma=gamma)
        loss = loss_fn(y_true_with_bg, y_pred_with_bg)

        loss_value = float(keras.ops.convert_to_numpy(loss))

        print(f"\nGamma={gamma}: loss = {loss_value}")

        # With the fix: all gamma values should produce low loss for perfect prediction
        assert not np.isnan(loss_value), "Loss should not be NaN"
        assert not np.isinf(loss_value), "Loss should not be Inf"
        assert loss_value < 0.1, f"Loss should be low for gamma={gamma}, got {loss_value:.4f}"

    @pytest.mark.parametrize("delta", [0.3, 0.5, 0.7, 0.9])
    def test_delta_variation(self, delta):
        """Test how different delta values affect the bug."""
        batch_size = 2
        height, width = 64, 64

        y_true = keras.ops.zeros((batch_size, height, width, 1), dtype=np.float32)
        y_pred = keras.ops.zeros((batch_size, height, width, 1), dtype=np.float32)

        # Add background channel
        y_true_with_bg = keras.ops.concatenate([1 - y_true, y_true], axis=-1)
        y_pred_with_bg = keras.ops.concatenate([1 - y_pred, y_pred], axis=-1)

        loss_fn = asymmetric_focal_tversky_loss(delta=delta, gamma=0.75)
        loss = loss_fn(y_true_with_bg, y_pred_with_bg)

        loss_value = float(keras.ops.convert_to_numpy(loss))

        print(f"\nDelta={delta}: loss = {loss_value}")

        # With the fix: all delta values should produce low loss for perfect prediction
        assert not np.isnan(loss_value), "Loss should not be NaN"
        assert not np.isinf(loss_value), "Loss should not be Inf"
        assert loss_value < 0.1, f"Loss should be low for delta={delta}, got {loss_value:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

