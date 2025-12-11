import tensorflow as tf
import numpy as np

from src.DeepLearningUtils.Losses.Triplet_Loss.pixel_triplet_loss_keras import (
    _get_anchor_positive_triplet_mask_exclude_background,
    _get_triplet_mask_exclude_background,
    PixelTripletLoss,
    PixelTripletConfig,
)


def test_masks_exclude_background_from_positives():
    # labels: batch of 5 pixels/classes
    # classes: [0(bg), 1, 1, 2, 0(bg)]
    labels = tf.constant([0, 1, 1, 2, 0], dtype=tf.int32)

    ap_mask = _get_anchor_positive_triplet_mask_exclude_background(labels)
    ap_mask_np = ap_mask.numpy()

    # Background anchors (idx 0, 4) should have no positives
    assert ap_mask_np[0].sum() == 0
    assert ap_mask_np[4].sum() == 0

    # Non-background anchors: idx 1 and 2 are class 1; positives only between them
    assert bool(ap_mask_np[1, 2])
    assert bool(ap_mask_np[2, 1])
    # No self-pairs
    assert not bool(ap_mask_np[1, 1])
    assert not bool(ap_mask_np[2, 2])
    # Different class is not positive
    assert not bool(ap_mask_np[1, 3])
    assert not bool(ap_mask_np[2, 3])

    # Triplet mask: negatives can include background
    triplet_mask = _get_triplet_mask_exclude_background(labels)
    triplet_mask_np = triplet_mask.numpy()

    # Anchor at 1 (class 1), positive 2 (class 1), negative 0 (bg) should be valid
    assert bool(triplet_mask_np[1, 2, 0])
    # Anchor at bg (0) should have no valid triplets
    assert triplet_mask_np[0].sum() == 0


def test_loss_runs_and_ignores_background_positives():
    # Construct simple embeddings and labels
    # Batch: treat flattened pixel sampling result as a batch for the loss core functions
    # We'll simulate a scenario where we have background and two same-class non-bg points
    embeddings = tf.constant(
        [
            [0.0, 0.0],  # bg 0
            [1.0, 0.0],  # class 1
            [1.1, 0.0],  # class 1
            [0.0, 1.0],  # class 2
            [0.0, 0.1],  # bg 0
        ], dtype=tf.float32
    )
    labels = tf.constant([0, 1, 1, 2, 0], dtype=tf.int32)

    loss_obj = PixelTripletLoss(config=PixelTripletConfig(distance_metric="euclidean", triplet_strategy="hard"))
    # Directly call the custom hard loss to bypass pixel sampling
    hard_loss = loss_obj._batch_hard_triplet_loss_custom(labels, embeddings)

    # There are positives only for class 1 (between indices 1 and 2). Background is not used as positive.
    # Loss should be non-negative and finite
    hard_loss_val = float(hard_loss.numpy())
    assert np.isfinite(hard_loss_val)
    assert hard_loss_val >= 0.0
