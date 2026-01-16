"""
PyTorch implementation of Triplet Loss functions.

This module provides PyTorch equivalents of the Keras/TensorFlow triplet loss
implementations for embedding learning.

MIT License

Copyright (c) [2018] [Olivier Moindrot]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Modified by Paul Thompson 2024
PyTorch conversion 2026
"""

import torch
from typing import Tuple


def _pairwise_distances(embeddings: torch.Tensor, squared: bool = False) -> torch.Tensor:
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = torch.matmul(embeddings, embeddings.t())

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = torch.diag(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = torch.clamp(distances, min=0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = (distances == 0.0).float()
        distances = distances + mask * 1e-16

        distances = torch.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels: torch.Tensor) -> torch.Tensor:
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

    Args:
        labels: torch.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: torch.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    batch_size = labels.size(0)
    device = labels.device
    indices_equal = torch.eye(batch_size, dtype=torch.bool, device=device)
    indices_not_equal = ~indices_equal

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

    # Handle potential extra dimensions (squeeze if needed)
    if labels_equal.dim() > 2:
        labels_equal = labels_equal.squeeze()

    # Combine the two masks
    mask = indices_not_equal & labels_equal

    return mask


def _get_anchor_negative_triplet_mask(labels: torch.Tensor) -> torch.Tensor:
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    Args:
        labels: torch.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: torch.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

    mask = ~labels_equal

    return mask


def _get_triplet_mask(labels: torch.Tensor) -> torch.Tensor:
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.

    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]

    Args:
        labels: torch.int32 `Tensor` with shape [batch_size]
        
    Returns:
        mask: torch.bool `Tensor` with shape [batch_size, batch_size, batch_size]
    """
    batch_size = labels.size(0)
    device = labels.device

    # Check that i, j and k are distinct
    indices_equal = torch.eye(batch_size, dtype=torch.bool, device=device)
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    distinct_indices = i_not_equal_j & i_not_equal_k & j_not_equal_k

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

    # Handle potential extra dimensions (squeeze if needed)
    if label_equal.dim() > 2:
        label_equal = label_equal.squeeze()

    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)

    valid_labels = i_equal_j & (~i_equal_k)

    # Combine the two masks
    mask = distinct_indices & valid_labels

    return mask


def batch_hard_triplet_loss(
    labels: torch.Tensor,
    embeddings: torch.Tensor,
    margin: float,
    squared: bool = False,
    normalize: bool = False
) -> torch.Tensor:
    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
        normalize: Boolean. If true, normalize pairwise distances.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)
    if normalize:
        max_dist = pairwise_dist.max()
        if max_dist > 0:
            pairwise_dist = pairwise_dist / max_dist

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = mask_anchor_positive.float()

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = mask_anchor_positive * pairwise_dist

    # shape (batch_size, 1)
    hardest_positive_dist, _ = anchor_positive_dist.max(dim=1, keepdim=True)

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = mask_anchor_negative.float()

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist, _ = pairwise_dist.max(dim=1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist, _ = anchor_negative_dist.min(dim=1, keepdim=True)
    scaling = anchor_negative_dist.mean(dim=1, keepdim=True) + 1e-16

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    # https://quaterion.qdrant.tech/tutorials/triplet_loss_trick
    triplet_loss = torch.clamp(
        (hardest_positive_dist - hardest_negative_dist) / scaling + margin,
        min=0.0
    )

    # Get final mean triplet loss
    triplet_loss = triplet_loss.mean()

    return triplet_loss


def batch_all_triplet_loss(
    labels: torch.Tensor,
    embeddings: torch.Tensor,
    margin: float,
    squared: bool = False,
    normalize: bool = False,
    remove_negative: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
        normalize: Boolean. If true, normalize pairwise distances.
        remove_negative: Boolean. If true, remove easy (negative) triplets.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
        fraction_positive_triplets: fraction of triplets with positive loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)
    if normalize:
        max_dist = pairwise_dist.max()
        if max_dist > 0:
            pairwise_dist = pairwise_dist / max_dist

    anchor_positive_dist = pairwise_dist.unsqueeze(2)
    anchor_negative_dist = pairwise_dist.unsqueeze(1)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    mask = mask.float()
    triplet_loss = mask * triplet_loss

    num_valid_triplets = mask.sum()

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = (triplet_loss > 1e-16).float()
    num_positive_triplets = valid_triplets.sum()
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Remove negative losses (i.e. the easy triplets)
    if remove_negative:
        triplet_loss = torch.clamp(triplet_loss, min=0.0)

        # Get final mean triplet loss over the positive valid triplets
        # num positive triplets will be less than num_valid_triplets
        # As the model improves, there will be fewer positive triplets
        # because more will be classified as "easy" and removed
        triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

    else:
        triplet_loss = triplet_loss.sum() / (num_valid_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets


def hard_negative_triplet_mining(
    labels: torch.Tensor,
    embeddings: torch.Tensor,
    margin: float,
    squared: bool = False
) -> torch.Tensor:
    """Build the triplet loss over a batch of embeddings using hard negative mining.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: tensor containing the triplet loss per anchor-positive pair
    """
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = mask_anchor_positive.float()

    anchor_positive_dist = mask_anchor_positive * pairwise_dist

    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = mask_anchor_negative.float()

    max_anchor_negative_dist, _ = pairwise_dist.max(dim=1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    hardest_negative_dist, _ = anchor_negative_dist.min(dim=1, keepdim=True)
    scaling = anchor_negative_dist.mean(dim=1, keepdim=True) + 1e-16

    triplet_loss = torch.clamp(
        (anchor_positive_dist - hardest_negative_dist) / scaling + margin,
        min=0.0
    )

    return triplet_loss


def semi_hard_negative_triplet_mining(
    labels: torch.Tensor,
    embeddings: torch.Tensor,
    margin: float,
    squared: bool = False
) -> torch.Tensor:
    """Build the triplet loss over a batch of embeddings using semi-hard negative mining.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        semi_hard_negative_dist: tensor containing semi-hard negative distances
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    anchor_positive_dist = pairwise_dist.unsqueeze(2)
    anchor_negative_dist = pairwise_dist.unsqueeze(1)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    mask = _get_triplet_mask(labels)
    mask = mask.float()
    triplet_loss = mask * triplet_loss

    # Get the semi-hard negative
    # The semi-hard negative is the negative that is closest to the anchor
    # but still further than the positive
    # (thus triplet_loss < 0 and triplet_loss is the smallest)
    semi_hard_negative_dist, _ = torch.where(
        triplet_loss < 0,
        triplet_loss,
        torch.zeros_like(triplet_loss)
    ).max(dim=1, keepdim=True)

    return semi_hard_negative_dist


def batch_distance_loss(
    labels: torch.Tensor,
    embeddings: torch.Tensor,
    squared: bool = False,
    normalize: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
        normalize: Boolean. If true, normalize pairwise distances.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
        fraction_negative_triplets: fraction of triplets with negative loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)
    if normalize:
        max_dist = pairwise_dist.max()
        if max_dist > 0:
            pairwise_dist = pairwise_dist / max_dist

    anchor_positive_dist = pairwise_dist.unsqueeze(2)
    anchor_negative_dist = pairwise_dist.unsqueeze(1)

    triplet_loss = anchor_positive_dist - anchor_negative_dist

    # Put to zero the invalid triplets
    mask = _get_triplet_mask(labels)
    mask = mask.float()
    triplet_loss = mask * triplet_loss

    num_valid_triplets = mask.sum()

    # Count number of negative triplets (where triplet_loss < 0)
    valid_triplets = (triplet_loss < -1e-16).float()
    num_negative_triplets = valid_triplets.sum()
    fraction_negative_triplets = num_negative_triplets / (num_valid_triplets + 1e-16)

    triplet_loss = triplet_loss.sum() / (num_valid_triplets + 1e-16)

    return triplet_loss, fraction_negative_triplets
