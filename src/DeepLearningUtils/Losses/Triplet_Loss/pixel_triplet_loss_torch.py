#!/usr/bin/env python3
"""
PyTorch implementation of Pixel-based Triplet Loss for Segmentation.

This module implements a PyTorch triplet loss that operates on pixel-level
embeddings for semantic segmentation tasks, mirroring the Keras implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

# Import PyTorch triplet loss helper functions
from .triplet_loss_torch import (
    _pairwise_distances,
    _get_anchor_positive_triplet_mask,
    _get_anchor_negative_triplet_mask,
    _get_triplet_mask,
)


def _get_anchor_positive_triplet_mask_exclude_background(labels: torch.Tensor) -> torch.Tensor:
    """
    Valid anchor-positive pairs where labels match and are non-background (>0).
    Shape: (batch, batch)
    """
    # Ensure labels is rank-1 [B]
    labels = labels.view(-1)
    batch_size = labels.size(0)
    device = labels.device

    # i != j
    indices_not_equal = ~torch.eye(batch_size, dtype=torch.bool, device=device)

    # label match
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B,B)

    # anchor must be non-background; since labels_equal, positive is also non-bg
    anchor_non_bg = labels.unsqueeze(0) > 0  # (1, B) broadcast along columns

    mask = indices_not_equal & labels_equal & anchor_non_bg
    return mask


def _get_triplet_mask_exclude_background(labels: torch.Tensor) -> torch.Tensor:
    """
    Valid triplets (a,p,n) where:
    - a != p, a != n, p != n
    - labels[a] == labels[p] and labels[a] > 0
    - labels[a] != labels[n] (negative can be background or any other class)
    Shape: (batch, batch, batch)
    """
    # Ensure labels is rank-1 [B]
    labels = labels.view(-1)
    batch_size = labels.size(0)
    device = labels.device

    # distinct indices masks
    eye = torch.eye(batch_size, dtype=torch.bool, device=device)  # (B,B)
    i_not_j = ~eye  # (B,B)

    # Expand to (B,B,B)
    i_not_j_3d = i_not_j.unsqueeze(2)  # (B,B,1)
    i_not_k_3d = i_not_j.unsqueeze(1)  # (B,1,B)
    j_not_k_3d = i_not_j.unsqueeze(0)  # (1,B,B)

    distinct = i_not_j_3d & i_not_k_3d & j_not_k_3d  # (B,B,B)

    # labels equality/inequality across axes (a,p,n)
    labels_a = labels.view(-1, 1, 1)   # (B,1,1)
    labels_p = labels.view(1, -1, 1)   # (1,B,1)
    labels_n = labels.view(1, 1, -1)   # (1,1,B)

    labels_equal_ap = labels_a == labels_p            # (B,B,B)
    labels_not_equal_an = labels_a != labels_n        # (B,B,B)

    # anchor must be non-background
    anchor_non_bg = labels_a > 0                      # (B,B,B)

    mask = distinct & labels_equal_ap & anchor_non_bg & labels_not_equal_an
    return mask


@dataclass
class PixelTripletConfig:
    """Configuration for pixel-based triplet loss."""
    margin: float = 1.0
    # Legacy parameters (kept for backward compatibility)
    background_pixels: int = 1000
    whisker_pixels: int = 500
    # New balanced sampling parameters
    max_samples_per_class: Optional[int] = None
    use_balanced_sampling: bool = True
    strict_per_class_balancing: bool = False
    distance_metric: str = "euclidean"
    triplet_strategy: str = "semi_hard"
    reduction: str = "mean"
    remove_easy_triplets: bool = False
    memory_warning_threshold: int = 10_000_000
    
    # Exact computation (loop-based)
    use_exact: bool = False
    batch_size_for_exact: int = 100
    class_balanced_weighting: bool = False

    def __post_init__(self):
        if self.margin <= 0:
            raise ValueError(f"margin must be > 0, got {self.margin}")
        if self.background_pixels <= 0:
            raise ValueError(f"background_pixels must be > 0, got {self.background_pixels}")
        if self.whisker_pixels <= 0:
            raise ValueError(f"whisker_pixels must be > 0, got {self.whisker_pixels}")
        if self.max_samples_per_class is not None and self.max_samples_per_class <= 0:
            raise ValueError(f"max_samples_per_class must be > 0 or None, got {self.max_samples_per_class}")
        if self.distance_metric not in ["euclidean", "cosine", "manhattan"]:
            raise ValueError(f"distance_metric must be one of ['euclidean', 'cosine', 'manhattan'], got {self.distance_metric}")
        if self.triplet_strategy not in ["hard", "semi_hard", "all"]:
            raise ValueError(f"triplet_strategy must be one of ['hard', 'semi_hard', 'all'], got {self.triplet_strategy}")
        if self.reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"reduction must be one of ['mean', 'sum', 'none'], got {self.reduction}")
        if self.batch_size_for_exact <= 0:
            raise ValueError(f"batch_size_for_exact must be > 0, got {self.batch_size_for_exact}")

        # class_balanced_weighting requires use_exact
        if self.class_balanced_weighting and not self.use_exact:
            print("Warning: class_balanced_weighting=True requires use_exact=True. "
                  "Setting use_exact=True automatically.")
            object.__setattr__(self, 'use_exact', True)

        # Provide guidance on the new parameters
        if self.use_balanced_sampling and self.max_samples_per_class is None:
            self.max_samples_per_class = min(self.background_pixels, self.whisker_pixels)


class PixelTripletLoss(nn.Module):
    """
    PyTorch Pixel-based triplet loss for semantic segmentation.
    
    This loss operates on pixel-level embeddings, sampling a specified number of pixels
    per class and computing triplet loss with semi-hard or hard negative mining.
    
    Key Features:
    - Upsamples embeddings to match label resolution (preserves thin structures)
    - Configurable distance metrics (Euclidean, cosine, Manhattan)
    - Balanced sampling to prevent class imbalance issues
    - Literature-compliant easy triplet handling
    
    Args:
        config: PixelTripletConfig instance with loss parameters
    """
    
    def __init__(
        self,
        config: Optional[PixelTripletConfig] = None,
    ):
        super().__init__()
        self.config = config or PixelTripletConfig()
        
    def _compute_pairwise_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distances using the specified distance metric.
        
        Args:
            embeddings: tensor of shape (batch_size, embed_dim)
            
        Returns:
            pairwise_distances: tensor of shape (batch_size, batch_size)
        """
        if self.config.distance_metric == "euclidean":
            return _pairwise_distances(embeddings, squared=False)
        elif self.config.distance_metric == "cosine":
            return self._cosine_distances(embeddings)
        elif self.config.distance_metric == "manhattan":
            return self._manhattan_distances(embeddings)
        else:
            raise ValueError(f"Unsupported distance metric: {self.config.distance_metric}")
    
    def _cosine_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise cosine distances.
        
        Cosine distance = 1 - cosine_similarity
        """
        # Normalize embeddings to unit vectors
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute cosine similarities
        cosine_similarities = torch.matmul(normalized_embeddings, normalized_embeddings.t())
        
        # Convert to cosine distances
        cosine_distances = 1.0 - cosine_similarities
        
        # Ensure distances are non-negative (handle numerical errors)
        cosine_distances = torch.clamp(cosine_distances, min=0.0)
        
        return cosine_distances
    
    def _manhattan_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise Manhattan (L1) distances.
        """
        # Expand embeddings for broadcasting
        embeddings_i = embeddings.unsqueeze(1)  # (batch_size, 1, embed_dim)
        embeddings_j = embeddings.unsqueeze(0)  # (1, batch_size, embed_dim)
        
        # Compute absolute differences and sum along the embedding dimension
        manhattan_distances = torch.sum(torch.abs(embeddings_i - embeddings_j), dim=2)
        
        return manhattan_distances
    
    def _batch_hard_triplet_loss_custom(
        self, 
        labels: torch.Tensor, 
        embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Custom implementation of batch hard triplet loss with configurable distance metrics.
        """
        # Get the pairwise distance matrix using the configured metric
        pairwise_dist = self._compute_pairwise_distances(embeddings)
        
        # For each anchor, get the hardest positive (exclude background positives)
        mask_anchor_positive = _get_anchor_positive_triplet_mask_exclude_background(labels)
        mask_anchor_positive = mask_anchor_positive.float()
        
        # We put to 0 any element where (a, p) is not valid 
        anchor_positive_dist = mask_anchor_positive * pairwise_dist
        
        # shape (batch_size, 1)
        hardest_positive_dist, _ = anchor_positive_dist.max(dim=1, keepdim=True)
        
        # For each anchor, get the hardest negative (background allowed as negative)
        mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
        mask_anchor_negative = mask_anchor_negative.float()
        
        # We add the maximum value in each row to the invalid negatives
        max_anchor_negative_dist, _ = pairwise_dist.max(dim=1, keepdim=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
        
        # shape (batch_size,)
        hardest_negative_dist, _ = anchor_negative_dist.min(dim=1, keepdim=True)
        scaling = anchor_negative_dist.mean(dim=1, keepdim=True) + 1e-16
        
        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        triplet_loss = torch.clamp(
            (hardest_positive_dist - hardest_negative_dist) / scaling + self.config.margin,
            min=0.0
        )
        
        # Get final mean triplet loss
        triplet_loss = triplet_loss.mean()
        
        return triplet_loss
    
    def _batch_all_triplet_loss_custom(
        self, 
        labels: torch.Tensor, 
        embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Custom implementation of batch all triplet loss with configurable distance metrics.
        """
        # Get the pairwise distance matrix using the configured metric
        pairwise_dist = self._compute_pairwise_distances(embeddings)
        
        anchor_positive_dist = pairwise_dist.unsqueeze(2)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)
        
        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.config.margin
        
        # Put to zero the invalid triplets (exclude background positives)
        mask = _get_triplet_mask_exclude_background(labels)
        mask = mask.float()
        triplet_loss = mask * triplet_loss
        
        num_valid_triplets = mask.sum()
        
        # Handle easy triplets based on configuration
        if self.config.remove_easy_triplets:
            # Remove negative losses (i.e. the easy triplets)
            triplet_loss = torch.clamp(triplet_loss, min=0.0)
            
            # Count number of positive triplets (where triplet_loss > 0)
            valid_triplets = (triplet_loss > 1e-16).float()
            num_positive_triplets = valid_triplets.sum()
            
            # Get final mean triplet loss over the positive valid triplets
            triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)
        else:
            # Include easy triplets (typical literature behavior)
            triplet_loss = triplet_loss.sum() / (num_valid_triplets + 1e-16)
        
        return triplet_loss

    def _compute_distances_from_anchors(
        self,
        anchor_embeddings: torch.Tensor,
        all_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distances from anchor embeddings to all embeddings.
        """
        if self.config.distance_metric == "euclidean":
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
            anchor_sq_norm = (anchor_embeddings ** 2).sum(dim=1, keepdim=True)  # (B, 1)
            all_sq_norm = (all_embeddings ** 2).sum(dim=1)  # (N,)
            dot_product = torch.matmul(anchor_embeddings, all_embeddings.t())  # (B, N)
            sq_distances = anchor_sq_norm - 2.0 * dot_product + all_sq_norm.unsqueeze(0)
            sq_distances = torch.clamp(sq_distances, min=0.0)
            return torch.sqrt(sq_distances + 1e-16)
        elif self.config.distance_metric == "cosine":
            # Normalize embeddings
            anchor_norm = F.normalize(anchor_embeddings, p=2, dim=1)
            all_norm = F.normalize(all_embeddings, p=2, dim=1)
            similarities = torch.matmul(anchor_norm, all_norm.t())
            return torch.clamp(1.0 - similarities, min=0.0)
        else:  # manhattan
            anchor_exp = anchor_embeddings.unsqueeze(1)  # (B, 1, D)
            all_exp = all_embeddings.unsqueeze(0)  # (1, N, D)
            return torch.sum(torch.abs(anchor_exp - all_exp), dim=2)

    def _batch_hard_triplet_loss_exact(
        self, 
        labels: torch.Tensor, 
        embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Batch hard triplet loss using loop-based exact computation.
        Memory usage is O(batch_size * N) instead of O(N^2).
        """
        num_pixels = embeddings.size(0)
        batch_size = self.config.batch_size_for_exact
        num_batches = (num_pixels + batch_size - 1) // batch_size
        device = embeddings.device

        all_labels = labels

        loss_sum = torch.tensor(0.0, device=device)
        valid_anchor_count = torch.tensor(0.0, device=device)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_pixels)

            anchor_embeddings = embeddings[start_idx:end_idx]  # (B, D)
            anchor_labels = labels[start_idx:end_idx]  # (B,)

            # Compute distances from anchors to all pixels
            distances = self._compute_distances_from_anchors(anchor_embeddings, embeddings)  # (B, N)

            # Create label masks
            anchor_labels_exp = anchor_labels.unsqueeze(1)  # (B, 1)
            all_labels_exp = all_labels.unsqueeze(0)  # (1, N)

            # Positive mask: same class, not self, anchor is non-background
            same_class = anchor_labels_exp == all_labels_exp  # (B, N)
            anchor_non_bg = anchor_labels > 0  # (B,)
            anchor_non_bg_exp = anchor_non_bg.unsqueeze(1)  # (B, 1)

            # Self mask
            batch_indices = torch.arange(start_idx, end_idx, device=device)
            all_indices = torch.arange(num_pixels, device=device)
            self_mask = batch_indices.unsqueeze(1) == all_indices.unsqueeze(0)
            not_self = ~self_mask

            # Valid positive: same class, not self, anchor is foreground
            positive_mask = same_class & not_self & anchor_non_bg_exp

            # Negative mask: different class
            negative_mask = ~same_class

            positive_mask_float = positive_mask.float()
            negative_mask_float = negative_mask.float()

            # For hardest positive: max distance among positives
            masked_positive_dist = distances * positive_mask_float + (1.0 - positive_mask_float) * (-1e9)
            hardest_positive_dist, _ = masked_positive_dist.max(dim=1)  # (B,)

            # For hardest negative: min distance among negatives
            masked_negative_dist = distances + (1.0 - negative_mask_float) * 1e9
            hardest_negative_dist, _ = masked_negative_dist.min(dim=1)  # (B,)

            # Scaling factor (mean of negative distances)
            neg_sum = (distances * negative_mask_float).sum(dim=1)
            neg_count = negative_mask_float.sum(dim=1)
            scaling = neg_sum / (neg_count + 1e-16)

            # Compute triplet loss for each anchor
            triplet_loss = torch.clamp(
                (hardest_positive_dist - hardest_negative_dist) / (scaling + 1e-16) + self.config.margin,
                min=0.0
            )

            # Only count anchors that have valid positives
            has_valid_positive = positive_mask.any(dim=1)
            has_valid_negative = negative_mask.any(dim=1)
            valid_anchor = has_valid_positive & has_valid_negative
            valid_anchor_float = valid_anchor.float()

            batch_loss = (triplet_loss * valid_anchor_float).sum()
            batch_count = valid_anchor_float.sum()

            loss_sum = loss_sum + batch_loss
            valid_anchor_count = valid_anchor_count + batch_count

        return loss_sum / (valid_anchor_count + 1e-16)

    def _batch_all_triplet_loss_exact(
        self, 
        labels: torch.Tensor, 
        embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Batch all triplet loss using loop-based exact computation.
        Memory usage is O(batch_size * N) instead of O(N^3).
        """
        num_pixels = embeddings.size(0)
        batch_size = self.config.batch_size_for_exact
        num_batches = (num_pixels + batch_size - 1) // batch_size
        device = embeddings.device

        all_labels = labels

        loss_sum = torch.tensor(0.0, device=device)
        triplet_count = torch.tensor(0.0, device=device)
        positive_triplet_count = torch.tensor(0.0, device=device)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_pixels)
            actual_batch_size = end_idx - start_idx

            anchor_embeddings = embeddings[start_idx:end_idx]
            anchor_labels = labels[start_idx:end_idx]

            distances = self._compute_distances_from_anchors(anchor_embeddings, embeddings)

            anchor_labels_exp = anchor_labels.unsqueeze(1)
            all_labels_exp = all_labels.unsqueeze(0)

            same_class = anchor_labels_exp == all_labels_exp
            diff_class = ~same_class
            anchor_non_bg = anchor_labels > 0
            anchor_non_bg_exp = anchor_non_bg.unsqueeze(1)

            batch_indices = torch.arange(start_idx, end_idx, device=device)
            all_indices = torch.arange(num_pixels, device=device)
            self_mask = batch_indices.unsqueeze(1) == all_indices.unsqueeze(0)
            not_self = ~self_mask

            positive_mask = same_class & not_self & anchor_non_bg_exp
            negative_mask = diff_class

            positive_mask_float = positive_mask.float()
            negative_mask_float = negative_mask.float()

            pos_count = positive_mask_float.sum(dim=1)
            neg_count = negative_mask_float.sum(dim=1)

            if self.config.remove_easy_triplets:
                # Need to compute each triplet individually
                for anchor_idx in range(actual_batch_size):
                    anchor_distances = distances[anchor_idx]
                    anchor_pos_mask = positive_mask[anchor_idx]
                    anchor_neg_mask = negative_mask[anchor_idx]

                    pos_indices = torch.where(anchor_pos_mask)[0]
                    neg_indices = torch.where(anchor_neg_mask)[0]

                    if len(pos_indices) == 0 or len(neg_indices) == 0:
                        continue

                    pos_dists = anchor_distances[pos_indices]
                    neg_dists = anchor_distances[neg_indices]

                    # Compute all pairwise triplet losses
                    pos_expanded = pos_dists.unsqueeze(1)
                    neg_expanded = neg_dists.unsqueeze(0)

                    triplet_losses = pos_expanded - neg_expanded + self.config.margin
                    triplet_losses = torch.clamp(triplet_losses, min=0.0)

                    anchor_loss = triplet_losses.sum()
                    anchor_total_triplets = float(len(pos_indices) * len(neg_indices))
                    anchor_positive_triplets = (triplet_losses > 1e-16).float().sum()

                    loss_sum = loss_sum + anchor_loss
                    triplet_count = triplet_count + anchor_total_triplets
                    positive_triplet_count = positive_triplet_count + anchor_positive_triplets
            else:
                # Efficient computation
                pos_dist_sum = (distances * positive_mask_float).sum(dim=1)
                neg_dist_sum = (distances * negative_mask_float).sum(dim=1)
                triplets_per_anchor = pos_count * neg_count

                anchor_loss = (
                    pos_dist_sum * neg_count -
                    neg_dist_sum * pos_count +
                    self.config.margin * triplets_per_anchor
                )

                valid_anchor = triplets_per_anchor > 0
                valid_anchor_float = valid_anchor.float()

                batch_loss = (anchor_loss * valid_anchor_float).sum()
                batch_count = (triplets_per_anchor * valid_anchor_float).sum()

                loss_sum = loss_sum + batch_loss
                triplet_count = triplet_count + batch_count

        if self.config.remove_easy_triplets:
            return loss_sum / (positive_triplet_count + 1e-16)
        else:
            return loss_sum / (triplet_count + 1e-16)

    def _batch_hard_triplet_loss_exact_class_balanced(
        self, 
        labels: torch.Tensor, 
        embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Batch hard triplet loss with class-balanced weighting.
        """
        num_pixels = embeddings.size(0)
        batch_size = self.config.batch_size_for_exact
        num_batches = (num_pixels + batch_size - 1) // batch_size
        device = embeddings.device

        all_labels = labels
        max_classes = 100

        loss_per_class = torch.zeros(max_classes, device=device)
        count_per_class = torch.zeros(max_classes, device=device)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_pixels)

            anchor_embeddings = embeddings[start_idx:end_idx]
            anchor_labels = labels[start_idx:end_idx]

            distances = self._compute_distances_from_anchors(anchor_embeddings, embeddings)

            anchor_labels_exp = anchor_labels.unsqueeze(1)
            all_labels_exp = all_labels.unsqueeze(0)

            same_class = anchor_labels_exp == all_labels_exp
            anchor_non_bg = anchor_labels > 0
            anchor_non_bg_exp = anchor_non_bg.unsqueeze(1)

            batch_indices = torch.arange(start_idx, end_idx, device=device)
            all_indices = torch.arange(num_pixels, device=device)
            self_mask = batch_indices.unsqueeze(1) == all_indices.unsqueeze(0)
            not_self = ~self_mask

            positive_mask = same_class & not_self & anchor_non_bg_exp
            negative_mask = ~same_class

            positive_mask_float = positive_mask.float()
            negative_mask_float = negative_mask.float()

            masked_positive_dist = distances * positive_mask_float + (1.0 - positive_mask_float) * (-1e9)
            hardest_positive_dist, _ = masked_positive_dist.max(dim=1)

            masked_negative_dist = distances + (1.0 - negative_mask_float) * 1e9
            hardest_negative_dist, _ = masked_negative_dist.min(dim=1)

            neg_count = negative_mask_float.sum(dim=1)
            scaling = (distances * negative_mask_float).sum(dim=1) / (neg_count + 1e-16)

            triplet_loss = torch.clamp(
                (hardest_positive_dist - hardest_negative_dist) / (scaling + 1e-16) + self.config.margin,
                min=0.0
            )

            has_valid_positive = positive_mask.any(dim=1)
            has_valid_negative = negative_mask.any(dim=1)
            valid_anchor = has_valid_positive & has_valid_negative
            valid_anchor_float = valid_anchor.float()

            # Accumulate per anchor class
            anchor_labels_clipped = torch.clamp(anchor_labels, 0, max_classes - 1)
            
            for i in range(len(anchor_labels_clipped)):
                class_idx = anchor_labels_clipped[i].item()
                if valid_anchor_float[i] > 0:
                    loss_per_class[class_idx] += triplet_loss[i]
                    count_per_class[class_idx] += valid_anchor_float[i]

        # Compute mean loss per class
        mean_loss_per_class = loss_per_class / (count_per_class + 1e-16)

        # Weight each class equally
        valid_class_mask = count_per_class > 0
        num_valid_classes = valid_class_mask.float().sum()

        total_loss = (mean_loss_per_class * valid_class_mask.float()).sum()
        final_loss = total_loss / (num_valid_classes + 1e-16)

        return final_loss

    def _batch_all_triplet_loss_exact_class_balanced(
        self, 
        labels: torch.Tensor, 
        embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Batch all triplet loss with class-balanced weighting.
        """
        num_pixels = embeddings.size(0)
        batch_size = self.config.batch_size_for_exact
        num_batches = (num_pixels + batch_size - 1) // batch_size
        device = embeddings.device

        all_labels = labels
        max_classes = 100

        loss_per_class = torch.zeros(max_classes, device=device)
        count_per_class = torch.zeros(max_classes, device=device)
        positive_count_per_class = torch.zeros(max_classes, device=device)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_pixels)
            actual_batch_size = end_idx - start_idx

            anchor_embeddings = embeddings[start_idx:end_idx]
            anchor_labels = labels[start_idx:end_idx]

            distances = self._compute_distances_from_anchors(anchor_embeddings, embeddings)

            anchor_labels_exp = anchor_labels.unsqueeze(1)
            all_labels_exp = all_labels.unsqueeze(0)

            same_class = anchor_labels_exp == all_labels_exp
            diff_class = ~same_class
            anchor_non_bg = anchor_labels > 0
            anchor_non_bg_exp = anchor_non_bg.unsqueeze(1)

            batch_indices = torch.arange(start_idx, end_idx, device=device)
            all_indices = torch.arange(num_pixels, device=device)
            self_mask = batch_indices.unsqueeze(1) == all_indices.unsqueeze(0)
            not_self = ~self_mask

            positive_mask = same_class & not_self & anchor_non_bg_exp
            negative_mask = diff_class

            positive_mask_float = positive_mask.float()
            negative_mask_float = negative_mask.float()

            pos_count = positive_mask_float.sum(dim=1)
            neg_count = negative_mask_float.sum(dim=1)

            anchor_labels_clipped = torch.clamp(anchor_labels, 0, max_classes - 1)

            if self.config.remove_easy_triplets:
                for anchor_idx in range(actual_batch_size):
                    class_idx = anchor_labels_clipped[anchor_idx].item()
                    anchor_distances = distances[anchor_idx]
                    anchor_pos_mask = positive_mask[anchor_idx]
                    anchor_neg_mask = negative_mask[anchor_idx]

                    pos_indices = torch.where(anchor_pos_mask)[0]
                    neg_indices = torch.where(anchor_neg_mask)[0]

                    if len(pos_indices) == 0 or len(neg_indices) == 0:
                        continue

                    pos_dists = anchor_distances[pos_indices]
                    neg_dists = anchor_distances[neg_indices]

                    pos_expanded = pos_dists.unsqueeze(1)
                    neg_expanded = neg_dists.unsqueeze(0)

                    triplet_losses = pos_expanded - neg_expanded + self.config.margin
                    triplet_losses = torch.clamp(triplet_losses, min=0.0)

                    loss_per_class[class_idx] += triplet_losses.sum()
                    count_per_class[class_idx] += float(len(pos_indices) * len(neg_indices))
                    positive_count_per_class[class_idx] += (triplet_losses > 1e-16).float().sum()
            else:
                pos_dist_sum = (distances * positive_mask_float).sum(dim=1)
                neg_dist_sum = (distances * negative_mask_float).sum(dim=1)
                triplets_per_anchor = pos_count * neg_count

                anchor_losses = (
                    pos_dist_sum * neg_count -
                    neg_dist_sum * pos_count +
                    self.config.margin * triplets_per_anchor
                )

                for i in range(len(anchor_labels_clipped)):
                    class_idx = anchor_labels_clipped[i].item()
                    if triplets_per_anchor[i] > 0:
                        loss_per_class[class_idx] += anchor_losses[i]
                        count_per_class[class_idx] += triplets_per_anchor[i]

        # Compute mean loss per class
        if self.config.remove_easy_triplets:
            mean_loss_per_class = loss_per_class / (positive_count_per_class + 1e-16)
            valid_class_mask = positive_count_per_class > 0
        else:
            mean_loss_per_class = loss_per_class / (count_per_class + 1e-16)
            valid_class_mask = count_per_class > 0

        num_valid_classes = valid_class_mask.float().sum()
        total_loss = (mean_loss_per_class * valid_class_mask.float()).sum()
        final_loss = total_loss / (num_valid_classes + 1e-16)

        return final_loss

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute pixel-based triplet loss.
        
        Args:
            y_true: Ground truth labels (batch_size, h2, w2, num_whiskers) or (batch_size, num_whiskers, h2, w2)
            y_pred: Encoder predictions (batch_size, h, w, feature_dim) or (batch_size, feature_dim, h, w)
            
        Returns:
            Scalar loss value
        """
        # Handle channel-first format (PyTorch default) vs channel-last format (Keras default)
        # Assume Keras format: (batch, h, w, channels)
        y_true = y_true.float()
        y_pred = y_pred.float()
        
        # Get dimensions
        pred_h, pred_w = y_pred.shape[1], y_pred.shape[2]
        label_h, label_w = y_true.shape[1], y_true.shape[2]
        
        # Upsample embeddings to match label resolution
        y_pred_resized = self._resize_embeddings(y_pred, label_h, label_w)
        
        # Convert labels to class indices
        class_labels = self._labels_to_classes(y_true)
        
        # Get pixels
        if self.config.class_balanced_weighting:
            sampled_embeddings, sampled_labels = self._get_all_pixels(
                y_pred_resized, class_labels
            )
        elif self.config.use_balanced_sampling:
            if self.config.strict_per_class_balancing:
                sampled_embeddings, sampled_labels = self._sample_pixels_strict_balanced(
                    y_pred_resized, class_labels
                )
            else:
                sampled_embeddings, sampled_labels = self._sample_pixels_balanced(
                    y_pred_resized, class_labels
                )
        else:
            sampled_embeddings, sampled_labels = self._sample_pixels_per_class_simple(
                y_pred_resized, class_labels
            )

        # Compute loss
        if self.config.use_exact:
            if self.config.class_balanced_weighting:
                if self.config.triplet_strategy == "hard":
                    loss = self._batch_hard_triplet_loss_exact_class_balanced(sampled_labels, sampled_embeddings)
                elif self.config.triplet_strategy == "all":
                    loss = self._batch_all_triplet_loss_exact_class_balanced(sampled_labels, sampled_embeddings)
                else:
                    loss = self._batch_hard_triplet_loss_exact_class_balanced(sampled_labels, sampled_embeddings)
            else:
                if self.config.triplet_strategy == "hard":
                    loss = self._batch_hard_triplet_loss_exact(sampled_labels, sampled_embeddings)
                elif self.config.triplet_strategy == "all":
                    loss = self._batch_all_triplet_loss_exact(sampled_labels, sampled_embeddings)
                else:
                    loss = self._batch_hard_triplet_loss_exact(sampled_labels, sampled_embeddings)
        else:
            if self.config.triplet_strategy == "hard":
                loss = self._batch_hard_triplet_loss_custom(sampled_labels, sampled_embeddings)
            elif self.config.triplet_strategy == "all":
                loss = self._batch_all_triplet_loss_custom(sampled_labels, sampled_embeddings)
            else:
                loss = self._batch_hard_triplet_loss_custom(sampled_labels, sampled_embeddings)

        return loss
    
    def _resize_embeddings(
        self, 
        embeddings: torch.Tensor, 
        target_h: int, 
        target_w: int
    ) -> torch.Tensor:
        """
        Resize embeddings to match label dimensions using nearest neighbor interpolation.
        """
        original_h = embeddings.shape[1]
        original_w = embeddings.shape[2]
        
        if original_h == target_h and original_w == target_w:
            return embeddings
        
        # Reshape for interpolation: (batch, h, w, channels) -> (batch, channels, h, w)
        embeddings_perm = embeddings.permute(0, 3, 1, 2)
        
        # Use nearest neighbor interpolation
        resized = F.interpolate(
            embeddings_perm,
            size=(target_h, target_w),
            mode='nearest'
        )
        
        # Reshape back: (batch, channels, h, w) -> (batch, h, w, channels)
        resized = resized.permute(0, 2, 3, 1)
        
        return resized
    
    def _labels_to_classes(self, labels: torch.Tensor) -> torch.Tensor:
        """Convert multi-channel binary masks to class indices."""
        # labels: (batch_size, h, w, num_whiskers)
        
        # Sum across whisker channels to find background
        whisker_sum = labels.sum(dim=-1)  # (batch_size, h, w)
        
        # Use argmax to find which whisker channel is active
        whisker_classes = labels.argmax(dim=-1) + 1  # (batch_size, h, w)
        
        # Set background pixels to class 0
        class_labels = torch.where(
            whisker_sum > 0.5,
            whisker_classes,
            torch.zeros_like(whisker_classes)
        )
        
        return class_labels
    
    def _sample_pixels_per_class_simple(
        self, 
        embeddings: torch.Tensor, 
        class_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simplified pixel sampling per class."""
        device = embeddings.device
        feature_dim = embeddings.shape[-1]
        
        embeddings_flat = embeddings.reshape(-1, feature_dim)
        class_labels_flat = class_labels.reshape(-1)
        
        # Sample background pixels
        background_mask = class_labels_flat == 0
        background_indices = torch.where(background_mask)[0]
        
        num_background = min(self.config.background_pixels, len(background_indices))
        perm = torch.randperm(len(background_indices), device=device)[:num_background]
        sampled_background_indices = background_indices[perm]
        background_embeddings = embeddings_flat[sampled_background_indices]
        background_labels = torch.zeros(num_background, dtype=torch.long, device=device)
        
        # Sample whisker pixels
        non_background_mask = class_labels_flat > 0
        non_background_indices = torch.where(non_background_mask)[0]
        
        num_whisker = min(self.config.whisker_pixels, len(non_background_indices))
        perm = torch.randperm(len(non_background_indices), device=device)[:num_whisker]
        sampled_whisker_indices = non_background_indices[perm]
        whisker_embeddings = embeddings_flat[sampled_whisker_indices]
        whisker_labels = class_labels_flat[sampled_whisker_indices]
        
        # Concatenate
        final_embeddings = torch.cat([background_embeddings, whisker_embeddings], dim=0)
        final_labels = torch.cat([background_labels, whisker_labels], dim=0)
        
        # Handle edge case
        if final_embeddings.shape[0] == 0:
            final_embeddings = torch.zeros((2, feature_dim), dtype=torch.float32, device=device)
            final_labels = torch.tensor([0, 1], dtype=torch.long, device=device)
        
        return final_embeddings, final_labels

    def _get_all_pixels(
        self,
        embeddings: torch.Tensor,
        class_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get ALL pixels without any sampling."""
        feature_dim = embeddings.shape[-1]
        
        embeddings_flat = embeddings.reshape(-1, feature_dim)
        labels_flat = class_labels.reshape(-1)

        return embeddings_flat, labels_flat

    def _sample_pixels_balanced(
        self, 
        embeddings: torch.Tensor, 
        class_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Balanced pixel sampling."""
        device = embeddings.device
        feature_dim = embeddings.shape[-1]
        
        embeddings_flat = embeddings.reshape(-1, feature_dim)
        class_labels_flat = class_labels.reshape(-1)
        
        # Count pixels
        background_mask = class_labels_flat == 0
        background_count = background_mask.sum().item()
        
        non_background_mask = class_labels_flat > 0
        non_background_count = non_background_mask.sum().item()
        
        # Find minimum
        min_available = min(background_count, non_background_count)
        
        if self.config.max_samples_per_class is not None:
            samples_per_class = min(min_available, self.config.max_samples_per_class)
        else:
            samples_per_class = min_available
        
        samples_per_class = max(samples_per_class, 1)
        
        # Sample background
        background_indices = torch.where(background_mask)[0]
        perm = torch.randperm(len(background_indices), device=device)[:samples_per_class]
        sampled_background_indices = background_indices[perm]
        background_embeddings = embeddings_flat[sampled_background_indices]
        background_labels = torch.zeros(samples_per_class, dtype=torch.long, device=device)
        
        # Sample non-background
        non_background_indices = torch.where(non_background_mask)[0]
        perm = torch.randperm(len(non_background_indices), device=device)[:samples_per_class]
        sampled_whisker_indices = non_background_indices[perm]
        whisker_embeddings = embeddings_flat[sampled_whisker_indices]
        whisker_labels = class_labels_flat[sampled_whisker_indices]
        
        final_embeddings = torch.cat([background_embeddings, whisker_embeddings], dim=0)
        final_labels = torch.cat([background_labels, whisker_labels], dim=0)
        
        if final_embeddings.shape[0] == 0:
            final_embeddings = torch.zeros((2, feature_dim), dtype=torch.float32, device=device)
            final_labels = torch.tensor([0, 1], dtype=torch.long, device=device)
        
        return final_embeddings, final_labels
    
    def _sample_pixels_strict_balanced(
        self, 
        embeddings: torch.Tensor, 
        class_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Strict per-class balanced sampling."""
        device = embeddings.device
        feature_dim = embeddings.shape[-1]
        
        embeddings_flat = embeddings.reshape(-1, feature_dim)
        class_labels_flat = class_labels.reshape(-1)
        
        # Get unique classes and counts
        unique_classes = torch.unique(class_labels_flat)
        
        # Find minimum count across all classes
        min_count = float('inf')
        for class_id in unique_classes:
            count = (class_labels_flat == class_id).sum().item()
            min_count = min(min_count, count)
        
        if self.config.max_samples_per_class is not None:
            samples_per_class = min(int(min_count), self.config.max_samples_per_class)
        else:
            samples_per_class = int(min_count)
        
        samples_per_class = max(samples_per_class, 1)
        
        # Sample from each class
        sampled_embeddings_list = []
        sampled_labels_list = []
        
        for class_id in unique_classes:
            class_mask = class_labels_flat == class_id
            class_indices = torch.where(class_mask)[0]
            
            perm = torch.randperm(len(class_indices), device=device)[:samples_per_class]
            sampled_indices = class_indices[perm]
            
            class_embeddings = embeddings_flat[sampled_indices]
            class_labels_sampled = torch.full(
                (samples_per_class,), 
                class_id.item(), 
                dtype=torch.long, 
                device=device
            )
            
            sampled_embeddings_list.append(class_embeddings)
            sampled_labels_list.append(class_labels_sampled)
        
        if len(sampled_embeddings_list) > 0:
            final_embeddings = torch.cat(sampled_embeddings_list, dim=0)
            final_labels = torch.cat(sampled_labels_list, dim=0)
        else:
            final_embeddings = torch.zeros((2, feature_dim), dtype=torch.float32, device=device)
            final_labels = torch.tensor([0, 1], dtype=torch.long, device=device)
        
        return final_embeddings, final_labels


def create_pixel_triplet_loss(
    margin: float = 1.0,
    background_pixels: int = 1000,
    whisker_pixels: int = 500,
    max_samples_per_class: Optional[int] = None,
    use_balanced_sampling: bool = True,
    strict_per_class_balancing: bool = False,
    distance_metric: str = "euclidean",
    triplet_strategy: str = "semi_hard",
    reduction: str = "mean",
    remove_easy_triplets: bool = False,
    use_exact: bool = False,
    batch_size_for_exact: int = 100,
    class_balanced_weighting: bool = False,
    **kwargs
) -> PixelTripletLoss:
    """
    Create a pixel triplet loss with specified parameters.
    
    Args:
        margin: Triplet loss margin
        background_pixels: Number of background pixels to sample
        whisker_pixels: Number of whisker pixels to sample per whisker
        max_samples_per_class: Maximum samples per class for balanced sampling
        use_balanced_sampling: Whether to use balanced sampling
        strict_per_class_balancing: True per-class balancing
        distance_metric: Distance metric ('euclidean', 'cosine', 'manhattan')
        triplet_strategy: Triplet mining strategy ('semi_hard', 'hard', 'all')
        reduction: Loss reduction ('mean', 'sum', 'none')
        remove_easy_triplets: Whether to exclude easy triplets in batch_all mode
        use_exact: If True, use loop-based exact computation
        batch_size_for_exact: Number of anchor pixels to process per iteration
        class_balanced_weighting: If True, weight each class equally

    Returns:
        PixelTripletLoss instance
    """
    config = PixelTripletConfig(
        margin=margin,
        background_pixels=background_pixels,
        whisker_pixels=whisker_pixels,
        max_samples_per_class=max_samples_per_class,
        use_balanced_sampling=use_balanced_sampling,
        strict_per_class_balancing=strict_per_class_balancing,
        distance_metric=distance_metric,
        triplet_strategy=triplet_strategy,
        reduction=reduction,
        remove_easy_triplets=remove_easy_triplets,
        use_exact=use_exact,
        batch_size_for_exact=batch_size_for_exact,
        class_balanced_weighting=class_balanced_weighting,
        **kwargs
    )
    return PixelTripletLoss(config=config)
