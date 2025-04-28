"""
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
"""


import tensorflow as tf
import keras


def batch_hard_triplet_loss(labels, embeddings, margin, squared=False, normalize=False):
    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)
    if normalize:
        pairwise_dist = tf.math.divide_no_nan(pairwise_dist, tf.reduce_max(pairwise_dist))

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = keras.ops.cast(mask_anchor_positive ,"float32")

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

    # shape (batch_size, 1)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = keras.ops.cast(mask_anchor_negative ,"float32")

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)
    scaling = tf.reduce_mean(anchor_negative_dist, axis=1, keepdims=True) + 1e-16

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    # triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)
    # https://quaterion.qdrant.tech/tutorials/triplet_loss_trick
    triplet_loss = tf.maximum(
        (hardest_positive_dist - hardest_negative_dist) / scaling + margin,
        0.0)

    # Get final mean triplet loss
    triplet_loss = tf.reduce_mean(triplet_loss)

    return triplet_loss


def batch_all_triplet_loss(labels, embeddings, margin, squared=False, normalize=False, remove_negative=False):
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)
    if normalize:
        pairwise_dist = tf.math.divide_no_nan(pairwise_dist, tf.reduce_max(pairwise_dist))

    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    #anchor_positive_mask = _get_anchor_positive_triplet_mask(labels)
    #anchor_positive_mask = keras.ops.cast(anchor_positive_mask, "float32")
    #anchor_positive_dist = tf.multiply(anchor_positive_mask, anchor_positive_dist)

    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
    #anchor_negative_mask = _get_anchor_negative_triplet_mask(labels)
    #anchor_negative_mask = keras.ops.cast(anchor_negative_mask, "float32")
    #anchor_negative_dist = tf.multiply(anchor_negative_mask, anchor_negative_dist)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    mask = keras.ops.cast(mask ,"float32")
    triplet_loss = tf.multiply(mask, triplet_loss)

    num_valid_triplets = tf.reduce_sum(mask)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = keras.ops.cast(tf.greater(triplet_loss, 1e-16) ,"float32")
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Remove negative losses (i.e. the easy triplets)
    if remove_negative:
        triplet_loss = tf.maximum(triplet_loss, 0.0)

        # Get final mean triplet loss over the positive valid triplets
        # num positive triplets will be less than num_valid_triplets
        # As the model improves, there will be fewer positive triplets
        # because more will be classified as "easy" and removed
        triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    else:
        triplet_loss = tf.reduce_sum(triplet_loss) / (num_valid_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets


def hard_negative_triplet_mining(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings using hard negative mining.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
    """
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = keras.ops.cast(mask_anchor_positive, "float32")

    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = keras.ops.cast(mask_anchor_negative, "float32")

    max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)
    scaling = tf.reduce_mean(anchor_negative_dist, axis=1, keepdims=True) + 1e-16

    triplet_loss = tf.maximum(
        (anchor_positive_dist - hardest_negative_dist) / scaling + margin,
        0.0)

    return triplet_loss


def semi_hard_negative_triplet_mining(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings using semi-hard negative mining.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    mask = keras.ops.cast(mask, "float32")
    triplet_loss = tf.multiply(mask, triplet_loss)

    # Get the semi-hard negative
    # The semi-hard negative is the negative that is closest to the anchor
    # but still further than the positive
    # (thus triplet_loss < 0 and triplet_loss is the smallest)
    semi_hard_negative_dist = tf.reduce_max(
        tf.where(triplet_loss < 0, triplet_loss, tf.zeros_like(triplet_loss)),
        axis=1,
        keepdims=True)

    return semi_hard_negative_dist


def _pairwise_distances(embeddings, squared=False):
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
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.linalg.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = keras.ops.cast(tf.equal(distances, 0.0) ,"float32")
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    # Added by PMT
    labels_equal = tf.squeeze(labels_equal)

    # Combine the two masks
    mask = tf.logical_and(indices_not_equal, labels_equal)

    return mask


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_not(labels_equal)

    return mask


def batch_distance_loss(labels, embeddings, squared=False, normalize=False):
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)
    if normalize:
        pairwise_dist = tf.math.divide_no_nan(pairwise_dist, tf.reduce_max(pairwise_dist))

    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    #anchor_positive_mask = _get_anchor_positive_triplet_mask(labels)
    #anchor_positive_mask = keras.ops.cast(anchor_positive_mask, "float32")
    #anchor_positive_dist = tf.multiply(anchor_positive_mask, anchor_positive_dist)

    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
    #anchor_negative_mask = _get_anchor_negative_triplet_mask(labels)
    #anchor_negative_mask = keras.ops.cast(anchor_negative_mask, "float32")
    #anchor_negative_dist = tf.multiply(anchor_negative_mask, anchor_negative_dist)

    triplet_loss = anchor_positive_dist - anchor_negative_dist

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    mask = keras.ops.cast(mask ,"float32")
    triplet_loss = tf.multiply(mask, triplet_loss)

    num_valid_triplets = tf.reduce_sum(mask)

    # Count number of negative triplets (where triplet_loss > 0)
    valid_triplets = keras.ops.cast(tf.less(triplet_loss, -1e-16) ,"float32")
    num_negative_triplets = tf.reduce_sum(valid_triplets)
    fraction_negative_triplets = num_negative_triplets / (num_valid_triplets + 1e-16)

    triplet_loss = tf.reduce_sum(triplet_loss) / (num_valid_triplets + 1e-16)

    return triplet_loss, fraction_negative_triplets

def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.

    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size, batch_size, batch_size]
    """



    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    # Added by PMT
    label_equal = tf.squeeze(label_equal)

    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask
