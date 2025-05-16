
import random

"""
Here we have several biased samplers.

"""

"""
Video Sequence Samplers
"""


def random_sampler_with_sequence(
        indexes,
        N_samples,
        seq_len,
        random_seed_value):
    """
    Given a list of indexes of frames from a longer video, we wish
    to sample N_samples which will include a sequence of seq_len
    frames with the index being the middle frame.

    This sampler will ensure that samples do not overlap, or in other words
    that the preceding frames of one sample
    are not the subsequent frames of another sample.

    Output: A sorted list of indexes of frames to keep. Each is the
    middle frame of a sequence of seq_len frames.

    Args:
        indexes: A list of indexes of frames from a longer video
        N_samples: The number of samples to take
        seq_len: The length of each sequence for the model.
        random_seed_value: The random seed value to use for the random sampler

    """

    random_sampler = random.Random(random_seed_value)

    keep_subset = []

    keep_indexes_to_sample = indexes.copy()

    while len(keep_subset) < N_samples:

        sampled_index = random_sampler.sample(keep_indexes_to_sample, 1)

        keep_subset.append(sampled_index[0])

        for j in range(-int(seq_len/2), int(seq_len/2)+1):
            if (sampled_index[0] - j) in keep_indexes_to_sample:
                keep_indexes_to_sample.remove(sampled_index[0] - j)

    keep_subset.sort()
    return keep_subset


def random_sampler_with_sequence_probability(
        indexes,
        N_samples,
        seq_len,
        random_seed_value,
        probabilities,):
    """



    Each element of keep_indexes to sample has a probability of being sampled
    based on some category

    """

    if len(indexes) != len(probabilities):
        raise ValueError("Length of indexes and probabilities must be the same")

    random_sampler = random.Random(random_seed_value)

    keep_subset = []

    keep_indexes_to_sample = indexes.copy()
    probabilities_to_sample = probabilities.tolist()

    while len(keep_subset) < N_samples:

        sampled_index = random_sampler.choices(
            keep_indexes_to_sample,
            weights=probabilities_to_sample)

        keep_subset.append(sampled_index[0])

        for j in range(-int(seq_len/2), int(seq_len/2)+1):
            if (sampled_index[0] - j) in keep_indexes_to_sample:

                # Find index of adjacent sample and set probability to 0
                adjacent_index = keep_indexes_to_sample.index(sampled_index[0] - j)

                probabilities_to_sample[adjacent_index] = 0

    keep_subset.sort()
    return keep_subset
