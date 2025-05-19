
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.parameters import StochasticParameter
import numpy as np
from imgaug.augmentables.batches import Batch
import imgaug.parameters as iap


class HorizontalSplit(iaa.Augmenter):
    """
    Splits an image horizontally at a specified percentage point, keeps one half (left or right),
    and resizes it to the original image dimensions.

    Parameters
    ----------
    percentage : float or tuple of float or list of float or StochasticParameter
        The percentage point along the horizontal axis where the image is split.
        - If float: uses that fixed value (e.g., 0.5 = middle of the image)
        - If tuple (a, b): uniformly samples percentage from interval [a, b]
        - If list: randomly selects one value from the list
        - If StochasticParameter: uses that parameter to sample values

    keep_left : bool or float or StochasticParameter
        Whether to keep the left side of the split.
        - If bool: always keeps that side (True=left, False=right)
        - If float: probability of keeping the left side (vs. right side)
        - If StochasticParameter: uses that parameter to sample boolean values
    """

    def __init__(self, percentage=(0.25, 0.75), keep_left=0.5,
                 name=None, deterministic=False, random_state=None):
        super(HorizontalSplit, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state
        )

        #if percentage includes 0.0 or 1.0, make sure keep left is not 0.0 or 1.0
        if isinstance(percentage, (list, tuple)):
            assert len(percentage) == 2, (
                f"Expected percentage to be a list or tuple of length 2, got {len(percentage)}."
            )
            assert percentage[0] < percentage[1], (
                f"Expected percentage[0] < percentage[1], got {percentage}."
            )
            assert 0.0 <= percentage[0] <= 1.0 and 0.0 <= percentage[1] <= 1.0, (
                f"Expected percentage values to be in range (0.0, 1.0), got {percentage}."
            )
            assert not (percentage[0] == 0.0 and keep_left == 1.0), (
                f"Expected keep_left to not be 0.0 when percentage[0] is 1.0, got {percentage} and {keep_left}."
            )
            assert not (percentage[1] == 1.0 and keep_left == 0.0), (
                f"Expected keep_left to not be 1.0 when percentage[1] is 0.0, got {percentage} and {keep_left}."
            )

        self.percentage = iap.handle_continuous_param(
            percentage, "percentage", value_range=(0, 1.0), tuple_to_uniform=True,
            list_to_choice=True
        )

        if isinstance(keep_left, bool):
            self.keep_left = iap.Deterministic(int(keep_left))
        elif ia.is_single_number(keep_left):
            assert 0 <= keep_left <= 1.0, (
                f"Expected keep_left to be in range [0.0, 1.0], got {keep_left}."
            )
            self.keep_left = iap.Binomial(keep_left)
        elif isinstance(keep_left, StochasticParameter):
            self.keep_left = keep_left
        else:
            raise Exception(
                f"Expected keep_left to be bool or number or StochasticParameter, got {type(keep_left)}."
            )

    def _augment_batch_(self, batch, random_state, parents, hooks):
        nb_images = batch.nb_rows
        percentages = self.percentage.draw_samples((nb_images,), random_state=random_state)
        keep_lefts = self.keep_left.draw_samples((nb_images,), random_state=random_state)
        keep_lefts = keep_lefts > 0.5  # Convert to boolean

        # Augment images
        if batch.images is not None:
            batch.images = self._augment_images_by_samples(batch.images, percentages, keep_lefts)

        # Augment segmentation maps
        if batch.segmentation_maps is not None:
            batch.segmentation_maps = self._augment_segmaps_by_samples(
                batch.segmentation_maps, percentages, keep_lefts
            )

        # Augment keypoints
        if batch.keypoints is not None:
            batch.keypoints = self._augment_keypoints_by_samples(
                batch.keypoints, percentages, keep_lefts
            )

        return batch

    def _augment_images_by_samples(self, images, percentages, keep_lefts):
        result = []
        for image, percentage, keep_left in zip(images, percentages, keep_lefts):
            height, width = image.shape[0:2]
            split_at = int(width * percentage)

            if keep_left:
                kept_part = image[:, 0:split_at]
                scale_factor = width / split_at
            else:
                kept_part = image[:, split_at:width]
                scale_factor = width / (width - split_at)

            # Resize the kept part to the original width
            resized_image = cv2.resize(
                kept_part,
                (width, height),
                interpolation=cv2.INTER_AREA)
            if len(resized_image.shape) == 2:
                resized_image = np.expand_dims(resized_image, axis=-1)

            result.append(resized_image)

        return result

    def _augment_segmaps_by_samples(self, segmaps, percentages, keep_lefts):
        result = []
        for segmap, percentage, keep_left in zip(segmaps, percentages, keep_lefts):
            # Get original dimensions
            height, width = segmap.shape[0:2]
            split_at = int(width * percentage)

            # Create a new segmentation map
            segmap_new = segmap.deepcopy()
            arr = segmap_new.arr

            # Split and resize
            if keep_left:
                arr_split = arr[:, 0:split_at]
            else:
                arr_split = arr[:, split_at:width]

            # Resize using nearest neighbor for segmentation maps
            arr_resized = cv2.resize(
                arr_split,
                (segmap_new.arr.shape[1], segmap_new.arr.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            if len(arr_resized.shape) == 2:
                arr_resized = np.expand_dims(arr_resized, axis=-1)

            segmap_new.arr = arr_resized
            result.append(segmap_new)

        return result

    def _augment_keypoints_by_samples(self, keypoints_on_images, percentages, keep_lefts):
        result = []
        for kpsoi, percentage, keep_left in zip(keypoints_on_images, percentages, keep_lefts):
            width = kpsoi.shape[1]
            split_at = int(width * percentage)

            # Create a new keypoints object
            kpsoi_new = kpsoi.deepcopy()

            # Filter and adjust keypoints
            if keep_left:
                # Keep only keypoints in the left part
                kps_new = [kp for kp in kpsoi_new.keypoints if kp.x < split_at]
                # Adjust coordinates based on resize
                scale_factor = width / split_at
                for kp in kps_new:
                    kp.x = kp.x * scale_factor
            else:
                # Keep only keypoints in the right part
                kps_new = [kp for kp in kpsoi_new.keypoints if kp.x >= split_at]
                # Adjust coordinates based on resize
                right_width = width - split_at
                scale_factor = width / right_width
                for kp in kps_new:
                    kp.x = (kp.x - split_at) * scale_factor

            kpsoi_new.keypoints = kps_new
            result.append(kpsoi_new)

        return result

    def get_parameters(self):
        return [self.percentage, self.keep_left]