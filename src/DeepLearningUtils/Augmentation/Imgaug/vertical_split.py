import cv2
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.parameters import StochasticParameter
import numpy as np
from imgaug.augmentables.batches import Batch
import imgaug.parameters as iap


class VerticalSplit(iaa.Augmenter):
    """
    Splits an image vertically at a specified percentage point, keeps one half (top or bottom),
    and resizes it to the original image dimensions.

    Parameters
    ----------
    percentage : float or tuple of float or list of float or StochasticParameter
        The percentage point along the vertical axis where the image is split.
        - If float: uses that fixed value (e.g., 0.5 = middle of the image)
        - If tuple (a, b): uniformly samples percentage from interval [a, b]
        - If list: randomly selects one value from the list
        - If StochasticParameter: uses that parameter to sample values

    keep_top : bool or float or StochasticParameter
        Whether to keep the top side of the split.
        - If bool: always keeps that side (True=top, False=bottom)
        - If float: probability of keeping the top side (vs. bottom side)
        - If StochasticParameter: uses that parameter to sample boolean values
    """

    def __init__(self, percentage=(0.25, 0.75), keep_top=0.5, 
                 name=None, deterministic=False, random_state=None):
        super(VerticalSplit, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state
        )

        # Check if percentage includes 0.0 or 1.0, make sure keep_top is not 0.0 or 1.0
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
            assert not (percentage[0] == 0.0 and keep_top == 1.0), (
                f"Expected keep_top to not be 0.0 when percentage[0] is 1.0, got {percentage} and {keep_top}."
            )
            assert not (percentage[1] == 1.0 and keep_top == 0.0), (
                f"Expected keep_top to not be 1.0 when percentage[1] is 0.0, got {percentage} and {keep_top}."
            )

        self.percentage = iap.handle_continuous_param(
            percentage, "percentage", value_range=(0, 1.0), tuple_to_uniform=True,
            list_to_choice=True
        )

        if isinstance(keep_top, bool):
            self.keep_top = iap.Deterministic(int(keep_top))
        elif ia.is_single_number(keep_top):
            assert 0 <= keep_top <= 1.0, (
                f"Expected keep_top to be in range [0.0, 1.0], got {keep_top}."
            )
            self.keep_top = iap.Binomial(keep_top)
        elif isinstance(keep_top, StochasticParameter):
            self.keep_top = keep_top
        else:
            raise Exception(
                f"Expected keep_top to be bool or number or StochasticParameter, got {type(keep_top)}."
            )

    def _augment_batch_(self, batch, random_state, parents, hooks):
        nb_images = batch.nb_rows
        percentages = self.percentage.draw_samples((nb_images,), random_state=random_state)
        keep_tops = self.keep_top.draw_samples((nb_images,), random_state=random_state)
        keep_tops = keep_tops > 0.5  # Convert to boolean

        # Augment images
        if batch.images is not None:
            batch.images = self._augment_images_by_samples(batch.images, percentages, keep_tops)

        # Augment segmentation maps
        if batch.segmentation_maps is not None:
            batch.segmentation_maps = self._augment_segmaps_by_samples(
                batch.segmentation_maps, percentages, keep_tops
            )

        # Augment keypoints
        if batch.keypoints is not None:
            batch.keypoints = self._augment_keypoints_by_samples(
                batch.keypoints, percentages, keep_tops
            )

        return batch

    def _augment_images_by_samples(self, images, percentages, keep_tops):
        result = []
        for image, percentage, keep_top in zip(images, percentages, keep_tops):
            height, width = image.shape[0:2]
            split_at = int(height * percentage)

            if keep_top:
                kept_part = image[0:split_at, :]
                scale_factor = height / split_at
            else:
                kept_part = image[split_at:height, :]
                scale_factor = height / (height - split_at)

            # Resize the kept part to the original dimensions
            resized_image = cv2.resize(
                kept_part,
                (width, height),
                interpolation=cv2.INTER_AREA)
            if len(resized_image.shape) == 2:
                resized_image = np.expand_dims(resized_image, axis=-1)

            result.append(resized_image)
        return result

    def _augment_segmaps_by_samples(self, segmaps, percentages, keep_tops):
        result = []
        for segmap, percentage, keep_top in zip(segmaps, percentages, keep_tops):
            # Get original dimensions
            split_at = int(segmap.shape[0] * percentage)

            # Create a new segmentation map
            segmap_new = segmap.deepcopy()
            arr = segmap_new.arr

            # Split and resize
            if keep_top:
                arr_split = arr[0:split_at, :]
            else:
                arr_split = arr[split_at:, :]

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

    def _augment_keypoints_by_samples(self, keypoints_on_images, percentages, keep_tops):
        result = []
        for kpsoi, percentage, keep_top in zip(keypoints_on_images, percentages, keep_tops):
            height = kpsoi.shape[0]
            split_at = int(height * percentage)

            # Create a new keypoints object
            kpsoi_new = kpsoi.deepcopy()

            # Filter and adjust keypoints
            if keep_top:
                # Keep only keypoints in the top part
                kps_new = [kp for kp in kpsoi_new.keypoints if kp.y < split_at]
                # Adjust coordinates based on resize
                scale_factor = height / split_at
                for kp in kps_new:
                    kp.y = kp.y * scale_factor
            else:
                # Keep only keypoints in the bottom part
                kps_new = [kp for kp in kpsoi_new.keypoints if kp.y >= split_at]
                # Adjust coordinates based on resize
                bottom_height = height - split_at
                scale_factor = height / bottom_height
                for kp in kps_new:
                    kp.y = (kp.y - split_at) * scale_factor

            kpsoi_new.keypoints = kps_new
            result.append(kpsoi_new)

        return result

    def get_parameters(self):
        return [self.percentage, self.keep_top]