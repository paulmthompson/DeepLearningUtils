

import cv2
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.parameters import StochasticParameter
import numpy as np
from imgaug.augmentables.batches import Batch
import imgaug.parameters as iap

class ThirdDiscard(iaa.Augmenter):
    """
    Divides an image into three parts, discards one outer part, and expands the middle part.
    
    The image is divided at percentage points p and (1-p). One of the outer thirds
    (from 0 to p, or from (1-p) to 1) is discarded. The middle third is expanded
    to maintain the original image size when combined with the kept outer third.
    
    Parameters
    ----------
    percentage : float or tuple of float or list of float or StochasticParameter
        The percentage point for division (must be between 0 and 0.5).
        - If float: uses that fixed value (e.g., 0.15 = divides at 15% and 85%)
        - If tuple (a, b): uniformly samples percentage from interval [a, b]
        - If list: randomly selects one value from the list
        - If StochasticParameter: uses that parameter to sample values
        
    keep_first : bool or float or StochasticParameter
        Whether to keep the first outer third of the split.
        - If bool: always keeps that side (True=first, False=last)
        - If float: probability of keeping the first side (vs. last side)
        - If StochasticParameter: uses that parameter to sample boolean values
        
    axis : str or int
        Axis along which to apply the split:
        - "horizontal" or 1: split along width (horizontal axis)
        - "vertical" or 0: split along height (vertical axis)
    """
    
    def __init__(self, percentage=(0.15, 0.3), keep_first=0.5, axis="horizontal",
                 name=None, deterministic=False, random_state=None):
        super(ThirdDiscard, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state
        )
        
        # Validate and handle the percentage parameter
        self.percentage = iap.handle_continuous_param(
            percentage, "percentage", value_range=(0, 0.5), tuple_to_uniform=True,
            list_to_choice=True
        )
        
        # Handle keep_first parameter
        if isinstance(keep_first, bool):
            self.keep_first = iap.Deterministic(int(keep_first))
        elif ia.is_single_number(keep_first):
            assert 0 <= keep_first <= 1.0, (
                f"Expected keep_first to be in range [0.0, 1.0], got {keep_first}."
            )
            self.keep_first = iap.Binomial(keep_first)
        elif isinstance(keep_first, StochasticParameter):
            self.keep_first = keep_first
        else:
            raise Exception(
                f"Expected keep_first to be bool or number or StochasticParameter, got {type(keep_first)}."
            )
        
        # Handle axis parameter
        if axis in ["horizontal", 1]:
            self.axis = 1  # horizontal
        elif axis in ["vertical", 0]:
            self.axis = 0  # vertical
        else:
            raise ValueError(f"Expected axis to be 'horizontal'/1 or 'vertical'/0, got {axis}")
    
    def _augment_batch_(self, batch, random_state, parents, hooks):
        nb_images = batch.nb_rows
        percentages = self.percentage.draw_samples((nb_images,), random_state=random_state)
        keep_firsts = self.keep_first.draw_samples((nb_images,), random_state=random_state)
        keep_firsts = keep_firsts > 0.5  # Convert to boolean
        
        # Augment images
        if batch.images is not None:
            batch.images = self._augment_images_by_samples(batch.images, percentages, keep_firsts)
        
        # Augment segmentation maps
        if batch.segmentation_maps is not None:
            batch.segmentation_maps = self._augment_segmaps_by_samples(
                batch.segmentation_maps, percentages, keep_firsts
            )
        
        # Augment keypoints
        if batch.keypoints is not None:
            batch.keypoints = self._augment_keypoints_by_samples(
                batch.keypoints, percentages, keep_firsts
            )
        
        return batch
    
    def _augment_images_by_samples(self, images, percentages, keep_firsts):
        result = []
        for image, percentage, keep_first in zip(images, percentages, keep_firsts):
            # Get dimensions
            if self.axis == 1:  # horizontal
                size = image.shape[1]
                split1 = int(size * percentage)
                split2 = int(size * (1 - percentage))
                
                # Extract the three parts
                first_part = image[:, 0:split1]
                middle_part = image[:, split1:split2]
                last_part = image[:, split2:]
                
                # Determine which parts to keep and resize
                if keep_first:
                    kept_outer = first_part
                    middle_width = split2 - split1
                    new_middle_width = size - split1
                    scale_factor = new_middle_width / middle_width
                    resized_middle = cv2.resize(
                        middle_part, 
                        (new_middle_width, image.shape[0]), 
                        interpolation=cv2.INTER_AREA
                    )
                    if middle_part.shape[-1] == 1:
                        resized_middle = resized_middle[..., np.newaxis]
                    # Combine parts
                    result_img = np.concatenate([kept_outer, resized_middle], axis=1)
                else:
                    kept_outer = last_part
                    middle_width = split2 - split1
                    new_middle_width = size - (size - split2)
                    scale_factor = new_middle_width / middle_width
                    resized_middle = cv2.resize(
                        middle_part, 
                        (new_middle_width, image.shape[0]), 
                        interpolation=cv2.INTER_AREA
                    )
                    if middle_part.shape[-1] == 1:
                        resized_middle = resized_middle[..., np.newaxis]
                    # Combine parts
                    result_img = np.concatenate([resized_middle, kept_outer], axis=1)
            
            else:  # vertical
                size = image.shape[0]
                split1 = int(size * percentage)
                split2 = int(size * (1 - percentage))
                
                # Extract the three parts
                first_part = image[0:split1, :]
                middle_part = image[split1:split2, :]
                last_part = image[split2:, :]
                
                # Determine which parts to keep and resize
                if keep_first:
                    kept_outer = first_part
                    middle_height = split2 - split1
                    new_middle_height = size - split1
                    scale_factor = new_middle_height / middle_height
                    resized_middle = cv2.resize(
                        middle_part, 
                        (image.shape[1], new_middle_height), 
                        interpolation=cv2.INTER_AREA
                    )
                    if middle_part.shape[-1] == 1:
                        resized_middle = resized_middle[..., np.newaxis]
                    # Combine parts
                    result_img = np.concatenate([kept_outer, resized_middle], axis=0)
                else:
                    kept_outer = last_part
                    middle_height = split2 - split1
                    new_middle_height = size - (size - split2)
                    scale_factor = new_middle_height / middle_height
                    resized_middle = cv2.resize(
                        middle_part, 
                        (image.shape[1], new_middle_height), 
                        interpolation=cv2.INTER_AREA
                    )
                    if middle_part.shape[-1] == 1:
                        resized_middle = resized_middle[..., np.newaxis]
                    # Combine parts
                    result_img = np.concatenate([resized_middle, kept_outer], axis=0)
            
            result.append(result_img)
        
        return result
    
    def _augment_segmaps_by_samples(self, segmaps, percentages, keep_firsts):
        result = []
        for segmap, percentage, keep_first in zip(segmaps, percentages, keep_firsts):
            segmap_new = segmap.deepcopy()
            arr = segmap_new.arr
            
            # Get dimensions
            if self.axis == 1:  # horizontal
                size = arr.shape[1]
                split1 = int(size * percentage)
                split2 = int(size * (1 - percentage))
                
                # Extract the three parts
                first_part = arr[:, 0:split1]
                middle_part = arr[:, split1:split2]
                last_part = arr[:, split2:]
                
                # Determine which parts to keep and resize
                if keep_first:
                    kept_outer = first_part
                    middle_width = split2 - split1
                    new_middle_width = size - split1
                    resized_middle = cv2.resize(
                        middle_part, 
                        (new_middle_width, arr.shape[0]), 
                        interpolation=cv2.INTER_NEAREST
                    )
                    if middle_part.shape[-1] == 1:
                        resized_middle = resized_middle[..., np.newaxis]
                    # Combine parts
                    result_arr = np.concatenate([kept_outer, resized_middle], axis=1)
                else:
                    kept_outer = last_part
                    middle_width = split2 - split1
                    new_middle_width = size - (size - split2)
                    resized_middle = cv2.resize(
                        middle_part, 
                        (new_middle_width, arr.shape[0]), 
                        interpolation=cv2.INTER_NEAREST
                    )
                    if middle_part.shape[-1] == 1:
                        resized_middle = resized_middle[..., np.newaxis]
                    # Combine parts
                    result_arr = np.concatenate([resized_middle, kept_outer], axis=1)
            
            else:  # vertical
                size = arr.shape[0]
                split1 = int(size * percentage)
                split2 = int(size * (1 - percentage))
                
                # Extract the three parts
                first_part = arr[0:split1, :]
                middle_part = arr[split1:split2, :]
                last_part = arr[split2:, :]
                
                # Determine which parts to keep and resize
                if keep_first:
                    kept_outer = first_part
                    middle_height = split2 - split1
                    new_middle_height = size - split1
                    resized_middle = cv2.resize(
                        middle_part, 
                        (arr.shape[1], new_middle_height), 
                        interpolation=cv2.INTER_NEAREST
                    )
                    if middle_part.shape[-1] == 1:
                        resized_middle = resized_middle[..., np.newaxis]
                    # Combine parts
                    result_arr = np.concatenate([kept_outer, resized_middle], axis=0)
                else:
                    kept_outer = last_part
                    middle_height = split2 - split1
                    new_middle_height = size - (size - split2)
                    resized_middle = cv2.resize(
                        middle_part, 
                        (arr.shape[1], new_middle_height), 
                        interpolation=cv2.INTER_NEAREST
                    )
                    if middle_part.shape[-1] == 1:
                        resized_middle = resized_middle[..., np.newaxis]
                    # Combine parts
                    result_arr = np.concatenate([resized_middle, kept_outer], axis=0)
            
            segmap_new.arr = result_arr
            result.append(segmap_new)
        
        return result
    
    def _augment_keypoints_by_samples(self, keypoints_on_images, percentages, keep_firsts):
        result = []
        for kpsoi, percentage, keep_first in zip(keypoints_on_images, percentages, keep_firsts):
            kpsoi_new = kpsoi.deepcopy()
            
            # Get dimensions
            if self.axis == 1:  # horizontal
                size = kpsoi.shape[1]
                split1 = int(size * percentage)
                split2 = int(size * (1 - percentage))
                middle_width = split2 - split1
                
                # Create new keypoints list
                kps_new = []
                
                if keep_first:
                    # Keep keypoints in first part
                    for kp in kpsoi_new.keypoints:
                        if kp.x < split1:
                            # Keypoints in the first part remain unchanged
                            kps_new.append(kp)
                        elif split1 <= kp.x < split2:
                            # Keypoints in middle part need to be scaled
                            new_middle_width = size - split1
                            scale_factor = new_middle_width / middle_width
                            new_x = split1 + (kp.x - split1) * scale_factor
                            kp_new = kp.deepcopy()
                            kp_new.x = new_x
                            kps_new.append(kp_new)
                        # Keypoints in the last part are discarded
                else:
                    # Keep keypoints in last part
                    for kp in kpsoi_new.keypoints:
                        if kp.x >= split2:
                            # Keypoints in the last part need to be shifted
                            new_middle_width = split2
                            scale_factor = new_middle_width / middle_width
                            shifted_x = (kp.x - split2) + new_middle_width
                            kp_new = kp.deepcopy()
                            kp_new.x = shifted_x
                            kps_new.append(kp_new)
                        elif split1 <= kp.x < split2:
                            # Keypoints in middle part need to be scaled
                            new_middle_width = split2
                            scale_factor = new_middle_width / middle_width
                            new_x = (kp.x - split1) * scale_factor
                            kp_new = kp.deepcopy()
                            kp_new.x = new_x
                            kps_new.append(kp_new)
                        # Keypoints in the first part are discarded
            
            else:  # vertical
                size = kpsoi.shape[0]
                split1 = int(size * percentage)
                split2 = int(size * (1 - percentage))
                middle_height = split2 - split1
                
                # Create new keypoints list
                kps_new = []
                
                if keep_first:
                    # Keep keypoints in first part
                    for kp in kpsoi_new.keypoints:
                        if kp.y < split1:
                            # Keypoints in the first part remain unchanged
                            kps_new.append(kp)
                        elif split1 <= kp.y < split2:
                            # Keypoints in middle part need to be scaled
                            new_middle_height = size - split1
                            scale_factor = new_middle_height / middle_height
                            new_y = split1 + (kp.y - split1) * scale_factor
                            kp_new = kp.deepcopy()
                            kp_new.y = new_y
                            kps_new.append(kp_new)
                        # Keypoints in the last part are discarded
                else:
                    # Keep keypoints in last part
                    for kp in kpsoi_new.keypoints:
                        if kp.y >= split2:
                            # Keypoints in the last part need to be shifted
                            new_middle_height = split2
                            scale_factor = new_middle_height / middle_height
                            shifted_y = (kp.y - split2) + new_middle_height
                            kp_new = kp.deepcopy()
                            kp_new.y = shifted_y
                            kps_new.append(kp_new)
                        elif split1 <= kp.y < split2:
                            # Keypoints in middle part need to be scaled
                            new_middle_height = split2
                            scale_factor = new_middle_height / middle_height
                            new_y = (kp.y - split1) * scale_factor
                            kp_new = kp.deepcopy()
                            kp_new.y = new_y
                            kps_new.append(kp_new)
                        # Keypoints in the first part are discarded
            
            kpsoi_new.keypoints = kps_new
            result.append(kpsoi_new)
        
        return result
    
    def get_parameters(self):
        return [self.percentage, self.keep_first]