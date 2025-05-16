from collections import OrderedDict
import os

import h5py
import numpy as np
from skimage.morphology import skeletonize


class SparseMask:
    """
    Given an image mask with values (0 - 1), we can save only the
    pixels that are above some threshold value
    """

    def __init__(self):
        self.x_indexes = []
        self.y_indexes = []
        self.probs = []
        self.height = 0
        self.width = 0

    def create_skeleton_from_image(self, image):
        """

        Parameters
        ----------
        image

        Returns
        -------

        """
        self.height = image.shape[0]
        self.width = image.shape[1]
        image = (image * 255).astype(np.uint8)
        medial_axis_mask = skeletonize(image)

        indexes = np.argwhere(medial_axis_mask > 0)

        if len(indexes) > 0:
            self.probs = image[medial_axis_mask > 0] / 255
            self.x_indexes = indexes[:, 0]
            self.y_indexes = indexes[:, 1]

    def create_mask_from_image(
        self,
        image,
        threshold,
    ):
        """

        Parameters
        ----------
        image: np.ndarray
            2D array with values (0 - 1)
        threshold: float
        """
        self.height = image.shape[0]
        self.width = image.shape[1]

        indexes = np.argwhere(image > threshold)

        if len(indexes) > 0:
            self.probs = image[image > threshold]
            self.x_indexes = indexes[:, 0]
            self.y_indexes = indexes[:, 1]

    def create_image(self):
        """

        Returns
        -------
        np.ndarray[float]
            This is a 3 dimension array representing the mask of the
            image on the full size image. (height x width x 1)
        """

        image = np.zeros((self.height, self.width, 1), dtype=np.float32)

        image[self.x_indexes, self.y_indexes, 0] = self.probs

        return image

    def mask_exists(self):
        """

        Returns
        -------
        bool:
            True indicates that there is a mask, whereas False indicates there is
            no mask for this frame
        """
        if len(self.probs) > 0:
            return True
        else:
            return False


class SparseVideoMask:
    """
    We can keep a collection of sparse masks from a series of image
    frames. This data structure only adds SparseMask objects when
    the mask for a particular frame is not empty (there are some pixels
    above the threshold).
    """

    def __init__(
        self,
        threshold,
        video_height,
        video_width,
    ):
        """

        Parameters
        ----------
        threshold: float
        video_height: int
        video_width: int
        """
        self.frames = []
        self.masks = OrderedDict()
        self.threshold = threshold
        self.video_height = video_height
        self.video_width = video_width

    def add_skeletons(
        self,
        video_labels,
        frame_offset,
    ):
        """

        Parameters
        ----------
        video_labels: np.ndarray
            3D array (frames x height x width). Each value is
            pixel (0-1).
        frame_offset: int
            This should indicate the value of the first frame in the video_labels
            data structure relative to the beginning of the entire video. If video labels contains
            frames 4, 5, 6, and 7, then this value should be 4.

        Returns
        -------

        """

        for i in range(0, video_labels.shape[0]):
            image_mask = SparseMask()
            image_mask.create_skeleton_from_image(
                video_labels[i, :, :],
            )
            if image_mask.mask_exists():
                self.masks[i + frame_offset] = image_mask
                self.frames.append(i + frame_offset)

    def add_frames(
        self,
        video_labels,
        frame_offset,
    ):
        """

        Parameters
        ----------
        video_labels: np.ndarray
            3D array (frames x height x width). Each value is
            pixel (0-1).
        frame_offset: int
            This should indicate the value of the first frame in the video_labels
            data structure relative to the beginning of the entire video. If video labels contains
            frames 4, 5, 6, and 7, then this value should be 4.

        Returns
        -------

        """
        for i in range(0, video_labels.shape[0]):
            image_mask = SparseMask()
            image_mask.create_mask_from_image(
                video_labels[i, :, :],
                self.threshold,
            )
            if image_mask.mask_exists():
                self.masks[i + frame_offset] = image_mask
                self.frames.append(i + frame_offset)

    def to_hdf5(self, hdf5_file):
        """

        Save masks to HDF5 file

        Parameters
        ----------
        hdf5_file: string

        Returns
        -------

        """
        try:
            os.remove(hdf5_file)
        except OSError:
            pass

        with h5py.File(hdf5_file, "w") as f:
            frames = np.array(self.frames, dtype=np.int64)
            f.create_dataset("frames", data=frames, dtype=np.int64)

            dt_int = h5py.vlen_dtype(np.dtype("int64"))
            dt_float = h5py.vlen_dtype(np.dtype("float32"))

            x_indexes = self.get_x_indexes()
            f.create_dataset("heights", data=x_indexes, dtype=dt_int)

            y_indexes = self.get_y_indexes()
            f.create_dataset("widths", data=y_indexes, dtype=dt_int)

            probs = self.get_probs()
            f.create_dataset("probs", data=probs, dtype=dt_float)

    def from_hdf5(self, hdf5_file):
        """
        Load masks from HDF5 file

        Parameters
        ----------
        hdf5_file

        Returns
        -------

        """
        with h5py.File(hdf5_file, "r") as f:
            self.frames = f.get("frames")
            heights = f.get("heights")
            widths = f.get("widths")
            probs = f.get("probs")

            for i in range(0, len(heights)):
                mask = SparseMask()
                mask.x_indexes = heights[i]
                mask.y_indexes = widths[i]
                mask.probs = probs[i]
                mask.height = self.video_height
                mask.width = self.video_width
                self.masks[self.frames[i]] = mask

    def get_x_indexes(self):
        x_indexes = []
        for frame_id in self.frames:
            x_indexes.append(self.masks[frame_id].x_indexes)

        x_indexes = np.array(x_indexes, dtype=np.object_)
        return x_indexes

    def get_y_indexes(self):
        y_indexes = []
        for frame_id in self.frames:
            y_indexes.append(self.masks[frame_id].y_indexes)

        y_indexes = np.array(y_indexes, dtype=np.object_)
        return y_indexes

    def get_probs(self):
        probs = []
        for frame_id in self.frames:
            probs.append(self.masks[frame_id].probs)

        probs = np.array(probs, dtype=np.object_)
        return probs
