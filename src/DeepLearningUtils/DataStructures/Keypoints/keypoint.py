import csv
from collections import OrderedDict

import numpy as np


class Keypoint:
    """
    Represents a single keypoint detected in an image.

    This class extracts the coordinates of the maximum intensity point in an image.
    When multiple points share the maximum value, it averages their coordinates.

    Parameters
    ----------
    image : numpy.ndarray
        2D array representing an image or probability map

    Attributes
    ----------
    prob : float
        Probability/confidence value of the keypoint
    x : int
        X-coordinate of the keypoint
    y : int
        Y-coordinate of the keypoint

    Raises
    ------
    ValueError
        If the input is not a valid 2D numpy array or is empty
    """

    def __init__(self, image):
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a numpy array")

        if image.size == 0:
            raise ValueError("Input image cannot be empty")

        if image.ndim != 2:
            raise ValueError(f"Expected 2D array, got {image.ndim}D array")

        self.prob = np.max(image)
        coords = np.argwhere(image == self.prob)

        if coords.size == 0:
            # This should not happen if image is not empty, but just in case
            self.x = 0
            self.y = 0
            self.prob = 0.0
        elif coords.shape[0] > 1:
            # Multiple points with max value - take average
            self.x = round(np.mean(coords[:, 0]))
            self.y = round(np.mean(coords[:, 1]))
        else:
            self.x = coords[0, 0]
            self.y = coords[0, 1]

    def __repr__(self):
        """Return string representation of the keypoint."""
        return f"Keypoint(x={self.x}, y={self.y}, prob={self.prob:.4f})"

    def as_tuple(self):
        """
        Returns the keypoint coordinates as a tuple.

        Returns
        -------
        tuple
            (x, y) coordinates of the keypoint
        """
        return (self.x, self.y)


class VideoKeypoints:
    def __init__(
        self,
        video_height,
        video_width,
    ):
        self.keypoints = []
        self.frames = []
        self._video_height = video_height
        self._video_width = video_width

    def add_frames(
        self,
        video_labels,
        frame_offset,
    ):
        """

        Parameters
        ----------
        video_labels
        frame_offset: int
            This should indicate the value of the first frame in the video_labels
            data structure relative to the beginning of the entire video. If video labels contains
            frames 4, 5, 6, and 7, then this value should be 4.

        Returns
        -------

        """
        for i in range(0, video_labels.shape[0]):
            this_keypoint = Keypoint(video_labels[i, :, :])
            self.keypoints.append(this_keypoint)
            self.frames.append(i + frame_offset)

    def to_csv(
        self,
        csv_path,
        scale_height=1.0,
        scale_width=1.0,
        threshold=0.0,
        delimiter=" ",
    ):
        """

        Parameters
        ----------
        csv_path: string

        Returns
        -------

        """
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=delimiter)
            writer.writerow(
                [
                    "Frame",
                    "X",
                    "Y",
                    "Probability",
                ]
            )
            for i in range(0, len(self.frames)):
                if self.keypoints[i].prob < threshold:
                    continue
                writer.writerow(
                    [
                        self.frames[i],
                        self.keypoints[i].x * scale_width,
                        self.keypoints[i].y * scale_height,
                        str(round(self.keypoints[i].prob, 2)),
                    ]
                )


def read_keypoint_csv(
        csv_path,
        delimiter=" ",):
    keypoint_list = []

    csvfile = open(csv_path, newline="")
    reader = csv.reader(csvfile, delimiter=delimiter)
    for row in reader:
        keypoint_list.append(row)

    print(f"Read {len(keypoint_list)} keypoints from {csv_path}")

    keypoint_list = keypoint_list[1:]  # Remove header

    keypoint_coordinates = OrderedDict()

    for i in range(len(keypoint_list)):
        frame = keypoint_list[i][0]
        keypoint_coordinates[int(frame)] = [
            round(float(keypoint_list[i][2])),
            round(float(keypoint_list[i][1])),
        ]

    return keypoint_coordinates
