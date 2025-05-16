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
    """
    Stores and manages keypoints across multiple frames of a video.

    This class provides functionality to track keypoints across frames
    and export them to various formats.

    Parameters
    ----------
    video_height : int
        Height of the video frames in pixels
    video_width : int
        Width of the video frames in pixels

    Attributes
    ----------
    keypoints : list
        List of Keypoint objects for each frame
    frames : list
        List of frame numbers corresponding to each keypoint
    """

    def __init__(self, video_height, video_width):
        if not isinstance(video_height, (int, float)) or video_height <= 0:
            raise ValueError("Video height must be a positive number")
        if not isinstance(video_width, (int, float)) or video_width <= 0:
            raise ValueError("Video width must be a positive number")

        self.keypoints = []
        self.frames = []
        self._video_height = video_height
        self._video_width = video_width

    def add_frames(self, video_labels, frame_offset):
        """
        Extract keypoints from multiple frames and add them to the collection.

        Parameters
        ----------
        video_labels : numpy.ndarray
            3D array of shape (n_frames, height, width) containing probability maps
        frame_offset : int
            The value of the first frame in video_labels relative to the beginning
            of the entire video. For example, if video_labels contains frames 4-7,
            frame_offset should be 4.

        Returns
        -------
        int
            Number of frames added

        Raises
        ------
        ValueError
            If video_labels is not a 3D numpy array
        """
        if not isinstance(video_labels, np.ndarray):
            raise ValueError("Input must be a numpy array")

        if video_labels.ndim != 3:
            raise ValueError(f"Expected 3D array, got {video_labels.ndim}D array")

        for i in range(video_labels.shape[0]):
            this_keypoint = Keypoint(video_labels[i, :, :])
            self.keypoints.append(this_keypoint)
            self.frames.append(i + frame_offset)

        return video_labels.shape[0]

    def to_csv(self, csv_path, scale_height=1.0, scale_width=1.0, threshold=0.0, delimiter=" "):
        """
        Export keypoints to a CSV file.

        Parameters
        ----------
        csv_path : str
            Path to save the CSV file
        scale_height : float, optional
            Scaling factor for height values, default is 1.0
        scale_width : float, optional
            Scaling factor for width values, default is 1.0
        threshold : float, optional
            Minimum probability threshold for keypoints to include, default is 0.0
        delimiter : str, optional
            CSV delimiter to use, default is space

        Returns
        -------
        int
            Number of keypoints written to file

        Raises
        ------
        IOError
            If the file cannot be written
        """
        try:
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile, delimiter=delimiter)
                writer.writerow(["Frame", "X", "Y", "Probability"])

                count = 0
                for i in range(len(self.frames)):
                    if self.keypoints[i].prob < threshold:
                        continue

                    writer.writerow([
                        self.frames[i],
                        self.keypoints[i].y * scale_width,  # Note: x and y appear swapped
                        self.keypoints[i].x * scale_height,
                        str(round(self.keypoints[i].prob, 2))
                    ])
                    count += 1

            return count

        except IOError as e:
            raise IOError(f"Error writing to {csv_path}: {str(e)}")

    def get_keypoint_at_frame(self, frame_number):
        """
        Retrieve the keypoint for a specific frame.

        Parameters
        ----------
        frame_number : int
            The frame number to retrieve

        Returns
        -------
        Keypoint or None
            The keypoint at the specified frame, or None if not found
        """
        if frame_number in self.frames:
            idx = self.frames.index(frame_number)
            return self.keypoints[idx]
        return None

    def filter_by_threshold(self, threshold):
        """
        Filter keypoints by probability threshold.

        Parameters
        ----------
        threshold : float
            Minimum probability threshold

        Returns
        -------
        VideoKeypoints
            A new VideoKeypoints object with filtered keypoints
        """
        filtered = VideoKeypoints(self._video_height, self._video_width)

        for i, kp in enumerate(self.keypoints):
            if kp.prob >= threshold:
                filtered.keypoints.append(kp)
                filtered.frames.append(self.frames[i])

        return filtered


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
