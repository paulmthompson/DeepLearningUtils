import pytest
import numpy as np
import tempfile
import os
from collections import OrderedDict
from src.DeepLearningUtils.DataStructures.Keypoints.keypoint import Keypoint, VideoKeypoints, read_keypoint_csv


class TestKeypoint:
    def test_keypoint_extracts_max_value_coordinates(self):
        # Simple case with one clear maximum
        test_image = np.zeros((10, 10))
        test_image[3, 7] = 0.8
        keypoint = Keypoint(test_image)

        assert keypoint.x == 3
        assert keypoint.y == 7
        assert keypoint.prob == 0.8

    def test_keypoint_averages_multiple_max_values(self):
        # Image with multiple identical maximum values
        test_image = np.zeros((10, 10))
        test_image[2, 4] = 0.9
        test_image[2, 6] = 0.9
        keypoint = Keypoint(test_image)

        assert keypoint.x == 2
        assert keypoint.y == 5  # Average of 4 and 6, rounded
        assert keypoint.prob == 0.9

    def test_keypoint_rejects_non_numpy_input(self):
        with pytest.raises(ValueError, match="Input image must be a numpy array"):
            Keypoint([[1, 2], [3, 4]])

    def test_keypoint_rejects_empty_array(self):
        with pytest.raises(ValueError, match="Input image cannot be empty"):
            Keypoint(np.array([]))

    def test_keypoint_rejects_non_2d_array(self):
        with pytest.raises(ValueError, match="Expected 2D array"):
            Keypoint(np.array([1, 2, 3]))

    def test_keypoint_handles_uniform_image(self):
        # Image with all identical values
        test_image = np.ones((5, 5))
        keypoint = Keypoint(test_image)

        assert keypoint.x == 2  # Average coordinates in 5x5 array
        assert keypoint.y == 2
        assert keypoint.prob == 1.0

    def test_keypoint_string_representation(self):
        test_image = np.zeros((5, 5))
        test_image[2, 3] = 0.75
        keypoint = Keypoint(test_image)

        assert repr(keypoint) == "Keypoint(x=2, y=3, prob=0.7500)"

    def test_keypoint_as_tuple_method(self):
        test_image = np.zeros((5, 5))
        test_image[2, 3] = 0.75
        keypoint = Keypoint(test_image)

        assert keypoint.as_tuple() == (2, 3)


class TestVideoKeypoints:
    def test_video_keypoints_initialization(self):
        video_kp = VideoKeypoints(480, 640)

        assert video_kp._video_height == 480
        assert video_kp._video_width == 640
        assert len(video_kp.keypoints) == 0
        assert len(video_kp.frames) == 0

    def test_add_frames_adds_keypoints(self):
        video_kp = VideoKeypoints(480, 640)

        # Create fake video labels with 3 frames
        video_labels = np.zeros((3, 10, 10))
        video_labels[0, 2, 3] = 0.7
        video_labels[1, 4, 5] = 0.8
        video_labels[2, 6, 7] = 0.9

        video_kp.add_frames(video_labels, frame_offset=5)

        assert len(video_kp.keypoints) == 3
        assert len(video_kp.frames) == 3

        # Check frame numbers
        assert video_kp.frames == [5, 6, 7]

        # Check keypoint values
        assert video_kp.keypoints[0].x == 2
        assert video_kp.keypoints[0].y == 3
        assert video_kp.keypoints[0].prob == 0.7

        assert video_kp.keypoints[1].x == 4
        assert video_kp.keypoints[1].y == 5
        assert video_kp.keypoints[1].prob == 0.8

        assert video_kp.keypoints[2].x == 6
        assert video_kp.keypoints[2].y == 7
        assert video_kp.keypoints[2].prob == 0.9

    def test_to_csv_exports_data_correctly(self):
        video_kp = VideoKeypoints(480, 640)

        # Create fake video labels with 3 frames
        video_labels = np.zeros((3, 10, 10))
        video_labels[0, 2, 3] = 0.7
        video_labels[1, 4, 5] = 0.8
        video_labels[2, 6, 7] = 0.9

        video_kp.add_frames(video_labels, frame_offset=5)

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
            csv_path = temp_file.name

        try:
            video_kp.to_csv(csv_path)

            # Read the file back
            with open(csv_path, 'r') as f:
                content = f.readlines()

            # Check header
            assert content[0].strip() == "Frame X Y Probability"

            # Check data entries
            assert content[1].strip() == "5 3.0 2.0 0.7"
            assert content[2].strip() == "6 5.0 4.0 0.8"
            assert content[3].strip() == "7 7.0 6.0 0.9"

        finally:
            # Clean up
            os.unlink(csv_path)

    def test_to_csv_respects_threshold(self):
        video_kp = VideoKeypoints(480, 640)

        # Create fake video labels with 3 frames with varying probabilities
        video_labels = np.zeros((3, 10, 10))
        video_labels[0, 2, 3] = 0.3
        video_labels[1, 4, 5] = 0.6
        video_labels[2, 6, 7] = 0.9

        video_kp.add_frames(video_labels, frame_offset=5)

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
            csv_path = temp_file.name

        try:
            # Set threshold to filter out low-confidence keypoints
            video_kp.to_csv(csv_path, threshold=0.5)

            # Read the file back
            with open(csv_path, 'r') as f:
                content = f.readlines()

            # Should only have header + 2 entries (not the 0.3 entry)
            assert len(content) == 3
            assert content[1].strip() == "6 5.0 4.0 0.6"
            assert content[2].strip() == "7 7.0 6.0 0.9"

        finally:
            # Clean up
            os.unlink(csv_path)

    def test_to_csv_applies_scaling(self):
        video_kp = VideoKeypoints(480, 640)

        # Create single keypoint
        video_labels = np.zeros((1, 10, 10))
        video_labels[0, 2, 4] = 0.8

        video_kp.add_frames(video_labels, frame_offset=10)

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
            csv_path = temp_file.name

        try:
            # Apply scaling factors
            video_kp.to_csv(csv_path, scale_height=2.0, scale_width=1.5)

            # Read the file back
            with open(csv_path, 'r') as f:
                content = f.readlines()

            # Check scaled coordinates (original: x=4, y=2)
            # After scaling: x=4*1.5=6, y=2*2.0=4
            assert content[1].strip() == "10 6.0 4.0 0.8"

        finally:
            # Clean up
            os.unlink(csv_path)


class TestReadKeypointCSV:
    def test_read_keypoint_csv_parses_correctly(self):
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as temp_file:
            temp_file.write("Frame X Y Probability\n")
            temp_file.write("1 100 200 0.9\n")
            temp_file.write("3 150 250 0.8\n")
            temp_file.write("5 120 220 0.7\n")
            csv_path = temp_file.name

        try:
            keypoint_coords = read_keypoint_csv(csv_path)

            # Check the parsed data structure
            assert isinstance(keypoint_coords, OrderedDict)
            assert len(keypoint_coords) == 3

            # Check individual entries - note y,x order in output
            assert keypoint_coords[1] == [200, 100]
            assert keypoint_coords[3] == [250, 150]
            assert keypoint_coords[5] == [220, 120]

        finally:
            # Clean up
            os.unlink(csv_path)

    def test_read_keypoint_csv_with_custom_delimiter(self):
        # Create a temporary CSV file with comma delimiter
        with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as temp_file:
            temp_file.write("Frame,X,Y,Probability\n")
            temp_file.write("1,100,200,0.9\n")
            temp_file.write("3,150,250,0.8\n")
            csv_path = temp_file.name

        try:
            keypoint_coords = read_keypoint_csv(csv_path, delimiter=",")

            # Check the parsed data structure
            assert len(keypoint_coords) == 2
            assert keypoint_coords[1] == [200, 100]
            assert keypoint_coords[3] == [250, 150]

        finally:
            # Clean up
            os.unlink(csv_path)

    def test_read_keypoint_csv_handles_float_coordinates(self):
        # Create a temporary CSV file with float coordinates
        with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as temp_file:
            temp_file.write("Frame X Y Probability\n")
            temp_file.write("1 100.6 200.4 0.9\n")
            temp_file.write("3 150.2 250.7 0.8\n")
            csv_path = temp_file.name

        try:
            keypoint_coords = read_keypoint_csv(csv_path)

            # Should round the float coordinates
            assert keypoint_coords[1] == [200, 101]
            assert keypoint_coords[3] == [251, 150]

        finally:
            # Clean up
            os.unlink(csv_path)