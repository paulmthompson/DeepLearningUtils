import numpy as np
import pandas as pd
import keras
import cv2

from typing import List, Tuple, Optional, Union, Any
import imgaug.augmenters as iaa
from imgaug.augmentables.lines import LineString, LineStringsOnImage

from src.DeepLearningUtils.utils.data_generation import (
    interpolate_points,
    create_line_mask,
    create_distance_map,
    validate_line_data
)


class LineDataGenerator(keras.utils.Sequence):
    """
    Data generator for line/whisker detection training.
    
    This generator handles both binary masks and distance maps for line detection.
    It supports data augmentation using the imgaug library and proper normalization
    of input images and labels. Can work with either numpy arrays (legacy) or
    pandas DataFrames (new interface).
    """
    
    def __init__(
        self,
        images: Optional[np.ndarray] = None,
        labels: Optional[List[List[np.ndarray]]] = None,
        dataframe: Optional[pd.DataFrame] = None,
        label_order: Optional[List[str]] = None,
        augmentation: Optional[iaa.Sequential] = None,
        batch_size: int = 32,
        training: bool = True,
        shuffle: bool = True,
        line_width: int = 1,
        max_point_distance: int = 5,
        use_distance_maps: bool = False,
        include_background: bool = False,
        compress_labels: bool = False,
        target_resolution: Optional[Tuple[int, int]] = None,
        **kwargs
    ):
        """
        Initialize the data generator.

        Parameters
        ----------
        images : Optional[np.ndarray]
            Legacy input: Input images array of shape (n_samples, height, width, channels)
            Values should be in range [0, 255]
        labels : Optional[List[List[np.ndarray]]]
            Legacy input: List of n_samples elements, each containing n_lines arrays of points
            Each point array should be of shape (n_points, 2) containing (x,y) coordinates
        dataframe : Optional[pd.DataFrame]
            New input: DataFrame with columns ['folder_id', 'image_name', 'image', 'labels']
            where 'labels' contains dict mapping label names to coordinate arrays
        label_order : Optional[List[str]]
            Order of labels to use for output channels. Required when using dataframe.
        augmentation : Optional[iaa.Sequential]
            imgaug augmentation sequence to apply during training
        batch_size : int
            Number of samples per batch
        training : bool
            Whether the generator is used for training (applies augmentation)
        shuffle : bool
            Whether to shuffle the data at the end of each epoch
        line_width : int
            Width of the lines in pixels
        max_point_distance : int
            Maximum allowed distance between consecutive points before interpolation
        use_distance_maps : bool
            Whether to return distance maps instead of binary masks
        include_background : bool
            Whether to include a background channel in the output masks
        compress_labels : bool
            If True, combine all labels into a single channel (any label present)
        target_resolution : Optional[Tuple[int, int]]
            Target resolution (height, width) to resize images to. If None, use original size.
        **kwargs
            Additional arguments passed to keras.utils.Sequence

        Raises
        ------
        ValueError
            If input parameters are invalid or incompatible
        """
        super().__init__(**kwargs)
        
        # Store parameters first (before calling init methods)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.training = training
        self.seq = augmentation
        self.line_width = line_width
        self.max_point_distance = max_point_distance
        self.use_distance_maps = use_distance_maps
        self.include_background = include_background
        self.compress_labels = compress_labels
        self.target_resolution = target_resolution

        # Determine input mode
        if dataframe is not None:
            if images is not None or labels is not None:
                raise ValueError("Cannot specify both dataframe and legacy (images/labels) inputs")
            self._init_from_dataframe(dataframe, label_order)
        elif images is not None and labels is not None:
            if label_order is not None:
                raise ValueError("label_order only supported with dataframe input")
            self._init_from_arrays(images, labels)
        else:
            raise ValueError("Must specify either dataframe OR (images and labels)")

        # Initialize
        self.on_epoch_end()

    def _init_from_dataframe(self, dataframe: pd.DataFrame, label_order: Optional[List[str]]):
        """Initialize from pandas DataFrame input."""
        if dataframe.empty:
            raise ValueError("DataFrame is empty")

        required_columns = ['folder_id', 'image_name', 'image', 'labels']
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {missing_columns}")

        if label_order is None:
            raise ValueError("label_order must be specified when using dataframe input")

        self.dataframe = dataframe.copy()
        self.label_order = label_order
        self.sample_size = len(dataframe)
        self.indexes = np.arange(self.sample_size)
        self.use_dataframe = True

        # Validate that all specified labels exist in at least one sample
        all_labels = set()
        for _, row in dataframe.iterrows():
            all_labels.update(row['labels'].keys())

        missing_labels = [label for label in label_order if label not in all_labels]
        if missing_labels:
            raise ValueError(f"Labels not found in data: {missing_labels}")

        # Get image dimensions from first image
        first_image = dataframe.iloc[0]['image']
        if self.target_resolution is not None:
            self._image_height, self._image_width = self.target_resolution
        else:
            self._image_height, self._image_width = first_image.shape[:2]

    def _init_from_arrays(self, images: np.ndarray, labels: List[List[np.ndarray]]):
        """Initialize from legacy numpy array input."""
        # Validate input data
        if len(labels) == 0:
            raise ValueError("Labels list is empty")
        n_lines = len(labels[0])
        validate_line_data(images, labels, n_lines)
            
        # Validate value ranges
        if np.min(images) < 0 or np.max(images) > 255:
            raise ValueError("Image values must be in range [0, 255]")
            
        # Store data
        self.images = images
        self.labels = labels
        self.sample_size = images.shape[0]
        self.indexes = np.arange(self.sample_size)
        self.n_lines = n_lines
        self.use_dataframe = False

        # Get image dimensions
        if self.target_resolution is not None:
            self._image_height, self._image_width = self.target_resolution
        else:
            self._image_height, self._image_width = images.shape[1:3]

    def __len__(self) -> int:
        """Get the number of batches per epoch."""
        return int(np.floor(self.sample_size / self.batch_size))
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate one batch of data."""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Generate data
        X, y = self.__data_generation(indexes)
        
        return X, y
    
    def on_epoch_end(self):
        """Called at the end of every epoch. Shuffles the data if shuffle is True."""
        self.indexes = np.arange(self.sample_size)
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target resolution if specified."""
        if self.target_resolution is not None:
            return cv2.resize(image, (self.target_resolution[1], self.target_resolution[0]))
        return image

    def _resize_coordinates(self, coords: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        """Resize coordinates to match target resolution."""
        if self.target_resolution is None:
            return coords

        orig_h, orig_w = original_shape
        target_h, target_w = self.target_resolution

        # Scale coordinates
        coords_resized = coords.copy()
        coords_resized[:, 0] *= (target_w / orig_w)  # x coordinates
        coords_resized[:, 1] *= (target_h / orig_h)  # y coordinates

        return coords_resized

    def create_linestrings_from_lines(
        self,
        single_image_lines: List[np.ndarray]
    ) -> LineStringsOnImage:
        """
        Create LineStringsOnImage object from line points.
        
        Parameters
        ----------
        single_image_lines : List[np.ndarray]
            List of line point arrays
            
        Returns
        -------
        LineStringsOnImage
            Object containing all lines for the image
        """
        lsoi = LineStringsOnImage([], shape=(self._image_height, self._image_width))
        
        for line_points in single_image_lines:
            # Interpolate points if needed
            interpolated_points = interpolate_points(line_points, self.max_point_distance)
            lsoi.line_strings.append(LineString(interpolated_points))
            
        return lsoi
    
    def augment_images(
        self,
        X: np.ndarray,
        y: List[List[np.ndarray]]
    ) -> Tuple[np.ndarray, List[LineStringsOnImage]]:
        """
        Apply augmentation to images and lines.
        
        Parameters
        ----------
        X : np.ndarray
            Batch of images
        y : List[List[np.ndarray]]
            Batch of line points
            
        Returns
        -------
        Tuple[np.ndarray, List[LineStringsOnImage]]
            Augmented images and lines
        """
        images_aug = []
        lsois_aug = []
        
        for i in range(X.shape[0]):
            lsoi = self.create_linestrings_from_lines(y[i])
            
            image_aug, lsoi_aug = self.seq(image=X[i], line_strings=lsoi)
            
            # Clip lines that go outside the image
            with np.errstate(invalid="ignore"):
                lsoi_aug.clip_out_of_image_()
            
            # Skip augmentation if number of lines changed
            if len(lsoi.line_strings) != len(lsoi_aug.line_strings):
                images_aug.append(X[i])
                lsois_aug.append(lsoi)
            else:
                images_aug.append(image_aug)
                lsois_aug.append(lsoi_aug)
                
        return np.array(images_aug), lsois_aug
    
    def create_masks(
        self,
        lines: List[LineStringsOnImage]
    ) -> np.ndarray:
        """
        Create binary masks or distance maps from lines.
        
        Parameters
        ----------
        lines : List[LineStringsOnImage]
            List of LineStringsOnImage objects
            
        Returns
        -------
        np.ndarray
            Array of masks of shape (batch_size, height, width, n_channels)
        """
        masks = []
        
        for lsoi in lines:
            # Create mask for each line
            line_masks = []
            for line in lsoi.line_strings:
                points = np.array(line.coords)
                if self.use_distance_maps:
                    mask = create_line_mask(points, self._image_height, self._image_width, self.line_width)
                    mask = create_distance_map(mask)
                else:
                    mask = create_line_mask(points, self._image_height, self._image_width, self.line_width)
                line_masks.append(mask)
            
            # Stack line masks
            mask = np.stack(line_masks, axis=-1)
            
            if self.include_background:
                background_mask = np.expand_dims(
                    np.logical_not(np.any(mask, axis=-1)),
                    axis=-1
                )
                mask = np.concatenate((background_mask, mask), axis=-1)
            
            masks.append(mask)
            
        return np.stack(masks, axis=0)
    
    def create_masks_from_dataframe(self, batch_data: List[dict]) -> np.ndarray:
        """
        Create binary masks or distance maps from DataFrame batch data.

        Parameters
        ----------
        batch_data : List[dict]
            List of dictionaries containing image and label data

        Returns
        -------
        np.ndarray
            Array of masks of shape (batch_size, height, width, n_channels)
        """
        masks = []

        for data in batch_data:
            labels_dict = data['labels']
            original_shape = data['image'].shape[:2]

            # Create mask for each label category in specified order
            category_masks = []

            for label_name in self.label_order:
                if label_name in labels_dict:
                    # Get coordinates and resize if needed
                    coords = labels_dict[label_name]
                    coords_resized = self._resize_coordinates(coords, original_shape)

                    # Create mask
                    if self.use_distance_maps:
                        mask = create_line_mask(coords_resized, self._image_height, self._image_width, self.line_width)
                        mask = create_distance_map(mask)
                    else:
                        mask = create_line_mask(coords_resized, self._image_height, self._image_width, self.line_width)
                else:
                    # Create blank mask for missing labels
                    mask = np.zeros((self._image_height, self._image_width), dtype=np.float32)

                category_masks.append(mask)

            if self.compress_labels:
                # Combine all labels into single channel
                combined_mask = np.any(np.stack(category_masks, axis=-1), axis=-1).astype(np.float32)
                mask = np.expand_dims(combined_mask, axis=-1)
            else:
                # Stack individual category masks
                mask = np.stack(category_masks, axis=-1)

            if self.include_background:
                background_mask = np.expand_dims(
                    np.logical_not(np.any(mask > 0, axis=-1)).astype(np.float32),
                    axis=-1
                )
                mask = np.concatenate((background_mask, mask), axis=-1)

            masks.append(mask)

        return np.stack(masks, axis=0)

    def __data_generation(
        self,
        list_IDs_temp: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate one batch of data."""
        if self.use_dataframe:
            return self._data_generation_dataframe(list_IDs_temp)
        else:
            return self._data_generation_legacy(list_IDs_temp)

    def _data_generation_dataframe(self, list_IDs_temp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate batch from DataFrame input."""
        # Get batch data
        batch_data = []
        batch_images = []

        for idx in list_IDs_temp:
            row = self.dataframe.iloc[idx]
            image = self._resize_image(row['image'])
            batch_images.append(image)
            batch_data.append({
                'image': row['image'],  # Keep original for coordinate scaling
                'labels': row['labels']
            })

        X = np.stack(batch_images, axis=0)

        # Create masks
        y = self.create_masks_from_dataframe(batch_data)

        # Convert to float32 and normalize
        X = X.astype('float32')
        y = y.astype('float32')

        return X, y

    def _data_generation_legacy(self, list_IDs_temp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate batch from legacy numpy array input."""
        # Get batch data
        X = self.images[list_IDs_temp]
        y = [self.labels[i] for i in list_IDs_temp]
        
        # Resize images if needed
        if self.target_resolution is not None:
            X_resized = []
            for img in X:
                X_resized.append(self._resize_image(img))
            X = np.stack(X_resized, axis=0)

        # Apply augmentation if in training mode
        if self.training and self.seq is not None:
            X, y = self.augment_images(X, y)
        else:
            # Convert lines to LineStringsOnImage objects
            y = [self.create_linestrings_from_lines(lines) for lines in y]
        
        # Create masks
        y = self.create_masks(y)
        
        # Convert to float32
        X = X.astype('float32')
        y = y.astype('float32')
        
        return X, y
