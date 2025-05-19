import numpy as np
import keras


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
    of input images and labels.
    """
    
    def __init__(
        self,
        images: np.ndarray,
        labels: List[List[np.ndarray]],
        augmentation: Optional[iaa.Sequential] = None,
        batch_size: int = 32,
        training: bool = True,
        shuffle: bool = True,
        line_width: int = 1,
        max_point_distance: int = 5,
        use_distance_maps: bool = False,
        include_background: bool = True,
        **kwargs
    ):
        """
        Initialize the data generator.

        Parameters
        ----------
        images : np.ndarray
            Input images array of shape (n_samples, height, width, channels)
            Values should be in range [0, 255]
        labels : List[List[np.ndarray]]
            List of n_samples elements, each containing n_lines arrays of points
            Each point array should be of shape (n_points, 2) containing (x,y) coordinates
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
        **kwargs
            Additional arguments passed to keras.utils.Sequence

        Raises
        ------
        ValueError
            If input arrays have invalid shapes or value ranges
        """
        super().__init__(**kwargs)
        
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
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.training = training
        self.sample_size = images.shape[0]
        self.indexes = np.arange(self.sample_size)
        self.seq = augmentation
        self.n_lines = n_lines
        self.line_width = line_width
        self.max_point_distance = max_point_distance
        self.use_distance_maps = use_distance_maps
        self.include_background = include_background
        
        # Get image dimensions
        self._image_height = images.shape[1]
        self._image_width = images.shape[2]
        
        # Initialize
        self.on_epoch_end()
        
    def __len__(self) -> int:
        """
        Get the number of batches per epoch.
        
        Returns
        -------
        int
            Number of batches
        """
        return int(np.floor(self.sample_size / self.batch_size))
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate one batch of data.
        
        Parameters
        ----------
        index : int
            Batch index
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Batch of (images, labels)
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Generate data
        X, y = self.__data_generation(indexes)
        
        return X, y
    
    def on_epoch_end(self):
        """
        Called at the end of every epoch.
        Shuffles the data if shuffle is True.
        """
        self.indexes = np.arange(self.sample_size)
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
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
    
    def __data_generation(
        self,
        list_IDs_temp: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate one batch of data.
        
        Parameters
        ----------
        list_IDs_temp : np.ndarray
            Array of indices for this batch
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Batch of (images, labels) with proper normalization
        """
        # Get batch data
        X = self.images[list_IDs_temp]
        y = [self.labels[i] for i in list_IDs_temp]
        
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
        
        # Normalize images to [0, 1]
        X = X / 255.0
        
        return X, y 