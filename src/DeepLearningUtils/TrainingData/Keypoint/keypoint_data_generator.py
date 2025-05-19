import numpy as np
import tensorflow as tf
from typing import List, Tuple, Optional, Union, Any
import imgaug.augmenters as iaa


# Help design from 
#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
#https://www.kaggle.com/code/hics33/example-using-imgaug-keras

class KeypointDataGenerator(tf.keras.utils.Sequence):
    """
    Data generator for keypoint detection training.
    
    This generator handles both binary masks and heatmap-style keypoint labels.
    It supports data augmentation using the imgaug library and proper normalization
    of input images and labels.
    """
    
    def __init__(self, 
                 images: np.ndarray,
                 labels: np.ndarray,
                 augmentation: Optional[iaa.Sequential] = None,
                 batch_size: int = 32,
                 training: bool = True,
                 shuffle: bool = True,
                 **kwargs):
        """
        Initialize the data generator.

        Parameters
        ----------
        images : np.ndarray
            Input images array of shape (n_samples, height, width, channels)
            Values should be in range [0, 255]
        labels : np.ndarray
            Keypoint labels array of shape (n_samples, height, width, n_keypoints)
            For binary masks: values should be in range [0, 1]
            For heatmaps: values should be in range [0, 255]
        augmentation : Optional[iaa.Sequential]
            imgaug augmentation sequence to apply during training
        batch_size : int
            Number of samples per batch
        training : bool
            Whether the generator is used for training (applies augmentation)
        shuffle : bool
            Whether to shuffle the data at the end of each epoch
        **kwargs
            Additional arguments passed to tf.keras.utils.Sequence

        Raises
        ------
        ValueError
            If input arrays have invalid shapes or value ranges
        """
        super().__init__(**kwargs)
        
        # Validate input shapes
        if len(images.shape) != 4:
            raise ValueError(f"Images must be 4D array (n_samples, height, width, channels), got shape {images.shape}")
        if len(labels.shape) != 4:
            raise ValueError(f"Labels must be 4D array (n_samples, height, width, n_keypoints), got shape {labels.shape}")
        if images.shape[0] != labels.shape[0]:
            raise ValueError(f"Number of images ({images.shape[0]}) must match number of labels ({labels.shape[0]})")
        if images.shape[1:3] != labels.shape[1:3]:
            raise ValueError(f"Image dimensions {images.shape[1:3]} must match label dimensions {labels.shape[1:3]}")
            
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
        self.n_keypoints = labels.shape[3]
        
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
    
    def __data_generation(self, list_IDs_temp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        y = self.labels[list_IDs_temp]
        
        # Apply augmentation if in training mode
        if self.training and self.seq is not None:
            X, y = self.seq(images=X, segmentation_maps=y)
        
        # Convert to float32
        X = X.astype('float32')
        y = y.astype('float32')

        # Normalize heatmap labels to [0, 1]
        # Binary masks should already be in [0, 1]
        for k in range(self.n_keypoints):
            if np.max(y[:,:,:,k]) > 1.0:  # If it's a heatmap
                y[:,:,:,k] = y[:,:,:,k] / 255.0
        
        return X, y 