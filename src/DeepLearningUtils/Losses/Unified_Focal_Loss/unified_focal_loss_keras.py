"""
   Copyright 2022 Michael Yeung

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

    Modified 2025 by Paul Thompson
"""

import keras

# Helper function to enable loss function to be flexibly used for
# both 2D or 3D image segmentation - source: https://github.com/frankkramer-lab/MIScnn
def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5 : return [1, 2, 3]
    # Two dimensional
    elif len(shape) == 4 : return [1, 2]
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')

################################
#     Asymmetric Focal loss    #
################################
def asymmetric_focal_loss(delta=0.7, gamma=2.):
    """
    For Imbalanced datasets

    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
    """

    def loss_function(y_true, y_pred):

        axis = identify_axis(y_true.get_shape())

        epsilon = keras.config.epsilon()
        y_pred = keras.ops.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * keras.ops.log(y_pred)

        #calculate losses separately for each class, only suppressing background class
        back_ce = keras.ops.power(1 - y_pred[:, :, :, 0], gamma) * cross_entropy[:, :, :, 0]
        back_ce = (1 - delta) * back_ce

        fore_ce = cross_entropy[:, :, :, 1]
        fore_ce = delta * fore_ce

        loss = keras.ops.mean(keras.ops.sum(keras.ops.stack([back_ce, fore_ce], axis=-1), axis=-1))

        return loss

    return loss_function


#################################
# Asymmetric Focal Tversky loss #
#################################
def asymmetric_focal_tversky_loss(delta=0.7, gamma=0.75):
    """
    This is the implementation for binary segmentation.

    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    """
    def loss_function(y_true, y_pred):
        # Clip values to prevent division by zero error
        epsilon = keras.config.epsilon()
        y_pred = keras.ops.clip(y_pred, epsilon, 1. - epsilon)

        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = keras.ops.sum(y_true * y_pred, axis=axis)
        fn = keras.ops.sum(y_true * (1-y_pred), axis=axis)
        fp = keras.ops.sum((1-y_true) * y_pred, axis=axis)
        dice_class = (tp + epsilon)/(tp + delta*fn + (1-delta)*fp + epsilon)

        #calculate losses separately for each class, only enhancing foreground class
        back_dice = (1-dice_class[:, 0])
        fore_dice = (1-dice_class[:, 1]) * keras.ops.power(1-dice_class[:, 1], -gamma)

        # Average class scores
        loss = keras.ops.mean(keras.ops.stack([back_dice, fore_dice], axis=-1))
        return loss

    return loss_function


###########################################
#      Asymmetric Unified Focal loss      #
###########################################
def asym_unified_focal_loss(weight=0.5, delta=0.6, gamma=0.5):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to asymmetric Focal Tversky loss and asymmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    """
    def loss_function(y_true, y_pred):

        y_true = keras.ops.cast(y_true, "float32")
        y_pred = keras.ops.cast(y_pred, "float32")

        # Create background and prepend to last dimension
        
        y_true = keras.ops.concatenate([1 - y_true, y_true], axis=-1)
        y_pred = keras.ops.concatenate([1 - y_pred, y_pred], axis=-1)

        asymmetric_ftl = asymmetric_focal_tversky_loss(delta=delta, gamma=gamma)(y_true, y_pred)
        asymmetric_fl = asymmetric_focal_loss(delta=delta, gamma=gamma)(y_true, y_pred)
        if weight is not None:
            return (weight * asymmetric_ftl) + ((1-weight) * asymmetric_fl)
        else:
            return asymmetric_ftl + asymmetric_fl

    return loss_function


#################################
# Symmetric Focal Tversky loss  #
#################################
def symmetric_focal_tversky_loss(delta=0.7, gamma=0.75):
    """This is the implementation for binary segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    """
    def loss_function(y_true, y_pred):
        # Clip values to prevent division by zero error
        epsilon = keras.config.epsilon()
        y_pred = keras.ops.clip(y_pred, epsilon, 1. - epsilon)

        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = keras.ops.sum(y_true * y_pred, axis=axis)
        fn = keras.ops.sum(y_true * (1-y_pred), axis=axis)
        fp = keras.ops.sum((1-y_true) * y_pred, axis=axis)
        dice_class = (tp + epsilon)/(tp + delta*fn + (1-delta)*fp + epsilon)

        #calculate losses separately for each class, enhancing both classes
        back_dice = (1-dice_class[:, 0]) * keras.ops.power(1-dice_class[:, 0], -gamma)
        fore_dice = (1-dice_class[:, 1]) * keras.ops.power(1-dice_class[:, 1], -gamma)

        # Average class scores
        loss = keras.ops.mean(keras.ops.stack([back_dice, fore_dice], axis=-1))
        return loss

    return loss_function


################################
#       Symmetric Focal loss      #
################################
def symmetric_focal_loss(delta=0.7, gamma=2.):
    """
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
    """
    def loss_function(y_true, y_pred):

        axis = identify_axis(y_true.get_shape())

        epsilon = keras.config.epsilon()
        y_pred = keras.ops.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * keras.ops.log(y_pred)
        #calculate losses separately for each class
        back_ce = keras.ops.power(1 - y_pred[:, :, :, 0], gamma) * cross_entropy[:, :, :, 0]
        back_ce = (1 - delta) * back_ce

        fore_ce = keras.ops.power(1 - y_pred[:, :, :, 1], gamma) * cross_entropy[:, :, :, 1]
        fore_ce = delta * fore_ce

        loss = keras.ops.mean(keras.ops.sum(keras.ops.stack([back_ce, fore_ce], axis=-1), axis=-1))

        return loss

    return loss_function

###########################################
#      Symmetric Unified Focal loss       #
###########################################
def sym_unified_focal_loss(weight=0.5, delta=0.6, gamma=0.5):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to symmetric Focal Tversky loss and symmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    """
    def loss_function(y_true, y_pred):

        y_true = keras.ops.cast(y_true, "float32")
        y_pred = keras.ops.cast(y_pred, "float32")

        # Create background and foreground
        y_true = keras.ops.concatenate([1-y_true, y_true], axis=-1)
        y_pred = keras.ops.concatenate([1-y_pred, y_pred], axis=-1)

        symmetric_ftl = symmetric_focal_tversky_loss(delta=delta, gamma=gamma)(y_true, y_pred)
        symmetric_fl = symmetric_focal_loss(delta=delta, gamma=gamma)(y_true, y_pred)
        if weight is not None:
            return (weight * symmetric_ftl) + ((1-weight) * symmetric_fl)
        else:
            return symmetric_ftl + symmetric_fl
