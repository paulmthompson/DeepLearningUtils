import keras

class FocalLossMultiClass(keras.metrics.Metric):

    def __init__(self, num_classes, gamma=2.0, alpha=0.25, name='focal_loss_multiclass', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.gamma = gamma  # Focusing parameter
        self.alpha = alpha  # Balancing parameter

        # Sum of focal loss values and count for averaging
        self.total_focal_loss = self.add_weight(name="total_focal_loss", initializer='zeros')
        self.count = self.add_weight(name="count", initializer='zeros')
        self.smooth = 1e-6

    def update_state(self, y_true, y_pred, sample_weight=None):

        # Calculate focal loss for all classes
        batch_size = keras.ops.shape(y_true)[0]

        y_true = keras.ops.cast(y_true, "float32")
        y_pred = keras.ops.cast(y_pred, "float32")

        # Calculate pt (probability of true class)
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)

        # Calculate alpha factor
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)

        # Calculate modulating factor
        modulating_factor = keras.ops.power(1.0 - pt, self.gamma)

        # Calculate focal loss per sample and class
        focal_loss = -alpha_factor * modulating_factor * keras.ops.log(pt + self.smooth)

        # Sum over all classes
        focal_loss = keras.ops.sum(focal_loss, axis=-1)

        # Update total focal loss
        total_loss = keras.ops.sum(focal_loss)
        self.total_focal_loss.assign_add(total_loss)
        self.count.assign_add(keras.ops.cast(batch_size, dtype=self.count.dtype))

    def result(self):
        return self.total_focal_loss / (self.count + self.smooth)

    def reset_state(self):
        self.total_focal_loss.assign(0)
        self.count.assign(0)