import keras

class DiceMultiClass(keras.metrics.Metric):

    def __init__(self, num_classes, name='dice_multiclass', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes

        #Create true positive, false negative and false positive arrays (one for each class)
        self.true_positives = [self.add_weight(name=f"tp_{i}", initializer='zeros') for i in range(num_classes)]
        self.false_negatives = [self.add_weight(name=f"fn_{i}", initializer='zeros') for i in range(num_classes)]
        self.false_positives = [self.add_weight(name=f"fp_{i}", initializer='zeros') for i in range(num_classes)]
        self.smooth = 1e-6

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = keras.ops.cast(y_true, "float32")
        y_pred = keras.ops.cast(y_pred, "float32")

        for i in range(self.num_classes):
            intersection = keras.ops.sum(y_true[..., i] * y_pred[..., i])
            self.true_positives[i].assign_add(intersection)
            self.false_negatives[i].assign_add(keras.ops.sum(y_true[..., i]) - intersection)
            self.false_positives[i].assign_add(keras.ops.sum(y_pred[..., i]) - intersection)

    def result(self):

        dice_scores = [
        (2 * self.true_positives[i] + self.smooth) / (2 * self.true_positives[i] + self.false_negatives[i] + self.false_positives[i] + self.smooth)
        for i in range(self.num_classes)]

        return keras.ops.mean(dice_scores)

    def reset_state(self):

        for i in range(self.num_classes):
            self.true_positives[i].assign(0)
            self.false_negatives[i].assign(0)
            self.false_positives[i].assign(0)
