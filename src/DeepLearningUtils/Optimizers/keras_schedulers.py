import keras


class CosineDecayWithWarmup(keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self,
                 initial_learning_rate: float,
                 final_learning_rate: float,
                 n_warmup_steps: int,
                 n_cosine_steps: int,
                 accumulation_steps=1
                 ):

        self.initial_learning_rate: float = initial_learning_rate
        self.final_learning_rate: float = final_learning_rate
        self.n_warmup_steps: int = n_warmup_steps
        self.n_cosine_steps: int = n_cosine_steps
        self.accumulation_steps: int = accumulation_steps

    def __call__(self, step):

        def cosine_decay():
            discrete_step = step // self.accumulation_steps

            fraction = (discrete_step - self.n_warmup_steps) / (self.n_cosine_steps - self.n_warmup_steps)
            fraction = keras.ops.cast(fraction, "float32")

            fraction = keras.ops.minimum(fraction, 1.0)
            cos_scaling = 0.5 * (1 + keras.ops.cos(3.141 * fraction))
            lr = self.final_learning_rate * cos_scaling + self.initial_learning_rate
            return keras.ops.cast(lr, float)

        def warmup():
            discrete_step = step // self.accumulation_steps
            fraction = discrete_step / self.n_warmup_steps
            fraction = keras.ops.cast(fraction, "float32")
            max_fraction = keras.ops.cast(1.0, "float32")

            fraction = keras.ops.cond(fraction < max_fraction, lambda: fraction, lambda: max_fraction)
            lr = self.initial_learning_rate + (self.final_learning_rate - self.initial_learning_rate) * fraction
            return keras.ops.cast(lr, float)

        discrete_step = step // self.accumulation_steps
        lr = keras.ops.cond(discrete_step < self.n_warmup_steps, lambda: warmup(), lambda: cosine_decay())

        return lr


class Warmup(keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self,
                 initial_learning_rate: float,
                 final_learning_rate: float,
                 n_warmup_steps: int,
                 accumulation_steps=1
                 ):
        """

        Parameters
        ----------
        initial_learning_rate : float
            Initial learning rate.
        final_learning_rate : float
            Final learning rate.
        n_warmup_steps : int
            Number of warmup steps. Learning rate will be increased
            from initial_learning_rate to final_learning_rate linearly
            during these steps.
        accumulation_steps : int, optional
            Number of steps before updating the learning rate.
            The default is 1. This is useful for gradient accumulation so
            that learning rate is not updated while gradients are being
            accumulated across batches.
        """

        self.initial_learning_rate = initial_learning_rate
        self.final_learning_rate = final_learning_rate
        self.n_warmup_steps = n_warmup_steps
        self.accumulation_steps = accumulation_steps

    def __call__(self, step):

        def warmup():
            discrete_step = step // self.accumulation_steps
            fraction = discrete_step / self.n_warmup_steps
            fraction = keras.ops.cast(fraction, "float32")
            max_fraction = keras.ops.cast(1.0, "float32")
            fraction = keras.ops.cond(fraction < max_fraction, lambda: fraction, lambda: max_fraction)
            lr = self.initial_learning_rate + (self.final_learning_rate - self.initial_learning_rate) * fraction
            return lr

        lr = warmup()

        return lr

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "final_learning_rate": self.final_learning_rate,
            "n_warmup_steps": self.n_warmup_steps,
            "accumulation_steps": self.accumulation_steps
        }
