import keras


class MemoryEncoderLayer(keras.layers.Layer):
    def __init__(self,
                 base_memory_model: keras.Model,
                 mask_encoder_model: keras.Model,
                 height: int,
                 width: int,
                 channels: int,
                 base_memory_model_output_layer=None,
                 mask_encoder_output_layer=None,
                 combine_operation='dense',
                 activation=keras.layers.Activation('tanh'),
                 **kwargs):
        """

        This layer is going to to apply the memory encoder
        and mask encoder to a sequence of images.
        The output of the memory encoder is then combined
        with the mask encoder output.

        This can happen at different levels of the model if the
        layer names for the mask and base memory layers are specified.

        Parameters
        ----------
        base_memory_model : Model
            Base model for memory encoder
        mask_encoder_model : Model
            Mask encoder model
        height : int
            Height of the input image
        width : int
            Width of the input image
        channels : int
            Number of channels in the input image
        base_memory_model_output_layer : str
            Name of the output layer of the base memory model
        mask_encoder_output_layer : str
            Name of the output layer of the mask encoder model
        combine_operation : str


        """

        super(MemoryEncoderLayer, self).__init__(**kwargs)

        self.base_memory_model = base_memory_model
        self.mask_encoder_model = mask_encoder_model
        self.height = height
        self.width = width
        self.channels = channels
        self.combine_operation = combine_operation
        self.activation = activation

        # Create the memory model
        self.memory_model = MemoryModelBase(
            self.base_memory_model,
            input_shape=(self.height, self.width, self.channels),
            output_layer_name=base_memory_model_output_layer,
        )
        self.memory_model.build((None, self.height, self.width, self.channels))

        self.mask_encoder = MemoryModelBase(
            self.mask_encoder_model,
            input_shape=(self.height, self.width, 1),
            output_layer_name=mask_encoder_output_layer,
        )
        self.mask_encoder.build((None, self.height, self.width, 1))

    def build(self, input_shape):
        memory_input_shape, memory_label_shape = input_shape

        self.memory_model_distributed = keras.layers.TimeDistributed(
            self.memory_model
        )
        memory_encoder_out = self.memory_model_distributed.build(memory_input_shape)

        self.mask_encoder_distributed = keras.layers.TimeDistributed(
            self.mask_encoder
        )
        encoded_masks = self.mask_encoder_distributed.build(memory_label_shape)

        if self.combine_operation == 'add':
            self.combine_layer = keras.layers.Add()
        elif self.combine_operation == 'dense':

            self.combine_layer = keras.Sequential([
                keras.layers.Concatenate(axis=-1),
                keras.layers.Dense(
                    self.memory_model.model.output_shape[-1]
                )
            ])
            # self.combine_layer.build([memory_encoder_out, encoded_masks])

            self.combine_layer.layers[1].build(
                (None,
                 self.memory_model.model.output_shape[-3],
                 self.memory_model.model.output_shape[-2],
                 self.memory_model.model.output_shape[-1] * 2)
            )

        elif self.combine_operation == 'conv':
            self.combine_layer = keras.Sequential([
                keras.layers.Concatenate(axis=-1),
                keras.layers.TimeDistributed(
                    keras.layers.Conv2D(
                        128,
                        (5, 5),
                        padding='same'
                    )
                ),
                keras.layers.TimeDistributed(
                    keras.layers.Conv2D(
                        self.memory_model.model.output_shape[-1],
                        (1, 1),
                        padding='same'
                    )
                )
            ])
            self.combine_layer.build([self.memory_model.model.output_shape, self.mask_encoder.model.output_shape])
            """
            self.combine_layer.layers[1].build(
                (None, self.memory_model.model.output_shape[-3],
                 self.memory_model.model.output_shape[-2],
                 self.memory_model.model.output_shape[-1] * 2)
                 ) 
                 """
        else:
            raise ValueError(f"Combine operation {self.combine_operation} not supported")

    def call(self, inputs):
        memory_inputs, memory_labels = inputs

        # Apply the memory model
        memory_encoder_out = self.memory_model_distributed(memory_inputs)

        # Apply the mask encoder
        encoded_masks = self.mask_encoder_distributed(memory_labels)

        # Combine the outputs
        memory_encoder_out = self.combine_layer([memory_encoder_out, encoded_masks])

        memory_encoder_out = self.activation(memory_encoder_out)

        return memory_encoder_out

    def compute_output_shape(self, input_shape):
        memory_inputs_shape, memory_labels_shape = input_shape
        base_model_output = self.memory_model.model.output_shape
        output_shape = (memory_inputs_shape[0],
                        memory_inputs_shape[1],
                        base_model_output[1],
                        base_model_output[2],
                        base_model_output[3])
        return output_shape


class MemoryModelBase(keras.layers.Layer):
    """
    Our memory model will be applied to a sequence of images.
    In order to use the TimeDistributed layer for the sequence,
    we need to wrap the encoder in this custom layer.


    """

    def __init__(self,
                 base_model,
                 input_shape,
                 output_layer_name=None,
                 **kwargs):
        super().__init__(**kwargs)

        if output_layer_name is not None:
            outputs = base_model.get_layer(output_layer_name).output
        else:
            outputs = base_model.output

        self.model = keras.Model(base_model.input, outputs=outputs)

    def call(self, inputs):
        return self.model(inputs)

    def compute_output_shape(self, input_shape):
        # Assuming the input shape is (batch_size, sequence_length, H, W, C)
        return (input_shape[0], self.model.output_shape[1], self.model.output_shape[2], self.model.output_shape[3])