
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras

from src.DeepLearningUtils.Layers.Backbones.Efficientvit.efficientvit_keras import EfficientViT_B
from src.DeepLearningUtils.Layers.Backbones.Mask_Encoder.mask_encoder_keras import create_mask_encoder
from src.DeepLearningUtils.Layers.Backbones.Memory_Encoder.memory_encoder_keras import create_memory_model
from src.DeepLearningUtils.Layers.Attention.SequentialCoAttention.sequential_co_attention_keras import CoMemoryAttentionModule
from src.DeepLearningUtils.Layers.Attention.SequentialCoAttention.sequential_co_attention_keras import CoAttentionModule
from src.DeepLearningUtils.Layers.Decoders.decoder_keras import UNetDecoder
def create_model():
    key_dim = 128
    value_dim = 128

    embedding_dim = 64

    memory_SEQ_LEN = 5
    att_SEQ_LEN = 5

    generator_input_shape = (256, 256, 1)
    encoder_input_shape = (256, 256, 3)

    encoder_model_backbone = EfficientViT_B(
        input_shape=encoder_input_shape,
        activation="keras.activations.hard_swish",
        # activation="keras.activations.relu",
        num_blocks=[2, 2, 3, 3],
        out_channels=[16, 32, 64, 128],
        stem_width=16,
        expansions=4,
        block_types=["conv", "conv", "transform", "transform"],
        output_filters=256,
        anti_aliasing=True,
        upsample_levels=1,
    )

    encoder_backbone_outputs = []
    for layer in [
        "stem_MB_output",  # 128x128
        "stack_1_block_2_output",  # 64x64
        "stack_2_block_2_output",  # 32x32
        # "stack_3_block_3_output",  # 16x16
        "features_conv",
    ]:  # 8x8
        encoder_backbone_outputs.append(encoder_model_backbone.get_layer(layer).output)

    encoder_model = keras.models.Model(
        inputs=encoder_model_backbone.input,
        outputs=encoder_backbone_outputs,
        name="efficientvit")

    encoder_inputs_p = keras.Input(shape=encoder_input_shape, name="encoder_input")
    memory_inputs_p = keras.Input(shape=(memory_SEQ_LEN, *encoder_input_shape), name="memory_input")
    memory_labels_p = keras.Input(shape=(memory_SEQ_LEN, *generator_input_shape), name="memory_labels")
    memory_input_mask_p = keras.Input(shape=(memory_SEQ_LEN, ), name="memory_input_mask")
    encoder_outputs = encoder_model(encoder_inputs_p)

    base_memory_model = EfficientViT_B(
        input_shape=encoder_input_shape,
        activation="keras.activations.hard_swish",
        num_blocks=[2, 2, 3, 3],
        out_channels=[16, 32, 64, 128],
        stem_width=16,
        expansions=4,
        block_types=["conv", "conv", "transform", "transform"],
        output_filters=256,
        anti_aliasing=True,
        upsample_levels=1,
    )
    base_memory_model = keras.Model(base_memory_model.input, base_memory_model.get_layer("features_conv").output)

    mask_encoder_model = create_mask_encoder((256, 256, 1), 256, anti_aliasing=True)

    memory_model = create_memory_model(
        base_memory_model,
        mask_encoder_model,
        encoder_input_shape,
        memory_SEQ_LEN,
        base_memory_model_output_layer="features_conv",
        mask_encoder_output_layer="conv3",
        pooling_operation=keras.layers.Activation('linear'),
        memory_encoder_activation=keras.layers.Activation('linear'),
        combine_operation="add"
    )

    memory_encoder_out_features = memory_model([memory_inputs_p, memory_labels_p])

    memory_outputs = []
    for encoder_output_layer_id, encoder_output_layer, mask_output_layer, downsample_level in zip(
            [
                #  -5,
                #  -4,
                #  -3,
                # -2,
                -1
            ],
            [
                #   "stem_MB_output",  # 128x128
                #   "stack_1_block_2_output",  # 64x64
                #   "stack_2_block_2_output",  # 32x32
                # "stack_3_block_2_output",  # 16x16
                "features_conv",
            ],
            [
                #     "stem_MB_output",  # 128x128
                #   "stack_1_block_1_downsample_output",  # 64x64
                #   "stack_2_block_1_downsample_output",  # 32x32
                # "stack_3_block_1_downsample_output",  # 16x16
                "conv3"],
            [
                #   16,
                #  4,
                #   2,
                #   1,
                0,
            ]):

        if encoder_output_layer == "features_conv":
            encoder_attention_input = encoder_outputs[encoder_output_layer_id]
        else:
            encoder_feature_input = keras.layers.UpSampling2D(
                size=(2 * downsample_level, 2 * downsample_level),
                interpolation="bilinear")(encoder_outputs[-1])
            encoder_attention_input = keras.layers.Concatenate()(
                [encoder_feature_input, encoder_outputs[encoder_output_layer_id]])
            encoder_attention_input = keras.layers.Conv2D(encoder_outputs[encoder_output_layer_id].shape[-1], 1)(
                encoder_attention_input)

        attention_stack = 1
        encoder_attention_input = keras.ops.expand_dims(encoder_attention_input, 1)
        for j in range(attention_stack):
            memory_attention = CoMemoryAttentionModule(
                key_dim,
                value_dim,
                query_seq_len=1,
                key_seq_len=1,
                use_norm=False,
                attention_drop_rate=0.2,
                use_positional_embedding=True,
                use_linear_attention=False,
                use_qkv_embedding=False,
                name=f"memory_attention_{encoder_output_layer}_{j}")

            co_attention = CoAttentionModule(memory_attention)

            memory_output = co_attention(
                [encoder_attention_input,
                 memory_encoder_out_features,
                 memory_input_mask_p])
            encoder_attention_input = memory_output

        memory_outputs.append(memory_output)

        decoder = UNetDecoder(
            [128, 64, 32, 16],
            activation=keras.activations.hard_swish, )

        encoder_outputs[-1] = keras.ops.squeeze(memory_outputs[-1], 1)

        model_output = decoder([*encoder_outputs[::-1], encoder_inputs_p])

        memory_encoder_layer_name = f"memory_encoder_layer"

        model = keras.Model([encoder_inputs_p, memory_inputs_p, memory_labels_p, memory_input_mask_p], model_output)

        return model


