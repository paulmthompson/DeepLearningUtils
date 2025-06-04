
import torch
import torch.nn as nn

import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np

from src.DeepLearningUtils.Layers.LayerNorm2D.layernorm2d_pytorch import LayerNorm2d


def get_keras_weights_by_name(keras_layer):

    keras_weights = keras_layer.weights
    weights_by_layer = {}

    for weight in keras_weights:
        path = weight.path

        # get name is between last / and /
        name = path.split('/')[-2]
        print(path)
        if name not in weights_by_layer:
            weights_by_layer[name] = [weight.value.numpy()]
        else:
            weights_by_layer[name].append(weight.value.numpy())

    return weights_by_layer


def load_conv2d_weights(keras_layer, pytorch_module):
    if isinstance(pytorch_module, nn.Conv2d):
        print("Loading Conv2d weights for", keras_layer.name)
        weight = keras_layer.get_weights()[0]
        if pytorch_module.groups > 1:
            # Depthwise convolution
            keras_weights = torch.tensor(weight).permute(3, 2, 0, 1).contiguous()
            pytorch_module.weight.data = torch.reshape(keras_weights, pytorch_module.weight.shape)
        else:
            # Standard convolution
            pytorch_module.weight.data = torch.tensor(weight).permute(3, 2, 0, 1).contiguous()

        if pytorch_module.bias is not None:
            bias = keras_layer.get_weights()[1]
            pytorch_module.bias.data = torch.tensor(bias)
        else:
            print(f"Skipping bias for {keras_layer.name}")


def load_batchnorm_weights(keras_layer, pytorch_module):
    if isinstance(pytorch_module, nn.BatchNorm2d):
        print("Loading BatchNorm2d weights for", keras_layer.name)
        if pytorch_module.training:
            print("WARNING: BatchNorm2d layer is in training mode")
        weights = keras_layer.get_weights()
        if (pytorch_module.weight.data.shape != torch.tensor(weights[0]).shape):
            print(f"WARNING: Weight shapes do not match: {pytorch_module.weight.data.shape} != {torch.tensor(weights[0].shape)}")
        pytorch_module.weight.data = torch.tensor(weights[0])

        if (pytorch_module.bias.data.shape != torch.tensor(weights[1]).shape):
            print(f"WARNING: Bias shapes do not match: {pytorch_module.bias.data.shape} != {torch.tensor(weights[1].shape)}")
        pytorch_module.bias.data = torch.tensor(weights[1])

        if (pytorch_module.running_mean.shape != torch.tensor(weights[2]).shape):
            print(f"WARNING: Running mean shapes do not match: {pytorch_module.running_mean.shape} != {torch.tensor(weights[2].shape)}")
        pytorch_module.running_mean = torch.tensor(weights[2])

        if (pytorch_module.running_var.shape != torch.tensor(weights[3]).shape):
            print(f"WARNING: Running var shapes do not match: {pytorch_module.running_var.shape} != {torch.tensor(weights[3].shape)}")
        pytorch_module.running_var = torch.tensor(weights[3])

    else:
        print(f"Skipping non-BatchNorm2d layer {keras_layer.name}")

def load_layer_norm_weights(keras_layer_norm, pytorch_layer_norm):
    if isinstance(pytorch_layer_norm, nn.LayerNorm):
        pytorch_layer_norm.weight.data = torch.tensor(keras_layer_norm.get_weights()[0])
        pytorch_layer_norm.bias.data = torch.tensor(keras_layer_norm.get_weights()[1])
    else:
        print(f"Skipping non-LayerNorm layer {keras_layer_norm.name}")


def load_layernorm2d_weights(keras_layer, pytorch_module):
    if isinstance(pytorch_module, LayerNorm2d):
        print("Loading LayerNorm2d weights for", keras_layer.name)
        pytorch_module.weight.data = torch.tensor(keras_layer.weight.numpy())
        pytorch_module.bias.data = torch.tensor(keras_layer.bias.numpy())
    else:
        print(f"Skipping non-LayerNorm2d layer {keras_layer.name}")

def load_linear_weights(keras_layer, pytorch_module):
    if isinstance(pytorch_module, nn.Linear):
        print("Loading Linear weights for", keras_layer.name)
        weights = keras_layer.get_weights()
        pytorch_module.weight.data = torch.tensor(weights[0]).t().contiguous()
        if pytorch_module.bias is not None:
            pytorch_module.bias.data = torch.tensor(weights[1])
        else:
            print(f"Skipping bias for {keras_layer.name}")
    else:
        print(f"Skipping non-Linear layer {keras_layer.name}")


def load_keras_layer_weights(layer, pytorch_model, custom_loaders=None):

    keras_layer_name = layer.name

    if custom_loaders is None:
        custom_loaders = {}

    print("Checking for", keras_layer_name)

    for name, module in pytorch_model.named_modules():

        # trim name to what is after last dot
        name = name.split(".")[-1]

        if name == keras_layer_name:
            print(f"Loading {name}")

            if name in custom_loaders:
                custom_loaders[name](layer, module)
            elif isinstance(module, nn.Conv2d):
                load_conv2d_weights(layer, module)
            elif isinstance(module, nn.BatchNorm2d):
                load_batchnorm_weights(layer, module)
            elif isinstance(module, nn.Linear):
                load_linear_weights(layer, module)
            elif isinstance(module, LayerNorm2d):
                load_layernorm2d_weights(layer, module)
            else:
                print(f"Skipping {name}")
            break

def load_keras_weights_to_pytorch_by_name(keras_model, pytorch_model, custom_loaders=None):

    if custom_loaders is None:
        custom_loaders = {}

    for layer in keras_model.layers:
        load_keras_layer_weights(layer, pytorch_model, custom_loaders)


def load_keras_into_pytorch(
        keras_layer,
        pytorch_module,
        pytorch_name_formatting_fn=None,
):
    weights_by_layer = get_keras_weights_by_name(keras_layer)

    for keras_name, weights in weights_by_layer.items():

        print("Checking for", keras_name)

        for pytorch_name, module in pytorch_module.named_modules():

            # For the pytorch name, replace last . with _
            # pytorch_name = pytorch_name.replace(".", "_")
            if pytorch_name_formatting_fn is not None:
                pytorch_name = pytorch_name_formatting_fn(pytorch_name)
            if keras_name == pytorch_name:
                name = keras_name
                print(f"Loading {name}")

                if isinstance(module, nn.Conv2d):
                    weight = weights[0]
                    if module.groups == module.in_channels and module.groups > 1:
                        # Depthwise convolution
                        module.weight.data = torch.tensor(weight).permute(2, 3, 0, 1).contiguous()
                    else:
                        # Standard convolution
                        module.weight.data = torch.tensor(weight).permute(3, 2, 0, 1).contiguous()

                    if module.bias is not None:
                        bias = weights[1]
                        module.bias.data = torch.tensor(bias)
                    else:
                        print(f"Skipping bias for {name}")

                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data = torch.tensor(weights[0])
                    module.bias.data = torch.tensor(weights[1])
                    module.running_mean = torch.tensor(weights[2])
                    module.running_var = torch.tensor(np.sqrt(weights[3]))

                elif isinstance(module, nn.Linear):

                    module.weight.data = torch.tensor(weights[0]).t().contiguous()
                    if module.bias is not None:
                        bias = weights[1]
                        module.bias.data = torch.tensor(bias)
                    else:
                        print(f"Skipping bias for {name}")

                elif isinstance(module, nn.LayerNorm):
                    module.weight.data = torch.tensor(weights[0])
                    module.bias.data = torch.tensor(weights[1])

                else:
                    print(f"Skipping {name}")
                    # print(module)