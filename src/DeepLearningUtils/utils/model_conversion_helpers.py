
import torch
import torch.nn as nn

import os
os.environ["KERAS_BACKEND"] = "torch"

import keras


def get_keras_weights_by_name(keras_layer):

    keras_weights = keras_layer.weights
    weights_by_layer = {}

    for weight in keras_weights:
        path = weight.path

        # get name is between last / and /
        name = path.split('/')[-2]
        print(path)
        if name not in weights_by_layer:
            weights_by_layer[name] = [weight.value.cpu().detach().numpy()]
        else:
            weights_by_layer[name].append(weight.value.cpu().detach().numpy())

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
        weights = keras_layer.get_weights()
        pytorch_module.weight.data = torch.tensor(weights[0])
        pytorch_module.bias.data = torch.tensor(weights[1])
        pytorch_module.running_mean = torch.tensor(weights[2])
        pytorch_module.running_var = torch.tensor(weights[3])
    else:
        print(f"Skipping non-BatchNorm2d layer {keras_layer.name}")


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


def load_keras_weights_to_pytorch_by_name(keras_model, pytorch_model):

    for layer in keras_model.layers:
        keras_layer_name = layer.name

        print("Checking for", keras_layer_name)

        for name, module in pytorch_model.named_modules():

            #trim name to what is after last dot
            name = name.split(".")[-    1]

            if name == keras_layer_name:
                print(f"Loading {name}")

                if isinstance(module, nn.Conv2d):
                    load_conv2d_weights(layer, module)
                elif isinstance(module, nn.BatchNorm2d):
                        load_batchnorm_weights(layer, module)
                elif isinstance(module, nn.Linear):
                    load_linear_weights(layer, module)
                else:
                    print(f"Skipping {name}")
                break
