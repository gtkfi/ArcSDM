from collections import OrderedDict
from typing import Optional, Sequence

import torch
import torch.nn as nn

from arcsdm.mlp.classification.types import HiddenLayerSpec


class MLPClassifierModel(nn.Module):
    def __init__(self, input_dims: int, hidden_layers: Sequence[HiddenLayerSpec], last_layer: HiddenLayerSpec) -> None:
        super(MLPClassifierModel, self).__init__()

        all_layers = list(hidden_layers) + [last_layer]

        layers = []
        layers.append(("lin_input", nn.Linear(in_features=input_dims, out_features=all_layers[0][0])))
        for i in range(len(all_layers) - 1):
            neurons, activation_func, dropout_rate = tuple(all_layers[i])
            next_layer_neurons, _, _ = tuple(all_layers[i + 1])

            if (activation_func is not None) and (self.get_activation_function(activation_func) is not None):
                layers.append((f"a_{i}", self.get_activation_function(activation_func)))
            if (dropout_rate is not None) and (dropout_rate != 0):
                layers.append((f"do_{i}", nn.Dropout(dropout_rate)))

            layers.append((f"l_{i}", nn.Linear(in_features=neurons, out_features=next_layer_neurons)))

        idx = len(all_layers)
        _, activation_func, dropout_rate = tuple(all_layers[-1])
        if (activation_func is not None) and (self.get_activation_function(activation_func) is not None):
            layers.append((f"a_{idx}", self.get_activation_function(activation_func)))
        if (dropout_rate is not None) and (dropout_rate != 0):
            layers.append((f"do_{idx}", nn.Dropout(dropout_rate)))

        self.layers = nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def get_activation_function(self, name: str) -> Optional[nn.Module]:
        name = name.lower().strip()
        if name == "relu":
            return nn.ReLU()
        elif name == "tanh":
            return nn.Tanh()
        elif name == "sigmoid":
            return nn.Sigmoid()
        elif name == "softmax":
            return nn.Softmax(dim=1)
        else:
            return None
