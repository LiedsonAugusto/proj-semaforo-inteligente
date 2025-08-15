import torch
import torch.nn as nn
import torch.nn.functional as F

class SemaforoInteligente(nn.Module):
    def __init__(self, input_size, output_size=40, next_size_ratio=.8, dropout=.1, first_layer_mulpl=3):
        super(SemaforoInteligente, self).__init__()
        camadas = []

        # print(f"{input_size} -> {input_size*first_layer_mulpl}")
        camadas.append(nn.Linear(input_size, int(input_size*first_layer_mulpl)))
        camadas.append(nn.ReLU())
        # camadas.append(nn.Dropout(dropout))

        actual_size = int(input_size*first_layer_mulpl)
        next_size = int(actual_size*next_size_ratio)
        if next_size == actual_size:
            next_size -= 10
        while next_size > output_size:
            # print(f"{actual_size} -> {next_size}")
            camadas.append(nn.Linear(actual_size, next_size))
            camadas.append(nn.ReLU())
            # camadas.append(nn.Dropout(dropout))
            actual_size = next_size
            next_size = int(actual_size*next_size_ratio)
            if next_size == actual_size:
                next_size -= 10

        # print(f"{actual_size} -> {output_size}")
        camadas.append(nn.Linear(actual_size, output_size))
        self.camadas = nn.Sequential(*camadas)

    def forward(self, x):
        return self.camadas(x)


import torch
import torch.nn as nn
import torch.nn.functional as F

class SemaforoInteligente2(nn.Module):
    def __init__(self, input_size, output_size=40, next_size_ratio=.8, dropout=.1, first_layer_mulpl=3):
        super(SemaforoInteligente2, self).__init__()
        camadas = []

        conv_out_channels = 32
        kernel_size = 4
        stride = 2
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=conv_out_channels,
                               kernel_size=kernel_size, stride=stride)

        conv_seq_len = (input_size - kernel_size) // stride + 1
        conv_flat_size = conv_out_channels * conv_seq_len

        camadas.append(nn.Linear(conv_flat_size, int(conv_flat_size * first_layer_mulpl)))
        camadas.append(nn.ReLU())

        actual_size = int(conv_flat_size * first_layer_mulpl)
        next_size = int(actual_size * next_size_ratio)
        if next_size == actual_size:
            next_size -= 10
        while next_size > output_size:
            camadas.append(nn.Linear(actual_size, next_size))
            camadas.append(nn.ReLU())
            actual_size = next_size
            next_size = int(actual_size * next_size_ratio)
            if next_size == actual_size:
                next_size -= 10

        camadas.append(nn.Linear(actual_size, output_size))
        self.camadas = nn.Sequential(*camadas)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = x.flatten(1)
        return self.camadas(x)
