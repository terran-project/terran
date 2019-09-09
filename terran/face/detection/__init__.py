import mxnet as mx
import torch
import torch.nn as nn

from itertools import chain


def _load_from_mxnet(path):
    sym, arg_params, aux_params = mx.model.load_checkpoint(path, 0)

    bn_translation = {
        'beta': 'bias',
        'gamma': 'weight',
        'running_mean': 'running_mean',
        'moving_mean': 'running_mean',
        'running_var': 'running_var',
        'moving_var': 'running_var',
    }

    state_dict = {}
    for key, value in chain(arg_params.items(), aux_params.items()):
        if not key.startswith('mobilenet0'):
            continue

        value = torch.Tensor(value.asnumpy())

        _, block, weight = key.split('_', 2)

        block_type = 'conv' if 'conv' in block else 'batchnorm'
        block_num = int(block[len(block_type):])

        # Get the equivalent block number: each conv-sep block is numbered as
        # a single one, and we need to have special consideration for the first
        # and final.
        eq_num = int(block_num / 2) - 1
        eq_type = 'sep' if round(block_num % 2) == 1 else 'conv'

        if block_type == 'conv':
            seq_num = 0
            weight_type = weight
        elif block_type == 'batchnorm':
            seq_num = 1
            weight_type = bn_translation[weight]

        if block_num in [0, 1]:
            # 0, 1 for first conv & bn, 3, 4 for second conv & bn.
            eq_num = seq_num + 3 * block_num
            new_key = (
                f'first_conv_block.{eq_num}.{weight_type}'
            )
        elif block_num == 26:
            new_key = (
                f'final_conv.{seq_num}.{weight_type}'
            )
        else:
            new_key = (
                f'blocks.{eq_num}.{eq_type}_block.{seq_num}.{weight_type}'
            )

        # max_len = max([len(p) for p in chain(arg_params, aux_params)])
        # print(f'{key:>{max_len}} >> {new_key}')
        state_dict[new_key] = value

    return state_dict


class ConvBlock(nn.Module):

    def __init__(self, in_c, out_c, stride=1):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_c, momentum=0.9),
            nn.ReLU(),
        )

        self.sep_block = nn.Sequential(
            nn.Conv2d(
                out_c, out_c, 3, stride=stride, padding=1, groups=out_c,
                bias=False
            ),
            nn.BatchNorm2d(out_c, momentum=0.9),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.conv_block(x)
        out = self.sep_block(out)
        return out


class BaseNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        # The first convolutional block is different from the rest: the large
        # stride and kernel size is on the non-separable convolution instead.
        self.first_conv_block = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8, momentum=0.9),
            nn.ReLU(),

            nn.Conv2d(8, 8, 3, stride=1, padding=1, groups=8, bias=False),
            nn.BatchNorm2d(8, momentum=0.9),
            nn.ReLU(),
        )

        self.blocks = nn.Sequential(
            ConvBlock(8, 16, stride=2),

            ConvBlock(16, 32),
            ConvBlock(32, 32, stride=2),

            ConvBlock(32, 64),
            ConvBlock(64, 64, stride=2),

            ConvBlock(64, 128),
            ConvBlock(128, 128),
            ConvBlock(128, 128),
            ConvBlock(128, 128),
            ConvBlock(128, 128),
            ConvBlock(128, 128, stride=2),

            ConvBlock(128, 256),
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(256, 256, 1, bias=False),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.first_conv_block(x)
        out = self.blocks(out)
        out = self.final_conv(out)
        return out
