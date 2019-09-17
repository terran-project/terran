import torch.nn as nn


def mx_eval_at_symbol(path, layer, arr):
    import mxnet as mx
    sym, arg_params, aux_params = mx.model.load_checkpoint(path, 0)

    output = sym.get_internals()[layer]

    arr = arr.transpose([0, 3, 1, 2])

    model = mx.mod.Module(symbol=output, context=mx.cpu(), label_names=None)
    model.bind(data_shapes=[("data", arr.shape)], for_training=False)
    model.set_params(arg_params, aux_params)

    model.forward(
        mx.io.DataBatch(
            data=(mx.ndarray.array(arr),),
            provide_data=['data', arr.shape]
        ),
        is_train=False
    )

    out = model.get_outputs()[0]

    return out


def map_bn(from_, to):
    mapping = {
        'beta': 'bias',
        'gamma': 'weight',
        'moving_mean': 'running_mean',
        'moving_var': 'running_var',
    }
    return {
        f'{to}.{t}': f'{from_}_{f}'
        for f, t in mapping.items()
    }


def _map_mxnet_weights(path):
    import torch
    import mxnet as mx
    from itertools import chain

    sym, arg_params, aux_params = mx.model.load_checkpoint(path, 0)

    translations = {}

    # Initial and final layers.
    translations.update(map_bn('bn0', 'initial_layer.1'))
    translations.update(map_bn('bn1', 'final_layer.0'))
    translations.update(map_bn('fc1', 'final_layer.4'))
    translations['initial_layer.0.weight'] = 'conv0_weight'
    translations['initial_layer.2.weight'] = 'relu0_gamma'
    translations['final_layer.3.weight'] = 'pre_fc1_weight'
    translations['final_layer.3.bias'] = 'pre_fc1_bias'

    # Stages.
    units_per_stage = [3, 13, 30, 3]
    channels = [64, 64, 128, 256, 512]

    for stage_idx, num_units in enumerate(units_per_stage):
        # Number of channels for previous and current stage.
        prev_c = channels[stage_idx]
        curr_c = channels[stage_idx + 1]

        for unit_idx in range(num_units):
            translations.update(map_bn(f'stage{stage_idx+1}_unit{unit_idx+1}_bn1', f'stages.{stage_idx}.{unit_idx}.body.0'))
            translations.update(map_bn(f'stage{stage_idx+1}_unit{unit_idx+1}_bn2', f'stages.{stage_idx}.{unit_idx}.body.2'))
            translations.update(map_bn(f'stage{stage_idx+1}_unit{unit_idx+1}_bn3', f'stages.{stage_idx}.{unit_idx}.body.5'))

            translations.update({
                f'stages.{stage_idx}.{unit_idx}.body.1.weight': f'stage{stage_idx+1}_unit{unit_idx+1}_conv1_weight',
                f'stages.{stage_idx}.{unit_idx}.body.3.weight': f'stage{stage_idx+1}_unit{unit_idx+1}_relu1_gamma',
                f'stages.{stage_idx}.{unit_idx}.body.4.weight': f'stage{stage_idx+1}_unit{unit_idx+1}_conv2_weight',
            })

            dimensions_match = prev_c == curr_c and unit_idx != 0
            if not dimensions_match:
                translations.update(map_bn(f'stage{stage_idx+1}_unit{unit_idx+1}_sc', f'stages.{stage_idx}.{unit_idx}.shortcut.1'))
                translations.update({
                    f'stages.{stage_idx}.{unit_idx}.shortcut.0.weight': f'stage{stage_idx+1}_unit{unit_idx+1}_conv1sc_weight',
                })

    translations = {v: k for k, v in translations.items()}

    state_dict = {}
    for key, value in chain(arg_params.items(), aux_params.items()):
        state_dict[translations[key]] = torch.tensor(value.asnumpy())

    return state_dict


def test_model():
    import os
    import numpy as np
    import mxnet as mx
    import torch
    from terran.face.recognition.arcface import FaceResNet100, mx_eval_at_symbol, _map_mxnet_weights
    from PIL import Image

    image = Image.open(
        os.path.expanduser('~/images/asdf.jpg')
    ).resize((112, 112))
    arr = np.expand_dims(np.asarray(image), 0)

    path = os.path.expanduser('~/dev/terran/checkpoints/model-r100-ii/model')

    model = FaceResNet100()
    model.load_state_dict(_map_mxnet_weights(path))
    model = model.eval()

    pout = model(torch.tensor(
        arr.transpose([0, 3, 1, 2]).astype(np.float32)
    )).detach().numpy()

    mxout = mx_eval_at_symbol(path, 'fc1_output', arr).asnumpy()

    print(np.max(np.abs(mxout - pout)))


class Unit(nn.Module):

    def __init__(self, in_c, out_c, stride=1):
        super().__init__()

        self.dimensions_match = in_c == out_c and stride == 1

        self.body = nn.Sequential(
            nn.BatchNorm2d(in_c, momentum=0.9, eps=2e-5),

            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c, momentum=0.9, eps=2e-5),
            nn.PReLU(num_parameters=out_c),

            nn.Conv2d(out_c, out_c, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c, momentum=0.9, eps=2e-5),
        )

        # If input and output dimensions within unit don't match, apply a
        # convolutional layer to adjust the number of channels.
        if not self.dimensions_match:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c, momentum=0.9, eps=2e-5),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        body = self.body(x)
        shortcut = self.shortcut(x)
        return body + shortcut


class FaceResNet100(nn.Module):

    def __init__(self):
        super().__init__()

        # Network hyperparameters.
        self.units_per_stage = [3, 13, 30, 3]
        self.channels = [64, 64, 128, 256, 512]

        # Preprocessing parameters.
        self.mean = 127.5
        self.std = 0.0078125

        # Initial layers applied.
        self.initial_layer = nn.Sequential(
            nn.Conv2d(3, self.channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(self.channels[0], momentum=0.9, eps=2e-5),
            nn.PReLU(num_parameters=self.channels[0]),
        )

        stages = []
        for stage_idx, num_units in enumerate(self.units_per_stage):
            # Number of channels for previous and current stage.
            prev_c = self.channels[stage_idx]
            curr_c = self.channels[stage_idx + 1]
            num_units = self.units_per_stage[stage_idx]

            units = nn.Sequential(*[
                Unit(prev_c, curr_c, stride=2),
                *[
                    Unit(curr_c, curr_c)
                    for _ in range(num_units - 1)
                ]
            ])

            stages.append(units)

        self.stages = nn.ModuleList(stages)

        # Final FC size is 7x7x512: 7x7 produced by an initial size of 112x112
        # and a downsampling rate of 16 of the Resnet, and 512 channels.
        self.final_layer = nn.Sequential(
            nn.BatchNorm2d(self.channels[-1], momentum=0.9, eps=2e-5),
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Linear(7 * 7 * 512, 512),
            nn.BatchNorm1d(512, momentum=0.9, eps=2e-5),
        )

    def forward(self, x):
        preprocessed = (x - self.mean) * self.std

        out = self.initial_layer(preprocessed)

        for stage in self.stages:
            out = stage(out)

        out = self.final_layer(out)

        return out
