import torch.nn as nn


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
