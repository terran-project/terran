import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvSepBlock(nn.Module):

    def __init__(self, in_c, out_c, stride=1, return_both=False):
        """Building block for base network.

        Consists of common Conv, BN and ReLU sequence, followed by the same
        sequence but with a separable Conv.

        Paramters
        ---------
        return_both : bool
            Return the outputs of both inner components, the conv and the
            separable blocks. We do this because it's the conv block the one
            that's used as feature pyramid.

        """
        super().__init__()

        self.return_both = return_both

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_c, momentum=0.9, eps=1e-5),
            nn.ReLU(),
        )

        self.sep_block = nn.Sequential(
            nn.Conv2d(
                out_c, out_c, 3, stride=stride, padding=1, groups=out_c,
                bias=False
            ),
            nn.BatchNorm2d(out_c, momentum=0.9, eps=1e-5),
            nn.ReLU(),
        )

    def forward(self, x):
        conv = self.conv_block(x)
        sep = self.sep_block(conv)

        if self.return_both:
            out = conv, sep
        else:
            out = sep

        return out


class BaseNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        # The first convolutional block is different from the rest: the large
        # stride and kernel size is on the non-separable convolution instead.
        self.first_conv_block = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8, momentum=0.9, eps=1e-5),
            nn.ReLU(),

            nn.Conv2d(8, 8, 3, stride=1, padding=1, groups=8, bias=False),
            nn.BatchNorm2d(8, momentum=0.9, eps=1e-5),
            nn.ReLU(),
        )

        # We're going to extract intermediate feature maps at downsampling
        # ratios of 8 and 16, so we group the outputs in a way that we can
        # retrieve them easily enough on the `forward` pass.
        self.scales = nn.ModuleList([
            nn.Sequential(
                ConvSepBlock(8, 16, stride=2),

                ConvSepBlock(16, 32),
                ConvSepBlock(32, 32, stride=2),

                ConvSepBlock(32, 64),
                ConvSepBlock(64, 64, stride=2, return_both=True),
            ),

            nn.Sequential(
                ConvSepBlock(64, 128),
                ConvSepBlock(128, 128),
                ConvSepBlock(128, 128),
                ConvSepBlock(128, 128),
                ConvSepBlock(128, 128),
                ConvSepBlock(128, 128, stride=2, return_both=True),
            ),
        ])

        self.final_conv = nn.Sequential(
            ConvSepBlock(128, 256),
            nn.Conv2d(256, 256, 1, bias=False),
            nn.BatchNorm2d(256, momentum=0.9, eps=1e-5),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.first_conv_block(x)

        feature_maps = []
        for scale in self.scales:
            conv, out = scale(out)
            feature_maps.append(conv)

        out = self.final_conv(out)
        feature_maps.append(out)

        return feature_maps


class ContextModule(nn.Module):
    """Context module to expand the receptive field of the feature map.

    Every point in the feature map will be a mixture of a 3x3, a 5x5 and a 7x7
    receptive fields. The first 128 channels correspond to the 3x3 one, and the
    remaining 64 and 64 to the rest.
    """

    def __init__(self):
        super().__init__()

        self.context_3x3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32, momentum=0.9, eps=2e-5),
            nn.ReLU(),
        )

        self.dimension_reducer = nn.Sequential(
            nn.Conv2d(64, 16, 3, padding=1),
            nn.BatchNorm2d(16, momentum=0.9, eps=2e-5),
            nn.ReLU(),
        )

        self.context_5x5 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16, momentum=0.9, eps=2e-5),
            nn.ReLU(),
        )

        self.context_7x7 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16, momentum=0.9, eps=2e-5),
            nn.ReLU(),

            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16, momentum=0.9, eps=2e-5),
            nn.ReLU(),
        )

    def forward(self, x):
        red = self.dimension_reducer(x)

        # By applying successive 3x3 convolutions, we get an effective
        # receptive field of 5x5 and 7x7.
        ctx_3x3 = self.context_3x3(x)
        ctx_5x5 = self.context_5x5(red)
        ctx_7x7 = self.context_7x7(red)

        out = torch.cat([ctx_3x3, ctx_5x5, ctx_7x7], dim=1)

        return out


class PyramidRefiner(nn.Module):
    """Refines the feature pyramids from the base network into usable form.

    Normalizes channel sizes, mixes them up, and runs them through the context
    module.
    """

    def __init__(self):
        super().__init__()

        self.conv_stride8 = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64, momentum=0.9, eps=2e-5),
            nn.ReLU(),
        )
        self.conv_stride16 = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64, momentum=0.9, eps=2e-5),
            nn.ReLU(),
        )
        self.conv_stride32 = nn.Sequential(
            nn.Conv2d(256, 64, 1),
            nn.BatchNorm2d(64, momentum=0.9, eps=2e-5),
            nn.ReLU(),
        )

        self.aggr_stride8 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64, momentum=0.9, eps=2e-5),
            nn.ReLU(),
        )
        self.aggr_stride16 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64, momentum=0.9, eps=2e-5),
            nn.ReLU(),
        )

        self.context_stride8 = ContextModule()
        self.context_stride16 = ContextModule()
        self.context_stride32 = ContextModule()

    def forward(self, x):
        """Forward pass for refiner.

        Expects `x` to be an array of three tensors, one per feature. Returns
        the same.
        """
        stride8, stride16, stride32 = x

        # First run every scale through the conv layers.
        proc_stride8 = self.conv_stride8(stride8)
        proc_stride16 = self.conv_stride16(stride16)
        proc_stride32 = self.conv_stride32(stride32)

        # Upsample the smaller to the bigger ones and sum. When upsampling,
        # resulting tensor might be bigger by one spatial dimension (if initial
        # dimension wasn't power of 2), so we slice the tensor to the same
        # size.
        ups_stride32 = F.interpolate(
            proc_stride32, scale_factor=2
        )[:, :, :proc_stride16.shape[2], :proc_stride16.shape[3]]
        proc_stride16 = self.aggr_stride16(
            proc_stride16 + ups_stride32
        )

        ups_stride16 = F.interpolate(
            proc_stride16, scale_factor=2
        )[:, :, :proc_stride8.shape[2], :proc_stride8.shape[3]]
        proc_stride8 = self.aggr_stride8(
            proc_stride8 + ups_stride16
        )

        # Now run every scale through a context module.
        ctx_stride8 = self.context_stride8(proc_stride8)
        ctx_stride16 = self.context_stride16(proc_stride16)
        ctx_stride32 = self.context_stride32(proc_stride32)

        return [ctx_stride8, ctx_stride16, ctx_stride32]


class OutputsPredictor(nn.Module):
    """Uses the feature pyramid to predict the final deltas for the network."""

    def __init__(self):
        super().__init__()

        # There are two anchors per point: two scales and one ratio.
        self.num_anchors = 2

        # Layers for anchor class, bbox coords and landmark coords.
        self.cls_stride8 = nn.Conv2d(64, 2 * self.num_anchors, 1)
        self.cls_stride16 = nn.Conv2d(64, 2 * self.num_anchors, 1)
        self.cls_stride32 = nn.Conv2d(64, 2 * self.num_anchors, 1)

        self.bbox_stride8 = nn.Conv2d(64, 4 * self.num_anchors, 1)
        self.bbox_stride16 = nn.Conv2d(64, 4 * self.num_anchors, 1)
        self.bbox_stride32 = nn.Conv2d(64, 4 * self.num_anchors, 1)

        self.landmark_stride8 = nn.Conv2d(64, 10 * self.num_anchors, 1)
        self.landmark_stride16 = nn.Conv2d(64, 10 * self.num_anchors, 1)
        self.landmark_stride32 = nn.Conv2d(64, 10 * self.num_anchors, 1)

    def forward(self, x):
        """Forward pass for output predictor.

        Expects `x` to hold one feature map per stride.
        """
        stride8, stride16, stride32 = x

        cls_score8 = self.cls_stride8(stride8)
        cls_score16 = self.cls_stride16(stride16)
        cls_score32 = self.cls_stride32(stride32)

        # Reshape the class scores so we can easily calculate the softmax along
        # the two entries per anchor.
        N, A, H, W = cls_score8.shape
        cls_prob8 = F.softmax(
            cls_score8.view(N, 2, -1, W), dim=1
        ).view(N, A, H, W)
        N, A, H, W = cls_score16.shape
        cls_prob16 = F.softmax(
            cls_score16.view(N, 2, -1, W), dim=1
        ).view(N, A, H, W)
        N, A, H, W = cls_score32.shape
        cls_prob32 = F.softmax(
            cls_score32.view(N, 2, -1, W), dim=1
        ).view(N, A, H, W)

        bbox_pred8 = self.bbox_stride8(stride8)
        bbox_pred16 = self.bbox_stride16(stride16)
        bbox_pred32 = self.bbox_stride32(stride32)

        landmark_pred8 = self.landmark_stride8(stride8)
        landmark_pred16 = self.landmark_stride16(stride16)
        landmark_pred32 = self.landmark_stride32(stride32)

        return [
            cls_prob32,
            bbox_pred32,
            landmark_pred32,

            cls_prob16,
            bbox_pred16,
            landmark_pred16,

            cls_prob8,
            bbox_pred8,
            landmark_pred8,
        ]


class RetinaFace(nn.Module):
    """RetinaFace model with a pseudo-MobileNet backbone.

    Consists of three inner modules: a base network to get a feature pyramid
    from, a pyramid refiner, that adds context and mixes the results a bit, and
    an output predictor that returns the final predictions for class, bounding
    box and landmarks per anchor.

    """

    def __init__(self):
        super().__init__()

        self.base = BaseNetwork()
        self.refiner = PyramidRefiner()
        self.outputs = OutputsPredictor()

    def forward(self, x):
        out = self.base(x)
        out = self.refiner(out)
        out = self.outputs(out)

        return out
