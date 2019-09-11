import numpy as np
import mxnet as mx
import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import chain
from PIL import Image


BASE_MAPPING = {
    'rf_c1_aggr_bias': 'aggr_stride8.0.bias',
    'rf_c1_aggr_weight': 'aggr_stride8.0.weight',
    'rf_c1_aggr_bn_beta': 'aggr_stride8.1.bias',
    'rf_c1_aggr_bn_gamma': 'aggr_stride8.1.weight',
    'rf_c1_aggr_bn_moving_mean': 'aggr_stride8.1.running_mean',
    'rf_c1_aggr_bn_moving_var': 'aggr_stride8.1.running_var',

    'rf_c1_red_conv_bias': 'conv_stride8.0.bias',
    'rf_c1_red_conv_weight': 'conv_stride8.0.weight',
    'rf_c1_red_conv_bn_beta': 'conv_stride8.1.bias',
    'rf_c1_red_conv_bn_gamma': 'conv_stride8.1.weight',
    'rf_c1_red_conv_bn_moving_mean': 'conv_stride8.1.running_mean',
    'rf_c1_red_conv_bn_moving_var': 'conv_stride8.1.running_var',

    'rf_c2_aggr_bias': 'aggr_stride16.0.bias',
    'rf_c2_aggr_weight': 'aggr_stride16.0.weight',
    'rf_c2_aggr_bn_beta': 'aggr_stride16.1.bias',
    'rf_c2_aggr_bn_gamma': 'aggr_stride16.1.weight',
    'rf_c2_aggr_bn_moving_mean': 'aggr_stride16.1.running_mean',
    'rf_c2_aggr_bn_moving_var': 'aggr_stride16.1.running_var',

    'rf_c2_lateral_bias': 'conv_stride16.0.bias',
    'rf_c2_lateral_weight': 'conv_stride16.0.weight',
    'rf_c2_lateral_bn_beta': 'conv_stride16.1.bias',
    'rf_c2_lateral_bn_gamma': 'conv_stride16.1.weight',
    'rf_c2_lateral_bn_moving_mean': 'conv_stride16.1.running_mean',
    'rf_c2_lateral_bn_moving_var': 'conv_stride16.1.running_var',

    'rf_c3_lateral_bias': 'conv_stride32.0.bias',
    'rf_c3_lateral_weight': 'conv_stride32.0.weight',
    'rf_c3_lateral_bn_beta': 'conv_stride32.1.bias',
    'rf_c3_lateral_bn_gamma': 'conv_stride32.1.weight',
    'rf_c3_lateral_bn_moving_mean': 'conv_stride32.1.running_mean',
    'rf_c3_lateral_bn_moving_var': 'conv_stride32.1.running_var',
}

CONTEXT_MAPPING = {
    'conv1_bias': 'context_3x3.0.bias',
    'conv1_weight': 'context_3x3.0.weight',
    'conv1_bn_beta': 'context_3x3.1.bias',
    'conv1_bn_gamma': 'context_3x3.1.weight',
    'conv1_bn_moving_mean': 'context_3x3.1.running_mean',
    'conv1_bn_moving_var': 'context_3x3.1.running_var',

    'context_conv1_bias': 'dimension_reducer.0.bias',
    'context_conv1_weight': 'dimension_reducer.0.weight',
    'context_conv1_bn_beta': 'dimension_reducer.1.bias',
    'context_conv1_bn_gamma': 'dimension_reducer.1.weight',
    'context_conv1_bn_moving_mean': 'dimension_reducer.1.running_mean',
    'context_conv1_bn_moving_var': 'dimension_reducer.1.running_var',

    'context_conv2_bias': 'context_5x5.0.bias',
    'context_conv2_weight': 'context_5x5.0.weight',
    'context_conv2_bn_beta': 'context_5x5.1.bias',
    'context_conv2_bn_gamma': 'context_5x5.1.weight',
    'context_conv2_bn_moving_mean': 'context_5x5.1.running_mean',
    'context_conv2_bn_moving_var': 'context_5x5.1.running_var',

    'context_conv3_1_bias': 'context_7x7.0.bias',
    'context_conv3_1_weight': 'context_7x7.0.weight',
    'context_conv3_1_bn_beta': 'context_7x7.1.bias',
    'context_conv3_1_bn_gamma': 'context_7x7.1.weight',
    'context_conv3_1_bn_moving_mean': 'context_7x7.1.running_mean',
    'context_conv3_1_bn_moving_var': 'context_7x7.1.running_var',

    'context_conv3_2_bias': 'context_7x7.3.bias',
    'context_conv3_2_weight': 'context_7x7.3.weight',
    'context_conv3_2_bn_beta': 'context_7x7.4.bias',
    'context_conv3_2_bn_gamma': 'context_7x7.4.weight',
    'context_conv3_2_bn_moving_mean': 'context_7x7.4.running_mean',
    'context_conv3_2_bn_moving_var': 'context_7x7.4.running_var',
}

OUTPUT_MAPPING = {
    'face_rpn_cls_score_stride8_bias': 'cls_stride8.bias',
    'face_rpn_cls_score_stride8_weight': 'cls_stride8.weight',
    'face_rpn_cls_score_stride16_bias': 'cls_stride16.bias',
    'face_rpn_cls_score_stride16_weight': 'cls_stride16.weight',
    'face_rpn_cls_score_stride32_bias': 'cls_stride32.bias',
    'face_rpn_cls_score_stride32_weight': 'cls_stride32.weight',

    'face_rpn_bbox_pred_stride8_bias': 'bbox_stride8.bias',
    'face_rpn_bbox_pred_stride8_weight': 'bbox_stride8.weight',
    'face_rpn_bbox_pred_stride16_bias': 'bbox_stride16.bias',
    'face_rpn_bbox_pred_stride16_weight': 'bbox_stride16.weight',
    'face_rpn_bbox_pred_stride32_bias': 'bbox_stride32.bias',
    'face_rpn_bbox_pred_stride32_weight': 'bbox_stride32.weight',

    'face_rpn_landmark_pred_stride8_bias': 'landmark_stride8.bias',
    'face_rpn_landmark_pred_stride8_weight': 'landmark_stride8.weight',
    'face_rpn_landmark_pred_stride16_bias': 'landmark_stride16.bias',
    'face_rpn_landmark_pred_stride16_weight': 'landmark_stride16.weight',
    'face_rpn_landmark_pred_stride32_bias': 'landmark_stride32.bias',
    'face_rpn_landmark_pred_stride32_weight': 'landmark_stride32.weight',
}


def load_model(path):
    model = RetinaFace().eval()

    state_dict = {}
    state_dict.update({
        f'base.{k}': v for k, v in _load_from_mxnet(path).items()
    })
    state_dict.update({
        f'refiner.{k}': v for k, v in _load_pr_from_mxnet(path).items()
    })
    state_dict.update({
        f'outputs.{k}': v for k, v in _load_op_from_mxnet(path).items()
    })

    model.load_state_dict(state_dict)

    return model


def run_tests():
    from terran.face.detection import (
        _load_from_mxnet, _load_pr_from_mxnet, BaseNetwork, mx_eval_at_symbol,
        similar_enough, load_model, PyramidRefiner, OutputsPredictor,
        _load_op_from_mxnet,
    )

    arr = np.expand_dims(np.asarray(
        Image.open('/home/agustin/dev/faceotron/image.png')
    ), 0)

    base = BaseNetwork()
    base.load_state_dict(
        _load_from_mxnet(
            '/home/agustin/dev/insightface/RetinaFace/model/mnet.25'
        )
    )
    base = base.eval()

    ref = PyramidRefiner()
    ref.load_state_dict(
        _load_pr_from_mxnet(
            '/home/agustin/dev/insightface/RetinaFace/model/mnet.25'
        )
    )
    ref = ref.eval()

    model = OutputsPredictor()
    model.load_state_dict(
        _load_op_from_mxnet(
            '/home/agustin/dev/insightface/RetinaFace/model/mnet.25'
        )
    )

    base_out = base(torch.Tensor(arr.transpose([0, 3, 1, 2])))
    ref_out = ref(base_out)
    pyt_out = model(ref_out)


def similar_enough(pyt_arr1, mx_arr2, thr=1e-5):
    arr1 = pyt_arr1.detach().numpy()
    arr2 = mx_arr2.asnumpy()

    # Percentage of differences below threshold.
    return len(
        np.flatnonzero(np.abs(arr1 - arr2) < thr)
    ) / np.prod(arr1.shape), np.linalg.norm(arr1 - arr2)


def mx_eval_at_symbol(path, layer, arr):
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

    max_len = max([len(p) for p in chain(arg_params, aux_params)])
    translations = []

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
        elif block_num in [24, 25]:
            new_key = (
                f'final_conv.0.{eq_type}_block.{seq_num}.{weight_type}'
            )
        elif block_num == 26:
            seq_num += 1
            new_key = (
                f'final_conv.{seq_num}.{weight_type}'
            )
        else:
            first_num = min(int(eq_num / 5), 1)
            second_num = eq_num if first_num == 0 else eq_num - 5
            new_key = (
                f'scales.{first_num}.{second_num}.{eq_type}_block.{seq_num}'
                f'.{weight_type}'
            )

        translations.append(f'{key:>{max_len}} >> {new_key}')
        state_dict[new_key] = value

    # for line in sorted(translations):
    #     print(line)

    return state_dict


def _load_pr_from_mxnet(path):
    sym, arg_params, aux_params = mx.model.load_checkpoint(path, 0)

    state_dict = {}
    for key, value in chain(arg_params.items(), aux_params.items()):
        if not key.startswith('rf_'):
            continue

        value = torch.Tensor(value.asnumpy())

        if '_det_' in key:
            base, rest = key[:5], key[10:]

            new_base = {
                'rf_c1': 'context_stride8',
                'rf_c2': 'context_stride16',
                'rf_c3': 'context_stride32',
            }[base]
            new_rest = CONTEXT_MAPPING[rest]

            new_key = '.'.join([new_base, new_rest])

        else:
            new_key = BASE_MAPPING[key]

        state_dict[new_key] = value

    return state_dict


def _load_op_from_mxnet(path):
    sym, arg_params, aux_params = mx.model.load_checkpoint(path, 0)

    state_dict = {}
    for key, value in chain(arg_params.items(), aux_params.items()):
        if not key.startswith('face_rpn_'):
            continue

        value = torch.Tensor(value.asnumpy())
        state_dict[OUTPUT_MAPPING[key]] = value

    return state_dict


class ConvSepBlock(nn.Module):

    def __init__(self, in_c, out_c, stride=1, return_both=False):
        """Building block for base network.

        Consists of common Conv, BN and ReLU sequence, followed by the same
        sequence but with a separable Conv.

        Arguments:
            return_both (bool): Return the outputs of both inner components,
                the conv and the separable blocks. We do this because it's the
                conv block the one that's used as feature pyramid.

        """
        super().__init__()

        self.return_both = return_both

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
            nn.BatchNorm2d(8, momentum=0.9),
            nn.ReLU(),

            nn.Conv2d(8, 8, 3, stride=1, padding=1, groups=8, bias=False),
            nn.BatchNorm2d(8, momentum=0.9),
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
            nn.BatchNorm2d(256, momentum=0.9),
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
            nn.BatchNorm2d(32, momentum=0.9),
            nn.ReLU(),
        )

        self.dimension_reducer = nn.Sequential(
            nn.Conv2d(64, 16, 3, padding=1),
            nn.BatchNorm2d(16, momentum=0.9),
            nn.ReLU(),
        )

        self.context_5x5 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16, momentum=0.9),
            nn.ReLU(),
        )

        self.context_7x7 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16, momentum=0.9),
            nn.ReLU(),

            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16, momentum=0.9),
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
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU(),
        )
        self.conv_stride16 = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU(),
        )
        self.conv_stride32 = nn.Sequential(
            nn.Conv2d(256, 64, 1),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU(),
        )

        self.aggr_stride8 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU(),
        )
        self.aggr_stride16 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64, momentum=0.9),
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
            cls_prob8,
            bbox_pred8,
            landmark_pred8,

            cls_prob16,
            bbox_pred16,
            landmark_pred16,

            cls_prob32,
            bbox_pred32,
            landmark_pred32,
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
