import mxnet as mx
import numpy as np
import os

from mxnet import gluon
from mxnet.gluon import nn


__all__ = ['face_detection']


# Turn off cuDNN autotune from here.
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
FACE_DETECTION_PATH = os.environ.get('FACE_DETECTION_PATH')
CTX = mx.gpu() if os.environ.get('MXNET_USE_GPU') else mx.cpu()
MODEL = None


def _get_face_detection_model():
    global MODEL
    if MODEL is None:
        if FACE_DETECTION_PATH is None:
            raise ValueError(
                '`FACE_DETECTION_PATH` environment variable not set. Point it '
                'to the face detection checkpoint location.'
            )
        MODEL = mobilefacedetnet_v2(
            os.path.expanduser(FACE_DETECTION_PATH)
        )
        MODEL.collect_params().reset_ctx(ctx=CTX)
    return MODEL


def _conv2d(channel, kernel, padding, stride):
    """A common conv-bn-leakyrelu cell."""
    cell = nn.HybridSequential(prefix='')
    cell.add(
        nn.Conv2D(
            channel,
            kernel_size=kernel,
            strides=stride,
            padding=padding,
            use_bias=False
        )
    )
    cell.add(
        nn.BatchNorm(epsilon=1e-5, momentum=0.9)
    )
    cell.add(
        nn.LeakyReLU(0.1)
    )
    return cell


def _upsample(x, stride=2):
    """Simple upsampling layer by stack pixel alongside horizontal and vertical
    directions.

    Parameters
    ----------
    x : mxnet.nd.NDArray or mxnet.symbol.Symbol
        The input array.
    stride : int, default is 2
        Upsampling stride
    """
    return x.repeat(
        axis=-1, repeats=stride
    ).repeat(
        axis=-2, repeats=stride
    )


class MFDetBasicBlockV1(gluon.HybridBlock):
    """Mobilefacedet basic block, a 1x1 reduce conv followed by a 3x3 conv.

    Parameters
    ----------
    channel : int
        Convolution channels for the 1x1 convolution.

    """
    def __init__(self, channel):
        super(MFDetBasicBlockV1, self).__init__()

        self.body = nn.HybridSequential(prefix='')
        # 1x1 reduce.
        self.body.add(_conv2d(channel, 1, 0, 1))
        # 3x3 conv expand.
        self.body.add(_conv2d(channel * 2, 3, 1, 1))

    def hybrid_forward(self, F, x):
        residual = x
        x = self.body(x)
        return x + residual


class MFDetV1(gluon.HybridBlock):
    """Mobilefacedet v1 backbone."""

    def __init__(self):
        super(MFDetV1, self).__init__()
        layers = [1, 2, 2, 2, 2]
        channels = [16, 32, 64, 128, 256, 256]

        with self.name_scope():
            self.features = nn.HybridSequential()

            # First 3x3 conv.
            self.features.add(_conv2d(channels[0], 3, 1, 1))

            for nlayer, channel in zip(layers, channels[1:]):
                # Add downsample conv with stride = 2.
                self.features.add(_conv2d(channel, 3, 1, 2))

                # Add nlayer basic blocks.
                for _ in range(nlayer):
                    self.features.add(
                        MFDetBasicBlockV1(channel // 2)
                    )

    def hybrid_forward(self, F, x):
        return self.features(x)


class YOLOOutputV3(gluon.HybridBlock):
    """YOLO output layer V3.

    Has a single class, as it's specific for face detection.

    Parameters
    ----------
    index : int
        Index of the yolo output layer, to avoid naming conflicts only.
    anchors : iterable
        The anchor setting. Reference: https://arxiv.org/pdf/1804.02767.pdf.
    stride : int
        Stride of feature map.

    """

    def __init__(self, index, anchors, stride):
        super(YOLOOutputV3, self).__init__()

        anchors = np.array(anchors).astype('float32')
        self.classes = 1  # TODO: Remove.
        self._num_pred = 1 + 4 + self.classes  # 1 objness + 4 box + num_class
        self._num_anchors = anchors.size // 2
        self._stride = stride

        with self.name_scope():
            all_pred = self._num_pred * self._num_anchors
            self.prediction = nn.Conv2D(
                all_pred, kernel_size=1, padding=0, strides=1
            )

            # Anchors will be multiplied to predictions.
            anchors = anchors.reshape(1, 1, -1, 2)
            self.anchors = self.params.get_constant(f'anchor_{index}', anchors)

            # Offsets will be added to predictions.
            alloc_size = (128, 128)
            grid_x = np.arange(alloc_size[1])
            grid_y = np.arange(alloc_size[0])
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)

            # Stack to `(n, n, 2)`.
            offsets = np.concatenate([
                grid_x[:, :, np.newaxis],
                grid_y[:, :, np.newaxis]
            ], axis=-1)

            # Expand dims to `(1, 1, n, n, 2)` so it's easier for broadcasting.
            offsets = np.expand_dims(
                np.expand_dims(offsets, axis=0), axis=0
            )
            self.offsets = self.params.get_constant(f'offset_{index}', offsets)

    def hybrid_forward(self, F, x, anchors, offsets):
        """Hybrid Forward of YOLOV3Output layer.

        Parameters
        ----------
        F : mxnet.nd or mxnet.sym
            `F` is mxnet.sym if hybridized or mxnet.nd if not.
        x : mxnet.nd.NDArray
            Input feature map.
        anchors : mxnet.nd.NDArray
            Anchors loaded from self, no need to supply.
        offsets : mxnet.nd.NDArray
            Offsets loaded from self, no need to supply.

        Returns
        -------
        mxnet.nd.NDArray
            During inference, return detections.

        """
        # Prediction flat to `(batch, pred per pixel, height * width)`.
        pred = self.prediction(x).reshape(
            (0, self._num_anchors * self._num_pred, -1)
        )

        # Transpose to `(batch, height * width, num_anchor, num_pred)`.
        pred = pred.transpose(axes=(0, 2, 1)).reshape(
            (0, -1, self._num_anchors, self._num_pred)
        )

        # Components.
        raw_box_centers = pred.slice_axis(axis=-1, begin=0, end=2)
        raw_box_scales = pred.slice_axis(axis=-1, begin=2, end=4)
        objness = pred.slice_axis(axis=-1, begin=4, end=5)
        class_pred = pred.slice_axis(axis=-1, begin=5, end=None)

        # Valid offsets, `(1, 1, height, width, 2)`, and reshape to `(1, height
        # * width, 1, 2)`.
        offsets = F.slice_like(
            offsets, x * 0, axes=(2, 3)
        ).reshape(
            (1, -1, 1, 2)
        )

        box_centers = F.broadcast_add(
            F.sigmoid(raw_box_centers), offsets
        ) * self._stride
        box_scales = F.broadcast_mul(F.exp(raw_box_scales), anchors)
        confidence = F.sigmoid(objness)
        class_score = F.broadcast_mul(F.sigmoid(class_pred), confidence)
        wh = box_scales / 2.0
        bbox = F.concat(box_centers - wh, box_centers + wh, dim=-1)

        # Prediction per class.
        bboxes = F.tile(bbox, reps=(self.classes, 1, 1, 1, 1))
        scores = F.transpose(
            class_score, axes=(3, 0, 1, 2)
        ).expand_dims(axis=-1)
        ids = F.broadcast_add(
            scores * 0,
            F.arange(0, self.classes).reshape((0, 1, 1, 1, 1))
        )
        detections = F.concat(ids, scores, bboxes, dim=-1)

        # Reshape to (B, xx, 6).
        detections = F.reshape(
            detections.transpose(axes=(1, 0, 2, 3, 4)),
            (0, -1, 6)
        )

        return detections


class YOLODetectionBlockV3(gluon.HybridBlock):
    """YOLO V3 detection block

    Tasked with the following:
    - Add a few conv layers.
    - Return the output.
    - Have a branch that does yolo detection.

    Parameters
    ----------
    channel : int
        Number of channels for 1x1 conv. 3x3 Conv will have 2*channel.

    """
    def __init__(self, channel):
        super(YOLODetectionBlockV3, self).__init__()

        with self.name_scope():
            self.body = nn.HybridSequential(prefix='')
            for _ in range(2):
                # 1x1 reduce, then 3x3 expand.
                self.body.add(_conv2d(channel, 1, 0, 1))
                self.body.add(_conv2d(channel * 2, 3, 1, 1))

            self.body.add(_conv2d(channel, 1, 0, 1))
            self.tip = _conv2d(channel * 2, 3, 1, 1)

    def hybrid_forward(self, F, x):
        route = self.body(x)
        tip = self.tip(route)
        return route, tip


class FaceYOLOv3(gluon.HybridBlock):
    """YOLO V3 detection network.

    Parameters
    ----------
    stages : mxnet.gluon.HybridBlock
        Staged feature extraction blocks. For example, 3 stages and 3 YOLO
        output layers are used original paper.
    channels : iterable
        Number of conv channels for each appended stage.
        `len(channels)` should match `len(stages)`.
    anchors : iterable
        The anchor setting. `len(anchors)` should match `len(stages)`.
    strides : iterable
        Strides of feature map. `len(strides)` should match `len(stages)`.
    nms_thresh : float, default is 0.45.
        Non-maximum suppression threshold. You can specify < 0 or > 1 to
        disable NMS.
    nms_topk : int, default is 400
        Apply NMS to top k detection results, use -1 to disable so that every
         Detection result is used in NMS.
    post_nms : int, default is 100
        Only return top `post_nms` detection results, the rest is
        discarded. The number is based on COCO dataset which has maximum 100
        objects per image. You can adjust this number if expecting more
        objects. You can use -1 to return all detections.

    Reference: https://arxiv.org/pdf/1804.02767.pdf.

    """
    def __init__(
        self, stages, channels, anchors, strides, nms_thresh=0.45,
        nms_topk=200, post_nms=100,
    ):
        super(FaceYOLOv3, self).__init__()

        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms

        with self.name_scope():
            self.stages = nn.HybridSequential()
            self.transitions = nn.HybridSequential()
            self.yolo_blocks = nn.HybridSequential()
            self.yolo_outputs = nn.HybridSequential()

            # Note that anchors and strides should be used in reverse order.
            for idx, stage, channel, anchor, stride in zip(
                range(len(stages)),
                stages,
                channels,
                anchors[::-1],
                strides[::-1]
            ):
                self.stages.add(stage)
                self.yolo_blocks.add(YOLODetectionBlockV3(channel))
                self.yolo_outputs.add(
                    YOLOOutputV3(idx, anchor, stride)
                )

                if idx > 0:
                    self.transitions.add(_conv2d(channel, 1, 0, 1))

    def hybrid_forward(self, F, x):
        """YOLOV3 network hybrid forward.

        Parameters
        ----------
        F : mxnet.nd or mxnet.sym
            `F` is mxnet.sym if hybridized or mxnet.nd if not.
        x : mxnet.nd.NDArray
            Input data.

        Returns
        -------
        (tuple of) mxnet.nd.NDArray
            During inference, return detections in shape (B, N, 6) with format
            `(cid, score, xmin, ymin, xmax, ymax)`.

        """
        all_detections = []

        # Feed-forward the base network.
        routes = []
        for stage in self.stages:
            x = stage(x)
            routes.append(x)

        # The YOLO output layers are used in reverse order, i.e., from very
        # deep layers to shallow.
        for idx, block, output in zip(
            range(len(routes)), self.yolo_blocks, self.yolo_outputs
        ):
            x, tip = block(x)
            dets = output(tip)
            all_detections.append(dets)
            if idx >= len(routes) - 1:
                break

            # Add transition layers.
            x = self.transitions[idx](x)

            # Upsample feature map reverse to shallow layers.
            upsample = _upsample(x, stride=2)
            route_now = routes[::-1][idx + 1]
            x = F.concat(
                F.slice_like(upsample, route_now * 0, axes=(2, 3)), route_now,
                dim=1,
            )

        # Concat all detection results from different stages.
        result = F.concat(*all_detections, dim=1)

        # Apply nms per class.
        if self.nms_thresh > 0 and self.nms_thresh < 1:
            result = F.contrib.box_nms(
                result, overlap_thresh=self.nms_thresh, valid_thresh=0.01,
                topk=self.nms_topk, id_index=0, score_index=1, coord_start=2,
                force_suppress=False
            )
            if self.post_nms > 0:
                result = result.slice_axis(axis=1, begin=0, end=self.post_nms)

        ids = result.slice_axis(axis=-1, begin=0, end=1)
        scores = result.slice_axis(axis=-1, begin=1, end=2)
        bboxes = result.slice_axis(axis=-1, begin=2, end=None)

        return ids, scores, bboxes

    def set_nms(self, nms_thresh=0.45, nms_topk=200, post_nms=100):
        """Set non-maximum suppression parameters.

        Parameters
        ----------
        nms_thresh : float, default is 0.45.
            Non-maximum suppression threshold. You can specify < 0 or > 1 to
            disable NMS.
        nms_topk : int, default is 400
            Apply NMS to top k detection results, use -1 to disable so that
             every Detection result is used in NMS.
        post_nms : int, default is 100
            Only return top `post_nms` detection results, the rest is
            discarded. The number is based on COCO dataset which has maximum
            100 objects per image. You can adjust this number if expecting more
            objects. You can use -1 to return all detections.

        """
        self._clear_cached_op()
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms


def mobilefacedetnet_v2(model_path):
    """Mobilefacedet: A YOLO3-like multi-scale with mfdet24 base network for
    fast face detection.

    Parameters
    ----------
    model_path : str
        Model weights storing path.

    Returns
    -------
    mxnet.gluon.HybridBlock
        Fully hybrid mobilefacedet network.

    """
    # Initialize and prepare the base network.
    base_net = MFDetV1()

    stages = [
        base_net.features[:9],
        base_net.features[9:12],
        base_net.features[12:]
    ]

    anchors = [
        [10, 12, 16, 20, 23, 28],
        [43, 52, 60, 75, 80, 94],
        [118, 147, 186, 232, 285, 316]
    ]

    strides = [8, 16, 32]
    filters = [256, 128, 64]

    # Instantiate the actual detection network, a YOLOv3.
    net = FaceYOLOv3(stages, filters, anchors, strides)
    net.load_parameters(model_path, ctx=CTX)

    return net


def preprocess_images(images, short_side=416):
    """Pre-process images to feed in to detection model.

    Parameters
    ----------
    images : np.array, shape = (batch_size, H, W, C)
        Images to run face detection over.
    short_side : int
        Resize images' short side to `short_side` before sending over to
        detection network.

    """
    H, W = images.shape[1:3]
    scale = short_side / min(H, W)

    resizer = mx.gluon.data.vision.transforms.Resize(
        size=(int(W * scale), int(H * scale)), keep_ratio=True
    )
    X = resizer(
        mx.nd.array(images, ctx=CTX)
    )

    X = mx.nd.image.to_tensor(X)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    X = mx.nd.image.normalize(X, mean=mean, std=std)

    return X


def face_detection(images, short_side=416):
    """Perform face detection on `images`.

    Parameters
    ----------
    images : np.array, shape = (batch_size, H, W, C)
        Images to run face detection over.
    short_side : int
        Resize images' short side to `short_side` before sending over to
        detection network.

    Returns
    -------
    nested list of dicts
        Nested list of dicts. One entry per image, and inside one entry per
        face detected containing the bounding boxes (key ``bbox``).
    """
    SCORE_THRESHOLD = 0.5

    # Pre-process the images.
    X = preprocess_images(images, short_side=short_side)

    # Calculate the scale factor of the images.
    H, W = images.shape[1:3]
    scale = short_side / min(H, W)

    # Run the batch images through the detection network.
    net = _get_face_detection_model()
    batch_labels, batch_scores, batch_bboxes = net(X)

    batch_objects = []
    for img_idx in range(len(images)):
        scores = batch_scores[img_idx, :, 0].asnumpy()
        bboxes = batch_bboxes[img_idx].asnumpy()
        above_threshold = np.flatnonzero(scores > SCORE_THRESHOLD)

        objects = []
        for idx in above_threshold:
            # Re-scale and make sure all coordinates are above 0.
            x_min, y_min, x_max, y_max = [
                max(0, int(coord / scale)) for coord in bboxes[idx]
            ]

            # Make sure no coordinate goes over the full image width or height.
            x_min = min(x_min, W - 1)
            x_max = min(x_max, W - 1)
            y_min = min(y_min, H - 1)
            y_max = min(y_max, H - 1)

            # If face has no width or height, ignore.
            if x_max - x_min <= 0 or y_max - y_min <= 0:
                continue

            objects.append({
                'bbox': [x_min, y_min, x_max, y_max],
            })

        batch_objects.append(objects)

    return batch_objects
