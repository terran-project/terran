import math
import numpy as np
import os
import torch
import time

from torchvision.ops import nms

from terran import default_device
from terran.face.detection.retinaface.utils.generate_anchor import (
    generate_anchors_fpn, anchors_plane,
)
from terran.face.detection.retinaface.model import (
    RetinaFace as RetinaFaceModel
)


def load_model():
    model = RetinaFaceModel()
    model.load_state_dict(torch.load(
        os.path.expanduser('~/.terran/checkpoints/retinaface-mnet.pth')
    ))
    model.eval()
    return model


def clip_boxes(boxes, im_shape):
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def bbox_pred(anchors, deltas):
    """Apply the bbox delta predictions on the base anchor coordinates.

    Paramters
    ---------
    anchors : torch.Tensor of shape (A, 4)
    deltas : torch.Tensor of shape (N, A, 4)

    Returns
    -------
    torch.Tensor of shape (N, A, 4)
        Adjusted bounding boxes.

    """
    widths = anchors[:, 2] - anchors[:, 0] + 1.0
    heights = anchors[:, 3] - anchors[:, 1] + 1.0
    ctr_x = anchors[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = anchors[:, 1] + 0.5 * (heights - 1.0)

    dx = deltas[..., 0]  # (N, A)
    dy = deltas[..., 1]
    dw = deltas[..., 2]
    dh = deltas[..., 3]

    pred_ctr_x = dx * widths + ctr_x  # (N, A)
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    pred = deltas
    # pred_boxes = torch.zeros_like(deltas)
    pred[..., 0] = pred_ctr_x - 0.5 * (pred_w - 1.0)
    pred[..., 1] = pred_ctr_y - 0.5 * (pred_h - 1.0)
    pred[..., 2] = pred_ctr_x + 0.5 * (pred_w - 1.0)
    pred[..., 3] = pred_ctr_y + 0.5 * (pred_h - 1.0)

    return pred


def landmark_pred(anchors, deltas):
    """Apply the landmark delta predictions on the base anchor coordinates.

    Paramters
    ---------
    anchors : torch.Tensor of shape (A, 4)
    deltas : torch.Tensor of shape (N, A, 5, 2)

    Returns
    -------
    torch.Tensor of shape (N, A, 5, 2)
        Adjusted landmark coordinates.

    """
    widths = anchors[:, 2] - anchors[:, 0] + 1.0
    heights = anchors[:, 3] - anchors[:, 1] + 1.0
    ctr_x = anchors[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = anchors[:, 1] + 0.5 * (heights - 1.0)

    pred = deltas
    for i in range(5):
        pred[..., i, 0] = deltas[..., i, 0] * widths + ctr_x
        pred[..., i, 1] = deltas[..., i, 1] * heights + ctr_y

    return pred


class RetinaFace:

    def __init__(self, device=default_device, ctx_id=0, nms_threshold=0.4):
        self.device = device
        self.ctx_id = ctx_id
        self.nms_threshold = nms_threshold

        self._feat_stride_fpn = [32, 16, 8]
        self.anchor_cfg = {
            8: {
                'SCALES': (2, 1),
                'BASE_SIZE': 16,
                'RATIOS': (1,),
            },
            16: {
                'SCALES': (8, 4),
                'BASE_SIZE': 16,
                'RATIOS': (1,),
            },
            32: {
                'SCALES': (32, 16),
                'BASE_SIZE': 16,
                'RATIOS': (1,),
            },
        }

        self.fpn_keys = []
        for s in self._feat_stride_fpn:
            self.fpn_keys.append('stride%s' % s)

        self._anchors_fpn = dict(
            zip(
                self.fpn_keys,
                generate_anchors_fpn(
                    dense_anchor=False, cfg=self.anchor_cfg
                ),
            )
        )

        for k in self._anchors_fpn:
            v = self._anchors_fpn[k].astype(np.float32)
            self._anchors_fpn[k] = v

        self._num_anchors = dict(
            zip(
                self.fpn_keys,
                [anchors.shape[0] for anchors in self._anchors_fpn.values()],
            )
        )

        self.model = load_model().to(self.device)

    def call(self, images, threshold=0.5):
        """Run the detection.

        `images` is a (N, H, W, C)-shaped array (np.float32).

        (Padding must be performed outside.)
        """
        H, W = images.shape[1:3]

        # Load the batch in to a `torch.Tensor` and pre-process by turning it
        # into a BGR format for the channels.
        t0 = time.time()
        data = torch.tensor(
            images, device=self.device, dtype=torch.float32
        ).permute(0, 3, 1, 2).flip(1)

        torch.cuda.synchronize()

        # Run the images through the network.

        t1 = time.time()
        print(f'    Loading {1000 * (t1 - t0):.1f}ms ({1000 * (t1 - t0) / len(images):.1f}ms/img)')
        # TODO: Why does `eval` not `no_grad()`?
        with torch.no_grad():
            net_out = self.model(data)

        torch.cuda.synchronize()

        t2 = time.time()
        print(f'    Model {1000 * (t2 - t1):.1f}ms ({1000 * (t2 - t1) / len(images):.1f}ms/img)')

        anchors_per_stride = {}
        for stride in self._feat_stride_fpn:
            # Downsampling ratio for current stride.
            height = math.ceil(H / stride)
            width = math.ceil(W / stride)

            A = self._num_anchors[f'stride{stride}']
            K = height * width

            anchors_fpn = self._anchors_fpn[f'stride{stride}']
            anchors = anchors_plane(height, width, stride, anchors_fpn)
            anchors = torch.tensor(
                anchors.reshape((K * A, 4)),
                device=self.device
            )
            anchors_per_stride[stride] = anchors

        torch.cuda.synchronize()

        t3 = time.time()
        print(f'    Anchor-Stuff {1000 * (t3 - t2):.1f}ms ({1000 * (t3 - t2) / len(images):.1f}ms/img)')

        t4 = 0
        t5 = 0
        proposals_list = []
        scores_list = []
        landmarks_list = []
        for stride_idx, stride in enumerate(self._feat_stride_fpn):
            # Three per stride: class, bbox, landmark.
            idx = stride_idx * 3

            anchors = anchors_per_stride[stride]
            A = self._num_anchors[f'stride{stride}']
            N = net_out[idx].shape[0]

            t4_0 = time.time()
            scores = net_out[idx]
            scores = scores[:, A:, :, :]
            scores = scores.permute(0, 2, 3, 1).reshape(N, -1)

            bbox_deltas = net_out[idx + 1]
            bbox_pred_len = bbox_deltas.shape[1] // A
            bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).reshape(
                (N, -1, bbox_pred_len)
            )

            landmark_deltas = net_out[idx + 2]
            landmark_pred_len = landmark_deltas.shape[1] // A
            landmark_deltas = landmark_deltas.permute(0, 2, 3, 1).reshape(
                (N, -1, 5, landmark_pred_len // 5)
            )

            torch.cuda.synchronize()
            t4 += time.time() - t4_0

            t5_0 = time.time()
            proposals = bbox_pred(anchors, bbox_deltas)
            landmarks = landmark_pred(anchors, landmark_deltas)

            torch.cuda.synchronize()
            t5 += time.time() - t5_0

            scores_list.append(scores)
            proposals_list.append(proposals)
            landmarks_list.append(landmarks)

        all_scores = torch.cat(scores_list, dim=1)
        all_proposals = torch.cat(proposals_list, dim=1)
        all_landmarks = torch.cat(landmarks_list, dim=1)

        t6 = time.time()
        batch_objects = []
        for image_idx in range(images.shape[0]):
            scores = all_scores[image_idx]
            proposals = all_proposals[image_idx]
            landmarks = all_landmarks[image_idx]

            order = torch.where(scores >= threshold)[0]
            proposals = proposals[order, :]
            scores = scores[order]
            landmarks = landmarks[order, :]

            if proposals.shape[0] == 0:
                batch_objects.append([])
                continue

            # Re-order all proposals according to score.
            order = scores.argsort(descending=True)
            proposals = proposals[order]
            scores = scores[order]
            landmarks = landmarks[order]

            # Run the predictions through NMS.
            keep = nms(proposals, scores, self.nms_threshold)
            proposals = proposals[keep].to('cpu').numpy()
            scores = scores[keep].to('cpu').numpy()
            landmarks = landmarks[keep].to('cpu').numpy()

            batch_objects.append([
                {'bbox': b,  'landmarks': l, 'score': s}
                for s, b, l in zip(scores, proposals, landmarks)
            ])

        tf = time.time()
        print(f'    Preparation {1000 * (t4):.1f}ms ({1000 * (t4) / len(images):.1f}ms/img)')
        print(f'    Decoding {1000 * (t5):.1f}ms ({1000 * (t5) / len(images):.1f}ms/img)')
        print(f'    Per-image {1000 * (tf - t6):.1f}ms ({1000 * (tf - t6) / len(images):.1f}ms/img)')

        return batch_objects
