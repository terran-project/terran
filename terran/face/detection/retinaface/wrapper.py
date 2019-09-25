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
    """Apply the delta predictions on the base anchor coordinates.

    Paramters
    ---------
    anchors : torch.Tensor of shape Nx4
    deltas : torch.Tensor of shape Nx4

    Returns
    -------
    torch.Tensor of shape Nx4
        Adjusted bounding boxes.

    """
    if anchors.shape[0] == 0:
        return torch.zeros([0, 4])

    widths = anchors[:, 2] - anchors[:, 0] + 1.0
    heights = anchors[:, 3] - anchors[:, 1] + 1.0
    ctr_x = anchors[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = anchors[:, 1] + 0.5 * (heights - 1.0)

    dx = deltas[:, 0:1]
    dy = deltas[:, 1:2]
    dw = deltas[:, 2:3]
    dh = deltas[:, 3:4]

    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = torch.exp(dw) * widths[:, None]
    pred_h = torch.exp(dh) * heights[:, None]

    pred_boxes = torch.zeros_like(deltas)
    pred_boxes[:, 0:1] = pred_ctr_x - 0.5 * (pred_w - 1.0)
    pred_boxes[:, 1:2] = pred_ctr_y - 0.5 * (pred_h - 1.0)
    pred_boxes[:, 2:3] = pred_ctr_x + 0.5 * (pred_w - 1.0)
    pred_boxes[:, 3:4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

    if deltas.shape[1] > 4:
        pred_boxes[:, 4:] = deltas[:, 4:]

    return pred_boxes


def landmark_pred(anchors, deltas):
    if anchors.shape[0] == 0:
        return torch.zeros([0, deltas.shape[1]])

    widths = anchors[:, 2] - anchors[:, 0] + 1.0
    heights = anchors[:, 3] - anchors[:, 1] + 1.0
    ctr_x = anchors[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = anchors[:, 1] + 0.5 * (heights - 1.0)

    pred = torch.zeros_like(deltas)
    for i in range(5):
        pred[:, i, 0] = deltas[:, i, 0] * widths + ctr_x
        pred[:, i, 1] = deltas[:, i, 1] * heights + ctr_y

    return pred


class RetinaFace:

    def __init__(self, device=default_device, ctx_id=0, nms_threshold=0.4):
        self.device = device
        self.ctx_id = ctx_id
        self.nms_threshold = nms_threshold

        self._feat_stride_fpn = [32, 16, 8]
        self.anchor_cfg = {
            "8": {
                "SCALES": (2, 1),
                "BASE_SIZE": 16,
                "RATIOS": (1,),
            },
            "16": {
                "SCALES": (8, 4),
                "BASE_SIZE": 16,
                "RATIOS": (1,),
            },
            "32": {
                "SCALES": (32, 16),
                "BASE_SIZE": 16,
                "RATIOS": (1,),
            },
        }

        self.fpn_keys = []
        for s in self._feat_stride_fpn:
            self.fpn_keys.append("stride%s" % s)

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

        batch_objects = []
        for batch_idx in range(images.shape[0]):

            proposals_list = []
            scores_list = []
            landmarks_list = []
            for _idx, s in enumerate(self._feat_stride_fpn):
                stride = int(s)

                # Three per stride: class, bbox, landmark.
                idx = _idx * 3

                scores = net_out[idx]
                scores = scores[
                    [batch_idx], self._num_anchors[f'stride{s}']:, :, :
                ]
                # TODO: Try to use a `view` to avoid copying.
                scores = scores.permute(0, 2, 3, 1).reshape(-1)

                idx += 1
                bbox_deltas = net_out[idx]
                bbox_deltas = bbox_deltas[[batch_idx], ...]

                height, width = bbox_deltas.shape[2], bbox_deltas.shape[3]

                A = self._num_anchors[f'stride{s}']
                K = height * width

                anchors_fpn = self._anchors_fpn[f'stride{s}']
                anchors = anchors_plane(height, width, stride, anchors_fpn)
                anchors = torch.tensor(
                    anchors.reshape((K * A, 4)),
                    device=self.device
                )

                bbox_deltas = bbox_deltas.permute(0, 2, 3, 1)
                bbox_pred_len = bbox_deltas.shape[3] // A
                bbox_deltas = bbox_deltas.reshape(-1, bbox_pred_len)

                proposals = bbox_pred(anchors, bbox_deltas)
                # TODO: See whether to do here or at the end. Or even
                # outside. Try to use `torch.clamp`.
                # proposals = clip_boxes(proposals, [H, W])

                order = torch.where(scores >= threshold)[0]
                proposals = proposals[order, :]
                scores = scores[order]

                idx += 1
                landmark_deltas = net_out[idx]
                landmark_deltas = landmark_deltas[[batch_idx], ...]

                landmark_pred_len = landmark_deltas.shape[1] // A
                landmark_deltas = landmark_deltas.permute(0, 2, 3, 1).reshape(
                    (-1, 5, landmark_pred_len // 5)
                )
                landmarks = landmark_pred(anchors, landmark_deltas)
                landmarks = landmarks[order, :]

                scores_list.append(scores)
                proposals_list.append(proposals)
                landmarks_list.append(landmarks)

            scores = torch.cat(scores_list)
            proposals = torch.cat(proposals_list)
            landmarks = torch.cat(landmarks_list)

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

        t3 = time.time()
        print(f'    Postprocess {1000 * (t3 - t2):.1f}ms ({1000 * (t3 - t2) / len(images):.1f}ms/img)')

        return batch_objects
