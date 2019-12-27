import math
import torch

from torchvision.ops import nms

from terran import default_device
from terran.checkpoint import get_checkpoint_path
from terran.face.detection.retinaface.anchors import (
    anchors_plane, generate_anchor_reference,
)
from terran.face.detection.retinaface.model import (
    RetinaFace as RetinaFaceModel
)


def load_model():
    model = RetinaFaceModel()
    model.load_state_dict(torch.load(
        get_checkpoint_path('terran.face.detection.retinaface.RetinaFace')
    ))
    model.eval()
    return model


def decode_bboxes(anchors, deltas):
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

    # Perform the decoding in-place.
    pred = deltas
    pred[..., 0] = pred_ctr_x - 0.5 * (pred_w - 1.0)
    pred[..., 1] = pred_ctr_y - 0.5 * (pred_h - 1.0)
    pred[..., 2] = pred_ctr_x + 0.5 * (pred_w - 1.0)
    pred[..., 3] = pred_ctr_y + 0.5 * (pred_h - 1.0)

    return pred


def decode_landmarks(anchors, deltas):
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

    # Perform the decoding in-place.
    pred = deltas
    for i in range(5):
        pred[..., i, 0] = deltas[..., i, 0] * widths + ctr_x
        pred[..., i, 1] = deltas[..., i, 1] * heights + ctr_y

    return pred


class RetinaFace:

    def __init__(self, device=default_device, nms_threshold=0.4):
        self.device = device
        self.nms_threshold = nms_threshold

        # Anchor settings and feature strides used, specific to the `mnet`
        # backbone.
        self.feature_strides = [32, 16, 8]
        self.anchor_settings = {
            8: {
                'scales': (2, 1),
                'base_size': 16,
                'ratios': (1,),
            },
            16: {
                'scales': (8, 4),
                'base_size': 16,
                'ratios': (1,),
            },
            32: {
                'scales': (32, 16),
                'base_size': 16,
                'ratios': (1,),
            },
        }

        self.anchor_references = dict(zip(
            self.feature_strides,
            generate_anchor_reference(
                settings=self.anchor_settings, device=self.device
            )
        ))

        self.num_anchors_per_stride = {
            stride: anchors.shape[0]
            for stride, anchors in self.anchor_references.items()
        }

        self.model = load_model().to(self.device)

    def call(self, images, threshold=0.5):
        """Run the detection.

        `images` is a (N, H, W, C)-shaped array (np.float32).

        (Padding must be performed outside.)
        """
        H, W = images.shape[1:3]

        # Load the batch in to a `torch.Tensor` and pre-process by turning it
        # into a BGR format for the channels.
        data = torch.as_tensor(
            images, device=self.device, dtype=torch.float32
        ).permute(0, 3, 1, 2).flip(1)

        # Run the images through the network. Disable gradients, as they're not
        # needed.
        with torch.no_grad():
            output = self.model(data)

        # Calculate the base anchor coordinates per stride of the FPN.
        anchors_per_stride = {}
        for stride in self.feature_strides:
            # Input dimensions after model downsampling (i.e. feature map
            # dimensions).
            height = math.ceil(H / stride)
            width = math.ceil(W / stride)

            anchor_ref = self.anchor_references[stride]
            anchors = anchors_plane(anchor_ref, height, width, stride)
            anchors_per_stride[stride] = anchors

        # Decode the outputs of the model, adjusting the anchors.
        proposals_list = []
        scores_list = []
        landmarks_list = []
        for stride_idx, stride in enumerate(self.feature_strides):
            # Three per stride: class, bbox, landmark.
            idx = stride_idx * 3

            anchors = anchors_per_stride[stride]
            A = self.num_anchors_per_stride[stride]
            N = output[idx].shape[0]

            scores = output[idx]
            scores = scores[:, A:, :, :]
            scores = scores.permute(0, 2, 3, 1).reshape(N, -1)

            bbox_deltas = output[idx + 1]
            bbox_pred_len = bbox_deltas.shape[1] // A
            bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).reshape(
                (N, -1, bbox_pred_len)
            )

            landmark_deltas = output[idx + 2]
            landmark_pred_len = landmark_deltas.shape[1] // A
            landmark_deltas = landmark_deltas.permute(0, 2, 3, 1).reshape(
                (N, -1, 5, landmark_pred_len // 5)
            )

            proposals = decode_bboxes(anchors, bbox_deltas)
            landmarks = decode_landmarks(anchors, landmark_deltas)

            scores_list.append(scores)
            proposals_list.append(proposals)
            landmarks_list.append(landmarks)

        batch_scores = torch.cat(scores_list, dim=1)
        batch_proposals = torch.cat(proposals_list, dim=1)
        batch_landmarks = torch.cat(landmarks_list, dim=1)

        # Collect the predictions per image, filtering low-scoring proposals
        # and performing NMS.
        batch_objects = []
        for image_idx in range(images.shape[0]):
            scores = batch_scores[image_idx]
            proposals = batch_proposals[image_idx]
            landmarks = batch_landmarks[image_idx]

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

            # Run the predictions through non-maximum suppression.
            keep = nms(proposals, scores, self.nms_threshold)
            proposals = proposals[keep].to('cpu').numpy()
            scores = scores[keep].to('cpu').numpy()
            landmarks = landmarks[keep].to('cpu').numpy()

            batch_objects.append([
                {'bbox': b, 'landmarks': l, 'score': s}
                for s, b, l in zip(scores, proposals, landmarks)
            ])

        return batch_objects
