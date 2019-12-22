import torch
import numpy as np

from terran import default_device


def anchors_plane(anchor_ref, feat_h, feat_w, stride):
    """Builds the anchors plane for the given feature map shape.

    Based on an anchor reference, reproduces it at every point of the feature
    map, according to the specified stride. `anchor_ref` is set in real-image
    coordinates, so we must know what the stride for the current feature map
    is.

    Parameters
    ----------
    anchor_ref : torch.Tensor of size (A, 4)
        Coordinates for each of the `A` anchors, centered at the origin.
    feat_h : int
        Height of the feature map.
    feat_w : int
        Width of the feature map.
    stride : int
        Number of pixels every which to apply the anchor reference.

    Returns
    -------
    torch.Tensor of size (feat_h * feat_w * A, 4)
        The dtype and device of the returned tensor is based on `anchor_ref`.

    """
    device = anchor_ref.device
    dtype = anchor_ref.dtype

    shift_y, shift_x = torch.meshgrid(
        torch.arange(feat_h, dtype=dtype, device=device) * stride,
        torch.arange(feat_w, dtype=dtype, device=device) * stride,
    )

    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)

    shifts = torch.stack([
        shift_x, shift_y, shift_x, shift_y,
    ], dim=-1)

    anchors = (
        anchor_ref[None, ...] + shifts[:, None, :]
    ).reshape(-1, 4)

    return anchors


def generate_anchor_reference(settings=None, device=default_device):
    """Generate anchor reference per stride, for the given settings."""
    feature_strides = sorted(settings.keys(), reverse=True)

    anchors = []
    for stride in feature_strides:
        base_size = settings[stride]["base_size"]
        ratios = np.array(settings[stride]["ratios"])
        scales = np.array(settings[stride]["scales"])

        anchor = torch.as_tensor(
            generate_anchors(base_size, ratios, scales, stride),
            dtype=torch.float32,
            device=default_device
        )

        anchors.append(anchor)

    return anchors


def generate_anchors(base_size, ratios, scales, stride):
    """Generate an anchor reference for the given properties."""
    base_anchor = np.array([0, 0, base_size - 1, base_size - 1])
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([
        _scale_enum(ratio_anchors[i, :], scales)
        for i in range(ratio_anchors.shape[0])
    ])

    return anchors


def _whctrs(anchor):
    """Return width, height, x center, and y center for an anchor (window)."""
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    ctr_x = anchor[0] + 0.5 * (w - 1)
    ctr_y = anchor[1] + 0.5 * (h - 1)
    return w, h, ctr_x, ctr_y


def _mkanchors(ws, hs, ctr_x, ctr_y):
    """Given a vector of widths (ws) and heights (hs) around a center
    (ctr_x, ctr_y), output a set of anchors (windows).
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]

    anchors = np.hstack([
        ctr_x - 0.5 * (ws - 1),
        ctr_y - 0.5 * (hs - 1),
        ctr_x + 0.5 * (ws - 1),
        ctr_y + 0.5 * (hs - 1),
    ])
    return anchors


def _ratio_enum(anchor, ratios):
    """Enumerate a set of anchors for each aspect ratio wrt an anchor."""
    w, h, ctr_x, ctr_y = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios

    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)

    anchors = _mkanchors(ws, hs, ctr_x, ctr_y)

    return anchors


def _scale_enum(anchor, scales):
    """Enumerate a set of anchors for each scale wrt an anchor."""
    w, h, ctr_x, ctr_y = _whctrs(anchor)
    ws = w * scales
    hs = h * scales

    anchors = _mkanchors(ws, hs, ctr_x, ctr_y)

    return anchors
