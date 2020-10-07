import numpy as np
import torch

from cv2 import resize, INTER_LINEAR

from terran import default_device
from terran.checkpoint import get_checkpoint_path
from terran.pose.openpose.model import BodyPoseModel


# TODO: Change names.
map_idx = [
    [31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44],
    [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30],
    [47, 48], [49, 50], [53, 54], [51, 52], [55, 56], [37, 38],
    [45, 46]
]

limbseq = [
    [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9],
    [9, 10], [10, 11], [2, 12], [12, 13], [13, 14], [2, 1],
    [1, 15], [15, 17], [1, 16], [16, 18], [3, 17], [6, 18]
]


def load_model():
    model = BodyPoseModel()
    model.load_state_dict(
        torch.load(
            get_checkpoint_path('terran.pose.openpose.OpenPose')
        )
    )
    model.eval()
    return model


def get_keypoints(peaks_by_id, humans, scale=1.0):
    """Build and return the final list of keypoints.

    Also adjust the final keypoint coordinates to the initial image size,
    considering the scaling factor.

    Parameters
    ----------
    peaks_by_id : np.ndarray of shape (N_k, 3)
        Peaks indexed by keypoint id, each row containing location and score.
    humans : np.ndarray of shape (N_d, 20)
        Humans found, with the corresponding keypoint ID at each of the first
        18 locations, plus number of keypoints and scores.
    scale : float
        Scale the image was resized with, needed to adjust final keypoint
        locations.

    Returns
    -------
    List of dicts
        Each dict corresponds to a detected human, and has two entries:
        * `keypoints`: a `np.ndarray`s of size (N_d, 18, 3).
        * `score`: the average score for the human's keypoints.

        `keypoints` is an array containing the keypoint locations for all the
        18 keypoints. The final axis contains the (x, y) coordinates, along
        with a third entry indicating whether the keypoint is present or not.

    """
    detections = []

    if torch.is_tensor(peaks_by_id):
        peaks_by_id = peaks_by_id.cpu().numpy()

    for human in humans:
        keypoints = np.zeros((18, 3), dtype=np.int32)
        for j in range(18):
            peak_id = np.int32(human[j])
            if peak_id != -1:
                y, x = peaks_by_id[peak_id][:2]

                y = (y / scale).astype(np.int32)
                x = (x / scale).astype(np.int32)
                keypoints[j] = (x, y, 1)

        # Average keypoint score: sum of scores over number of keypoints.
        score = human[-2] / human[-1]

        detections.append({
            'keypoints': keypoints,
            'score': score,
        })

    return detections


def resize_images(images, short_side=416):
    H, W = images.shape[1:3]
    scale = short_side / min(H, W)

    new_size = (
        int(W * scale), int(H * scale)
    )

    resized = np.empty(
        (images.shape[0], new_size[1], new_size[0], images.shape[3]),
        dtype=images.dtype
    )
    for idx, image in enumerate(images):
        resize(
            src=image,
            dst=resized[idx],
            dsize=new_size,
            interpolation=INTER_LINEAR,
        )

    return resized, scale


def preprocess_images(images):
    """Preprocess images as required by the pre-trained OpenPose model."""
    # Turn into `BCHW` format.
    out = np.transpose(images, (0, 3, 1, 2))
    out = out.astype(np.float32) / 255.0 - 0.5
    out = torch.as_tensor(out, device=default_device)
    return out


@torch.jit.script
def build_segments(loc_src, loc_dst, num_midpoints: int):
    """Build the `num_midpoint`-point segment between each source and
    destination keypoints of a given limb.

    Parameters
    ----------
    loc_src : torch.Tensor of shape (N_s, 2)
        Locations for the keypoints of one extreme of the limb.
    loc_dst : torch.Tensor of shape (N_d, 2)
        Locations for the keypoints of the other extreme of the limb.
    num_midpoints : int
        Number of points to return per segment.

    Returns
    -------
    torch.Tensor of shape (N_s, N_d, num_midpoints, 2)
        Locations for each of the points between each pair of keypoints.

    """
    count_src = loc_src.shape[0]
    count_dst = loc_dst.shape[0]

    segments = torch.zeros(
        count_src, count_dst, num_midpoints, 2,
        dtype=torch.float32, device=loc_src.device
    )
    for i in range(count_src):
        for j in range(count_dst):
            torch.linspace(
                loc_src[i][0], loc_dst[j][0], steps=num_midpoints,
                out=segments[i, j, :, 0]
            )
            torch.linspace(
                loc_src[i][1], loc_dst[j][1], steps=num_midpoints,
                out=segments[i, j, :, 1]
            )

    return segments


class OpenPose:

    def __init__(self, device=default_device, short_side=184):
        self.device = device
        self.model = load_model().to(self.device)

        self.short_side = short_side

        # Downsampling ratio for the model in use.
        self.downsampling_ratio = 8

        # Keypoint thresholds.
        self.keypoint_threshold = 0.1
        self.thresh_2 = 0.05
        self.human_threshold = 0.4

    def call(self, images):
        """Run the pose estimation model and its postprocessing.

        The preprocessing goes as follow:
        - Get the heatmaps from the model to identify peaks, i.e. candidate
          keypoint locations.
        - Use the PAFs from the model to obtain the limbs. In order to do so,
          for each limb, pick the pair of keypoints that maximizes the line
          integral between both extremes of the limb.
        - Build humans by filling in the needed limbs one by one. In case of a
          conflict (i.e. a limb belongs to two humans), tiebreak by assigning
          to one of them.

        Parameters
        ----------
        images : np.ndarray of shape (N, H, W, C)
            Batch of images to run model through.

        Returns
        -------
        List of humans detected by batch. Each person is specified by a
        `np.ndarry` of shape (18, 3).

        """
        resized, scale = resize_images(images, short_side=self.short_side)
        preprocessed = preprocess_images(resized)

        with torch.no_grad():
            batch_pafs, batch_heatmaps = self.model(preprocessed)

        # TODO: Not adding padding, so it won't detect keypoints in the right
        # border of the image.
        batch_pafs = torch.nn.functional.interpolate(
            batch_pafs,
            scale_factor=self.downsampling_ratio,
            mode='bicubic', align_corners=False,
        )
        batch_heatmaps = torch.nn.functional.interpolate(
            batch_heatmaps,
            scale_factor=self.downsampling_ratio,
            mode='bicubic', align_corners=False,
        )

        batch_objects = []
        for heatmaps, pafs in zip(batch_heatmaps, batch_pafs):

            # Start by obtaining the peak locations for every keypoint, by
            # looking at the returned heatmap.
            num_peaks = 0
            peak_locs = []
            peak_scores = []
            peak_ids = []

            for part in range(18):
                heatmap = heatmaps[part, :, :]

                # Search for local optima. Consider a 1px padding around the
                # map, as we need to make sure it's larger than any surrounding
                # coord.
                peaks_binary = (
                    (heatmap[1:-1, 1:-1] >= heatmap[0:-2, 1:-1])
                    & (heatmap[1:-1, 1:-1] >= heatmap[1:-1, :-2])
                    & (heatmap[1:-1, 1:-1] >= heatmap[2:, 1:-1])
                    & (heatmap[1:-1, 1:-1] >= heatmap[1:-1, 2:])
                    & (heatmap[1:-1, 1:-1] >= self.keypoint_threshold)
                )

                # Add one to the coordinates to account for the 1px padding.
                curr_peaks = torch.nonzero(peaks_binary) + 1
                curr_num_peaks = curr_peaks.shape[0]

                curr_peak_ids = torch.arange(
                    num_peaks, num_peaks + curr_num_peaks
                )
                curr_scores = heatmap[curr_peaks[:, 0], curr_peaks[:, 1]]

                peak_locs.append(curr_peaks)
                peak_scores.append(curr_scores)
                peak_ids.append(curr_peak_ids)

                num_peaks += curr_num_peaks

            # Now build the limbs out of the detected keypoints.

            # `all_connections` will be a list containing number-of-limbs
            # elements, each with a list of connections for said limb. Within
            # those list, the connection is represented by a 3-tuple of the
            # peak IDs of both extremes and the limb score.
            all_connections = []
            missing_limbs = []
            num_midpoints = 10

            for limb_id in range(len(map_idx)):
                # Calculate line integrals between each source and destination
                # keypoints for the current limb. Get the segments between the
                # points, the directions, and multiply by the values of the
                # vector field.
                paf = pafs[[x - 19 for x in map_idx[limb_id]], :, :]

                # Keypoint IDs for limb.
                kpid_src = limbseq[limb_id][0] - 1
                kpid_dst = limbseq[limb_id][1] - 1

                loc_src = peak_locs[kpid_src]
                loc_dst = peak_locs[kpid_dst]

                count_src = loc_src.shape[0]
                count_dst = loc_dst.shape[0]

                if count_src == 0 or count_dst == 0:
                    missing_limbs.append(limb_id)
                    all_connections.append(())
                    continue

                directions = (
                    loc_dst.reshape(1, -1, 2) - loc_src.reshape(-1, 1, 2)
                ).type(torch.float32)
                norms = torch.norm(directions, dim=2)
                directions = directions / norms[..., None]

                # Cast it to `torch.long` so we transform it into PAF
                # locations, for easy evaluation.
                segments = build_segments(
                    loc_src, loc_dst, num_midpoints
                ).type(torch.long)

                midpoint_scores = torch.mul(
                    paf[
                        :, segments[..., 0], segments[..., 1]
                    ].permute(3, 0, 1, 2),
                    # Flip the directions, as the network output is in `(x, y)`
                    # and our `directions` vector in `(y, x)`.
                    torch.flip(directions.permute(2, 0, 1), dims=(0,))
                ).sum(dim=1)

                # Score with length regularization.
                # TODO: Where does this heuristic come from? Does it have to be
                # like this? Why heigth and not width?
                reg_scores = (
                    midpoint_scores.sum(dim=0) / midpoint_scores.shape[0]
                    + torch.clamp(0.5 * pafs.shape[1] / norms - 1, max=0)
                )

                criterion_1 = (
                    (
                        midpoint_scores > self.thresh_2
                    ).sum(dim=0) > 0.8 * num_midpoints
                )
                criterion_2 = (reg_scores > 0)

                matching = (criterion_1 & criterion_2).nonzero()
                matching_scores = reg_scores[matching[:, 0], matching[:, 1]]

                connections = []
                seen = set()

                # Perform the actual matching by greedily building connections
                # with the highest-scoring pairs.
                # TODO: Improve from this part onwards..
                for match in matching.cpu().numpy()[
                    np.argsort(-matching_scores.cpu().numpy())
                ]:
                    i, j = match
                    s = reg_scores[i, j].cpu().numpy()
                    if i not in seen and j not in seen:
                        connections.append(
                            np.array([
                                peak_ids[kpid_src][i],
                                peak_ids[kpid_dst][j],
                                s
                            ])
                        )

                        if len(connections) >= min(count_src, count_dst):
                            break

                        seen.add(i)
                        seen.add(j)

                if connections:
                    connections = np.stack(connections)
                else:
                    connections = np.zeros((0, 3))

                all_connections.append(connections)

            # Build each person by progressively merging the found limbs.

            # Peak location and score, indexed by ID.
            peaks_by_id = np.array([
                tuple(p) + (sc,)
                for pks, scs in zip(peak_locs, peak_scores)
                for p, sc in zip(pks, scs)
            ])

            # First 18 entries are the keypoints, following one is the number
            # of keypoints, and the last one is the sum of the connection
            # scores.
            humans = np.ones((0, 20)) * -1

            for limb_id in range(len(map_idx)):
                if limb_id in missing_limbs:
                    continue

                peak_src = all_connections[limb_id][:, 0]
                peak_dst = all_connections[limb_id][:, 1]
                kpid_src, kpid_dst = np.array(limbseq[limb_id]) - 1

                for conn_idx in range(len(all_connections[limb_id])):

                    # Check for matches of the current connection with existing
                    # humans. By construction, up to two matches may occur.
                    matched_with = []
                    for human_idx, human in enumerate(humans):
                        if (
                            human[kpid_src] == peak_src[conn_idx]
                            or human[kpid_dst] == peak_dst[conn_idx]
                        ):
                            matched_with.append(human_idx)

                    if len(matched_with) == 1:
                        # There was a match with only one human. Add the
                        # destination keypoint to it. We don't add the source
                        # keypoint, as it must have been added already, due to
                        # the limbs being iterated in order.
                        human = humans[matched_with[0]]
                        if human[kpid_dst] != peak_dst[conn_idx]:
                            human[kpid_dst] = peak_dst[conn_idx]
                            human[-1] += 1
                            human[-2] += (
                                peaks_by_id[peak_dst[conn_idx].astype(int), 2]
                                + all_connections[limb_id][conn_idx][2]
                            )

                    elif len(matched_with) == 2:
                        human_1_idx, human_2_idx = matched_with
                        human_1 = humans[human_1_idx]
                        human_2 = humans[human_2_idx]

                        # Check the number of keypoints present in both humans
                        # matched. If they're non-overlapping, then we're
                        # connecting two human parts, so merge them into a
                        # single human.
                        membership = (
                            (human_1 >= 0).astype(int)
                            + (human_2 >= 0).astype(int)
                        )[:-2]
                        non_overlapping = (
                            len(np.flatnonzero(membership == 2)) == 0
                        )
                        if non_overlapping:
                            # Merge both humans and sum the number of keypoints
                            # and scores.
                            # The `+1` because absence of the limb is marked as
                            # `-1`.
                            human_1[:-2] += (human_2[:-2] + 1)
                            human_1[-2:] += human_2[-2:]
                            human_1[-2] += (
                                all_connections[limb_id][conn_idx][2]
                            )
                            humans = np.delete(humans, human_2_idx, 0)
                        else:
                            # There's a conflict, as the people overlap.
                            # Tiebreak by adding the connection into the first
                            # one.
                            human_1[kpid_dst] = peak_dst[conn_idx]
                            human_1[-1] += 1
                            human_1[-2] += (
                                peaks_by_id[peak_dst[conn_idx].astype(int), 2]
                                + all_connections[limb_id][conn_idx][2]
                            )

                    elif not matched_with and limb_id < 17:
                        # New human found, add a row.
                        human = np.ones(20) * -1
                        human[kpid_src] = peak_src[conn_idx]
                        human[kpid_dst] = peak_dst[conn_idx]
                        human[-1] = 2
                        human[-2] = sum(
                            peaks_by_id[
                                all_connections[limb_id][
                                    conn_idx, :2
                                ].astype(int),
                                2
                            ]
                        ) + all_connections[limb_id][conn_idx][2]
                        humans = np.vstack([humans, human])

            # Delete all detected humans with less than four keypoints and an
            # average keypoint score less than `0.4`.
            to_delete = []
            for human_idx, human in enumerate(humans):
                num_keypoints = human[-1]
                avg_score = human[-2] / human[-1]
                if num_keypoints < 4 or avg_score < self.human_threshold:
                    to_delete.append(human_idx)
            humans = np.delete(humans, to_delete, axis=0)

            # Build the final list of keypoints.
            batch_objects.append(
                get_keypoints(peaks_by_id, humans, scale=scale)
            )

        return batch_objects
