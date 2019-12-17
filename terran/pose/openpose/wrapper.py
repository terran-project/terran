import numpy as np
import lycon
import os
import torch

from lycon import resize

from terran import default_device
from terran.pose.openpose.model import BodyPoseModel


def load_model():
    model = BodyPoseModel()
    model.load_state_dict(torch.load(
        os.path.expanduser('~/.terran/checkpoints/openpose-body.pth')
    ))
    model.eval()
    return model


def _get_keypoints(candidates, subsets, scale=1.0):
    k = subsets.shape[0]
    keypoints = np.zeros((k, 18, 3), dtype=np.int32)
    for i in range(k):
        for j in range(18):
            index = np.int32(subsets[i][j])
            if index != -1:
                y, x = candidates[index][:2]
                y = (y.cpu().numpy() / scale).astype(np.int32)
                x = (x.cpu().numpy() / scale).astype(np.int32)
                keypoints[i][j] = (x, y, 1)
    return keypoints


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
            image, output=resized[idx],
            width=new_size[0], height=new_size[1],
            interpolation=lycon.Interpolation.LINEAR
        )

    return resized, scale


def preprocess_images(images):
    # Turn into `BCHW` format.
    out = np.transpose(images, (0, 3, 1, 2))
    out = out.astype(np.float32) / 255.0 - 0.5
    out = torch.as_tensor(out, device=default_device)
    return out


@torch.jit.script
def build_segments(loc_src, loc_dst, num_midpoints: int):
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

    def __init__(self, device=default_device):
        self.device = device
        self.model = load_model().to(self.device)

    def call(self, images):
        # t0 = time.time()

        # Add batch dimension if missing.
        if len(images.shape) == 3:
            images = np.expand_dims(images, 0)

        # Optimal seems to be 386 actually, but twice as slow.
        short_side = 184
        downsampling_ratio = 8

        keypoint_threshold = 0.1
        thresh_2 = 0.05

        resized, scale = resize_images(images, short_side=short_side)
        preprocessed = preprocess_images(resized)

        # torch.cuda.synchronize()
        # t1 = time.time()

        with torch.no_grad():
            pafs, heatmaps = self.model(preprocessed)

        # torch.cuda.synchronize()
        # t2 = time.time()

        # TODO: Resizing to `resized` size. Could be to full image size so that
        # the results are even more exact.
        # TODO: Not adding padding, so it won't detect keypoints in the right
        # border of the image.
        pafs = torch.nn.functional.interpolate(
            pafs,
            scale_factor=downsampling_ratio,
            mode='bicubic', align_corners=False,
        )
        heatmaps = torch.nn.functional.interpolate(
            heatmaps,
            scale_factor=downsampling_ratio,
            mode='bicubic', align_corners=False,
        )

        # torch.cuda.synchronize()
        # t3 = time.time()

        # TODO: Support batch sizes.

        num_peaks = 0

        peak_locs = []
        peak_scores = []
        peak_ids = []

        # TODO: Why 18 and not 19?
        for part in range(18):
            heatmap = heatmaps[:, part:part + 1, :, :]

            # First index is the batch, second the 1-channel dimension.
            heatmap = heatmap[0, 0]

            # Search for local optima. Consider a 1px padding around the map,
            # as we need to make sure it's larger than any surrounding coord.
            peaks_binary = (
                (heatmap[1:-1, 1:-1] >= heatmap[0:-2, 1:-1])
                & (heatmap[1:-1, 1:-1] >= heatmap[1:-1, :-2])
                & (heatmap[1:-1, 1:-1] >= heatmap[2:, 1:-1])
                & (heatmap[1:-1, 1:-1] >= heatmap[1:-1, 2:])
                & (heatmap[1:-1, 1:-1] >= keypoint_threshold)
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

        map_idx = [
            [31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44],
            [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30],
            [47, 48], [49, 50], [53, 54], [51, 52], [55, 56], [37, 38],
            [45, 46]
        ]
        limbseq = [
            [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
            [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
            [1, 16], [16, 18], [3, 17], [6, 18]
        ]

        # torch.cuda.synchronize()
        # t4 = time.time()

        all_connections = []
        spl_k = []
        num_midpoints = 10

        # tgg = 0
        for k in range(len(map_idx)):
            paf = pafs[0, [x - 19 for x in map_idx[k]], :, :]

            # Keypoint IDs for limb.
            kpid_src = limbseq[k][0] - 1
            kpid_dst = limbseq[k][1] - 1

            loc_src = peak_locs[kpid_src]
            loc_dst = peak_locs[kpid_dst]

            count_src = loc_src.shape[0]
            count_dst = loc_dst.shape[0]

            if count_src == 0 or count_dst == 0:
                spl_k.append(k)
                all_connections.append([])
                continue

            directions = (
                loc_dst.reshape(1, -1, 2) - loc_src.reshape(-1, 1, 2)
            ).type(torch.float32)
            norms = torch.norm(directions, dim=2)
            directions = directions / norms[..., None]

            # tg = time.time()
            segments = build_segments(
                loc_src, loc_dst, num_midpoints
            ).type(torch.long)
            # tgg += time.time() - tg

            midpoint_scores = torch.mul(
                paf[
                    :, segments[..., 0], segments[..., 1]
                ].permute(3, 0, 1, 2),
                # Flip the directions, as the network output is in `(x, y)` and
                # our `directions` vector in `(y, x)`.
                torch.flip(directions.permute(2, 0, 1), dims=(0,))
            ).sum(dim=1)

            # Score with length regularization.
            # TODO: Where does this heuristic come from? Does it have to be
            # like this? Why heigth and not width?
            reg_scores = (
                midpoint_scores.sum(dim=0) / midpoint_scores.shape[0]
                + torch.clamp(0.5 * pafs.shape[2] / norms - 1, max=0)
            )

            criterion_1 = (
                (midpoint_scores > thresh_2).sum(dim=0) > 0.8 * num_midpoints
            )
            criterion_2 = (reg_scores > 0)

            matching = (criterion_1 & criterion_2).nonzero()
            matching_scores = reg_scores[matching[:, 0], matching[:, 1]]

            connections = []
            seen = set()

            # TODO: Improve this part.
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

        candidate = np.array([
            tuple(p) + (sc,)
            for pks, scs in zip(peak_locs, peak_scores)
            for p, sc in zip(pks, scs)
        ])
        subset = np.ones((0, 20)) * -1

        # torch.cuda.synchronize()
        # t5 = time.time()

        for k in range(len(map_idx)):
            if k in spl_k:
                continue

            part_As = all_connections[k][:, 0]
            part_Bs = all_connections[k][:, 1]
            index_A, index_B = np.array(limbseq[k]) - 1

            for i in range(len(all_connections[k])):
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):
                    if (
                        subset[j][index_A] == part_As[i]
                        or subset[j][index_B] == part_Bs[i]
                    ):
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if subset[j][index_B] != part_Bs[i]:
                        subset[j][index_B] = part_Bs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += (
                            candidate[part_Bs[i].astype(int), 2]
                            + all_connections[k][i][2]
                        )
                elif found == 2:
                    j1, j2 = subset_idx
                    membership = (
                        (subset[j1] >= 0).astype(int)
                        + (subset[j2] >= 0).astype(int)
                    )[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += all_connections[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:
                        subset[j1][index_B] = part_Bs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += (
                            candidate[part_Bs[i].astype(int), 2]
                            + all_connections[k][i][2]
                        )
                elif not found and k < 17:
                    row = np.ones(20) * -1
                    row[index_A] = part_As[i]
                    row[index_B] = part_Bs[i]
                    row[-1] = 2
                    row[-2] = sum(
                        candidate[all_connections[k][i, :2].astype(int), 2]
                    ) + all_connections[k][i][2]
                    subset = np.vstack([subset, row])

        # torch.cuda.synchronize()
        # t6 = time.time()

        del_idx = []
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                del_idx.append(i)
        subset = np.delete(subset, del_idx, axis=0)

        # Adjust the final keypoint coordinates to the initial image size,
        # considering the scaling factor and the downsampling ratio of the
        # network.
        kps = _get_keypoints(
            candidate, subset, scale=scale  # / downsampling_ratio
        )

        # TODO: Not returning any score at all.

        # torch.cuda.synchronize()
        # t7 = time.time()

        # print(f't1 = {t1 - t0:.3f}s')
        # print(f't2 = {t2 - t1:.3f}s')
        # print(f't3 = {t3 - t2:.3f}s')
        # print(f't4 = {t4 - t3:.3f}s')
        # print(f't5 = {t5 - t4:.3f}s')
        # print(f'(tg = {tgg:.3f})s')
        # print(f't6 = {t6 - t5:.3f}s')
        # print(f't7 = {t7 - t6:.3f}s')
        # print(f'total = {t7 - t0:.3f}s')

        return kps
