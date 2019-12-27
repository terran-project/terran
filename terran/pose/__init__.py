import math
import numpy as np

from enum import Enum

from terran import default_device
from terran.checkpoint import get_class_for_checkpoint


TASK_NAME = 'pose-estimation'


class Keypoint(Enum):
    NOSE = 0
    NECK = 1

    R_SHOULDER = 2
    R_ELBOW = 3
    R_HAND = 4

    L_SHOULDER = 5
    L_ELBOW = 6
    L_HAND = 7

    R_HIP = 8
    R_KNEE = 9
    R_FOOT = 10

    L_HIP = 11
    L_KNEE = 12
    L_FOOT = 13

    R_EYE = 14
    L_EYE = 15
    R_EAR = 16
    L_EAR = 17


# TODO: This is *almost* like `terran.face.detection.merge_factory`. Unify them
# so we don't repeat so much code.
def merge_factory(method='padding'):
    """Merges a list of images into a single array.

    May do so by padding or cropping images. See `Estimation` for more
    information on the options.
    """

    def merge_in(images):
        # Check if not merged already and return early.
        if isinstance(images, np.ndarray):
            return images, {'merged': False}

        # TODO: Why is this commented?
        # if len(images) == 1:
        #     return np.array(images), {'merged': False}

        params = {'merged': True}

        if method == 'crop':
            raise NotImplementedError
        elif method == 'padding':
            max_height = max([arr.shape[0] for arr in images])
            max_width = max([arr.shape[1] for arr in images])
            padded = np.zeros(
                (len(images), max_height, max_width, 3),
                dtype=np.uint8
            )

            pads_per_image = []
            for idx, image in enumerate(images):
                diff_height = max(0, (max_height - image.shape[0]) / 2)
                diff_width = max(0, (max_width - image.shape[1]) / 2)
                pad_values = [
                    (
                        int(math.ceil(diff_height)),
                        int(math.floor(diff_height))
                    ),
                    (
                        int(math.ceil(diff_width)),
                        int(math.floor(diff_width))
                    ),
                    (0, 0),
                ]
                padded[idx, ...] = np.pad(image, pad_values)
                pads_per_image.append(pad_values)

            params['pads_per_image'] = pads_per_image
            return padded, params
        else:
            raise ValueError(
                'Invalid `method` set, options are `padding` or `crop`.'
            )

    def merge_out(poses_per_image, params):
        # If no merging occur, we have nothing to adjust.
        if not params['merged']:
            return poses_per_image

        if method == 'crop':
            raise NotImplementedError
        elif method == 'padding':
            new_poses_per_image = []
            for poses, pads in zip(poses_per_image, params['pads_per_image']):

                new_poses = []
                for pose in poses:
                    pads_per_axis = np.array([
                        pads[1][0], pads[0][0], 0
                    ]).reshape(1, -1)
                    keypoints = (
                        pose['keypoints'] - pads_per_axis
                    )
                    keypoints[keypoints[..., 2] == 0] = 0

                    new_poses.append({
                        'keypoints': keypoints,
                        'score': pose['score'],
                    })

                new_poses_per_image.append(new_poses)

            return new_poses_per_image
        else:
            raise ValueError(
                'Invalid `method` set, options are `padding` or `crop`.'
            )

    return merge_in, merge_out


class Estimation:

    def __init__(
        self, checkpoint=None, short_side=184, merge_method='padding',
        device=default_device, lazy=False
    ):
        """Initializes and loads the model for `checkpoint`.

        Parameters
        ----------
        checkpoint : str or None
            Checkpoint (and model) to use in order to perform pose estimation.
            If `None`, will use the default one for the task.
        short_side : int
            Resize images' short side to `short_side` before sending over to
            the pose estimation model. Default is `184` to keep the model fast
            enough, though for better results `386` is an appropriate value.
        merge_method : 'padding', 'crop'
            How to merge images together into a batch when receiving a
            list. Merge is done after resizing. Options are:

            * `padding`, which will add padding around images, possibly
              increasing total image size. If mixing portrait and landscape
              images, might be inefficient.

            * `crop`, which will center-crop the images to the smallest
              size. If images are of very different sizes, might end up
              cropping too much.
        device : torch.Device
            Device to load the model on.
        lazy : bool
            If set, will defer model loading until first call.

        """
        self.device = device
        self.estimation_cls = get_class_for_checkpoint(TASK_NAME, checkpoint)

        self.short_side = short_side

        # Load the model into memory unless we have the lazy loading set.
        self.model = (
            self.estimation_cls(
                device=self.device, short_side=self.short_side
            ) if not lazy else None
        )

        self.merge_in, self.merge_out = merge_factory(method=merge_method)

    def __repr__(self):
        return f'<Estimation({self.estimation_cls.__name__})>'

    def __call__(self, images):
        """Performs pose estimation on `images`.

        Derives the actual prediction to the model the `Estimation` object was
        initialized with.

        Parameters
        ----------
        images : list or tuple or np.ndarray

        Returns
        -------
        list
            List of dictionaries containing pose data for a single image, or a
            list of these entries thereof.

        """
        # If `images` is a single `np.ndarray`, turn into a list.
        expanded = False
        if (
            not (isinstance(images, list) or isinstance(images, tuple))
            and len(images.shape) == 3
        ):
            expanded = True
            images = np.expand_dims(images, 0)

        # Run generic image preprocessing, such as merging a list of images
        # into a single numpy array.
        images, merge_params = self.merge_in(images)

        # Call the actual model on the images. If we haven't loaded the model
        # yet due to lazy loading, load it.
        if self.model is None:
            self.model = self.estimation_cls(
                device=self.device, short_side=self.short_side
            )
        out = self.model.call(images)

        # Run the generic image postprocessing.
        out = self.merge_out(out, merge_params)

        return out[0] if expanded else out


pose_estimation = Estimation(lazy=True)
"""Default entry point to pose estimation.

This is an instantiation of the :class:`Estimation` class, lazily-loaded in
order to avoid reading the checkpoints on import. Refer to that class'
:func:`__call__ <Estimation.__call__>` method for more information.
"""
