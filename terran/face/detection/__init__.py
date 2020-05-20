import math
import numpy as np

from cv2 import resize, INTER_LINEAR

from terran import default_device
from terran.checkpoint import get_class_for_checkpoint


TASK_NAME = 'face-detection'


def resize_factory(short_side=416):

    def resize_in(images):
        # If input is a numpy array, we simply modify all images at once with a
        # single scale, returning the result also as an ndarray. If the input,
        # however, is a list, we must resize each image individually, keeping
        # the scales for each.
        if isinstance(images, np.ndarray):
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
                    interpolation=INTER_LINEAR
                )
            scales = scale
        else:
            resized = []
            scales = []
            for image in images:
                H, W = image.shape[0:2]
                scale = short_side / min(H, W)
                new_size = (int(W * scale), int(H * scale))

                resized.append(
                    resize(
                        src=image,
                        dsize=new_size,
                        interpolation=INTER_LINEAR
                    )
                )
                scales.append(scale)

        return resized, scales

    def resize_out(faces_per_image, scales):
        # If we receive a single scale for all images, turn it into a scale per
        # image.
        if not isinstance(scales, list):
            scales = [scales] * len(faces_per_image)

        new_faces_per_image = []
        for faces, scale in zip(faces_per_image, scales):

            new_faces = []
            # TODO: Move rounding to a separate preprocessor. (To also ensure
            # coordinates are valid.)
            for face in faces:
                new_faces.append({
                    'bbox': np.around(
                        face['bbox'] / scale
                    ).astype(np.int32),
                    'landmarks': np.around(
                        face['landmarks'] / scale
                    ).astype(np.int32),
                    'score': face['score'],
                })

            new_faces_per_image.append(new_faces)

        return new_faces_per_image

    return resize_in, resize_out


def merge_factory(method='padding'):
    """Merges a list of images into a single array.

    May do so by padding or cropping images. See `Detection` for more
    information on the options.
    """

    def merge_in(images):
        # Check if not merged already and return early.
        if isinstance(images, np.ndarray):
            return images, {'merged': False}

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

    def merge_out(faces_per_image, params):
        # If no merging occur, we have nothing to adjust.
        if not params['merged']:
            return faces_per_image

        if method == 'crop':
            raise NotImplementedError
        elif method == 'padding':
            new_faces_per_image = []
            for faces, pads in zip(faces_per_image, params['pads_per_image']):

                new_faces = []
                for face in faces:
                    bbox = np.array([
                        face['bbox'][0] - pads[1][0],
                        face['bbox'][1] - pads[0][0],
                        face['bbox'][2] - pads[1][0],
                        face['bbox'][3] - pads[0][0],
                    ])

                    pads_per_axis = np.array([
                        pads[1][0], pads[0][0]
                    ]).reshape(1, -1)
                    landmarks = (
                        face['landmarks'] - pads_per_axis
                    )

                    new_faces.append({
                        'bbox': bbox,
                        'landmarks': landmarks,
                        'score': face['score'],
                    })

                new_faces_per_image.append(new_faces)

            return new_faces_per_image
        else:
            raise ValueError(
                'Invalid `method` set, options are `padding` or `crop`.'
            )

    return merge_in, merge_out


class Detection:

    def __init__(
        self, checkpoint=None, short_side=416, merge_method='padding',
        device=default_device, lazy=False
    ):
        """Initializes and loads the model for `checkpoint`.

        Parameters
        ----------
        checkpoint : str
            Checkpoint (and model) to use in order to perform face detection.
            If `None`, will use the default one for the task.
        short_side : int
            Resize images' short side to `short_side` before sending over to
            detection model.
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
        self.detection_cls = get_class_for_checkpoint(TASK_NAME, checkpoint)

        # Load the model into memory unless we have the lazy loading set.
        self.model = (
            self.detection_cls(device=self.device) if not lazy else None
        )

        # TODO: Allow `max_size`, or by area.
        self.resize_in, self.resize_out = resize_factory(short_side=short_side)
        self.merge_in, self.merge_out = merge_factory(method=merge_method)

    def __repr__(self):
        return f'<Detection({self.detection_cls.__name__})>'

    def __call__(self, images):
        """Performs face detection on `images`.

        Derives the actual prediction to the model the `Detection` object was
        initialized with.

        Parameters
        ----------
        images : list of numpy.ndarray or numpy.ndarray
            Images to perform face detection on.

        Returns
        -------
        list of list of dicts, or list dict
            List of dictionaries containing face data for a single image, or a
            list of these entries thereof.

            Each entry is of the form::

                {
                    'bbox': [x_min, y_min, x_max, y_max],
                    'landmarks': ...,  # Array of shape (5, 2).
                    'score': ... # Confidence score.
                }

        """
        # If `images` is a single `np.ndarray`, turn into a list.
        expanded = False
        if (
            not (isinstance(images, list) or isinstance(images, tuple))
            and len(images.shape) == 3
        ):
            expanded = True
            images = np.expand_dims(images, 0)

        # Run generic image preprocessing, such as resizing images to
        # reasonable sizes.
        images, resize_params = self.resize_in(images)
        images, merge_params = self.merge_in(images)

        # Call the actual model on the images. If we haven't loaded the model
        # yet due to lazy loading, load it.
        if self.model is None:
            self.model = self.detection_cls(device=self.device)
        out = self.model.call(images)

        # Run generic result postprocessing, such as adjusting the coordinates
        # for rescaled images.
        # TODO: Add final validation transformer to check that all bounding
        # boxes are valid (larger than zero dimensions and clipped to image).
        out = self.merge_out(out, merge_params)
        out = self.resize_out(out, resize_params)

        return out[0] if expanded else out


face_detection = Detection(lazy=True)
"""Default entry point to face detection.

This is an instantiation of the :class:`Detection` class, lazily-loaded in
order to avoid reading the checkpoints on import. Refer to that class'
:func:`__call__ <Detection.__call__>` method for more information.
"""
