import numpy as np

from terran import default_device
from terran.pose.openpose import OpenPose


POSE_ESTIMATION_MODELS = {
    # Aliases.
    # TODO: Tentative names.
    'gpu-accurate': None,
    'gpu-realtime': OpenPose,
    'cpu-realtime': None,
    'edge-realtime': None,

    # Models.
    'openpose': OpenPose,
}


class Estimation:

    def __init__(
        self, checkpoint='gpu-realtime', short_side=184,
        merge_method='padding', device=default_device, lazy=False
    ):
        """Initializes and loads the model for `checkpoint`.

        Parameters
        ----------
        checkpoint : str
            Checkpoint (and model) to use in order to perform pose estimation.
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

        if checkpoint not in POSE_ESTIMATION_MODELS:
            raise ValueError(
                'Checkpoint not found, is it one of '
                '`terran.pose.POSE_ESTIMATION_MODELS`?'
            )
        self.checkpoint = checkpoint
        self.estimation_cls = POSE_ESTIMATION_MODELS[self.checkpoint]

        self.short_side = short_side

        # Load the model into memory unless we have the lazy loading set.
        self.model = (
            self.estimation_cls(
                device=self.device, short_side=self.short_side
            ) if not lazy else None
        )

    def __call__(self, images):
        """Performs pose estimation on `images`.

        Derives the actual prediction to the model the `Estimation` object was
        initialized with.

        Arguments
            images (list or tuple or np.ndarray).

        Returns
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

        # Call the actual model on the images. If we haven't loaded the model
        # yet due to lazy loading, load it.
        if self.model is None:
            self.model = self.estimation_cls(
                device=self.device, short_side=self.short_side
            )
        out = self.model.call(images)

        # return out[0] if expanded else out
        return out if expanded else out


pose_estimation = Estimation(lazy=True)
