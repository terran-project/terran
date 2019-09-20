import numpy as np

from terran import default_device
from terran.face.recognition.arcface_wrapper import ArcFace


FACE_RECOGNITION_MODELS = {
    # Aliases.
    # TODO: Tentative names.
    'gpu-accurate': None,
    'gpu-realtime': ArcFace,
    'cpu-realtime': None,
    'edge-realtime': None,

    # Models.
    'arcface-resnet100': ArcFace,
}


class Recognition:

    def __init__(
        self, checkpoint='gpu-realtime', device=default_device, lazy=False,
    ):
        """Initializes and loads the model for `checkpoint`.

        Parameters
        ----------
        checkpoint : str
            Checkpoint (and model) to use in order to perform face recognition.
        device : torch.Device
            Device to load the model on.
        lazy : bool
            If set, will defer model loading until first call.

        """
        self.device = device

        if checkpoint not in FACE_RECOGNITION_MODELS:
            raise ValueError(
                'Checkpoint not found, is it one of '
                '`terran.face.recognition.FACE_RECOGNITION_MODELS`?'
            )
        self.checkpoint = checkpoint
        self.recognition_cls = FACE_RECOGNITION_MODELS[self.checkpoint]

        # Load the model into memory unless we have the lazy loading set.
        self.model = (
            self.recognition_cls(device=self.device) if not lazy else None
        )

    def __call__(self, images, faces_per_image=None):
        """Performs face recognition on `images`.

        Derives the actual prediction to the model the `Recognition` object was
        initialized with.

        Parameters
        ----------
        images : list or tuple or np.ndarray
            Images
        faces_per_image : list

        Returns
        -------
        List of...

        """
        # Make sure that `images` and `faces_per_image`, if present, match
        # their ranks.
        if faces_per_image is not None:
            # TODO: Do.
            pass

        # If `images` is a single `np.ndarray`, turn into a list.
        expanded = False
        if (
            not (isinstance(images, list) or isinstance(images, tuple))
            and len(images.shape) == 3
        ):
            expanded = True
            # TODO: Not like this; might be list.
            images = np.expand_dims(images, 0)

            # TODO: Also `faces_per_image`. See exactly what we want to
            # support.
            if isinstance(faces_per_image, dict):
                faces_per_image = [[faces_per_image]]
            else:
                faces_per_image = [faces_per_image]

        if self.model is None:
            self.model = self.recognition_cls(device=self.device)
        out = self.model.call(images, faces_per_image)

        # TODO: Don't collapse back first dimension if we flattened the
        # results.
        # return out[0] if expanded else out
        return out


extract_features = Recognition(lazy=True)
