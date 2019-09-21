from terran import default_device
from terran.face.recognition.arcface import ArcFace


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
        images : list of numpy.ndarray or numpy.ndarray
            Images to perform face recognition on.
        faces_per_image : list of list of dicts.
            Each dict entry must contain `bbox` and `landmarks` keys, as
            returned by a `Detection` instance.

        Returns
        -------
        list of numpy.ndarray or numpy.ndarray
            One entry per image, with a numpy array of size (N_i, F), with F
            being the embedding size returned by the model.

        """
        # If `images` is a single image, turn into a list.
        expanded = False
        if (
            not (isinstance(images, list) or isinstance(images, tuple))
            and len(images.shape) == 3
        ):
            expanded = True
            images = [images]

            # Also expand `faces_per_image`. If only received a `dict`, must
            # expand it twice.
            if isinstance(faces_per_image, dict):
                faces_per_image = [[faces_per_image]]
            else:
                faces_per_image = [faces_per_image]

        # Make sure that `images` and `faces_per_image`, if present, match
        # their ranks.
        if faces_per_image is not None and len(faces_per_image) != len(images):
            raise ValueError(
                f'`images` and `faces_per_image` must be of the same size, '
                f'but the former is of size {len(images)} while the latter of '
                f'size {len(faces_per_image)}.'
            )

        if self.model is None:
            self.model = self.recognition_cls(device=self.device)
        out = self.model.call(images, faces_per_image)

        # What we return depends on how much we expanded.
        if expanded and isinstance(faces_per_image, dict):
            return out[0][0]
        elif expanded:
            return out[0]
        else:
            return out


extract_features = Recognition(lazy=True)
