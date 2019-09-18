import numpy as np

from PIL import Image

from terran import default_device
from terran.face.detection.retinaface_wrapper import RetinaFace


FACE_DETECTION_MODELS = {
    # Aliases.
    # TODO: Tentative names.
    'gpu-accurate': None,
    'gpu-realtime': RetinaFace,
    'cpu-realtime': None,
    'edge-realtime': None,

    # Models.
    'retinaface-mnet': RetinaFace,
}


def resize_factory(short_side=416):

    def resize_in(images):

        H, W = images.shape[1:3]
        scale = short_side / min(H, W)

        new_size = (
            int(W * scale), int(H * scale)
        )

        resized = np.stack([
            np.asarray(
                Image.fromarray(image).resize(
                    new_size, resample=Image.BILINEAR
                )
            ) for image in images
        ], axis=0)

        return resized, scale

    def resize_out(faces_per_image, scale):
        new_faces_per_image = []
        for faces in faces_per_image:

            new_faces = []
            for face in faces:
                new_faces.append({
                    'bbox': (face['bbox'] / scale).astype(np.int32),
                    'landmarks': (face['landmarks'] / scale).astype(np.int32),
                    'score': face['score'],
                })

            new_faces_per_image.append(new_faces)

        return new_faces_per_image

    return resize_in, resize_out


class Detection:

    def __init__(
        self, checkpoint='gpu-realtime', short_side=416, device=default_device,
        lazy=False
    ):
        """Initializes and loads the model for `checkpoint`.

        Parameters
        ----------
        checkpoint : str
            Checkpoint (and model) to use in order to perform face detection.
        short_side : int
            Resize images' short side to `short_side` before sending over to
            detection model.
        device : torch.Device
            Device to load the model on.
        lazy : bool
            If set, will defer model loading until first call.

        """
        self.device = device

        if checkpoint not in FACE_DETECTION_MODELS:
            raise ValueError(
                'Checkpoint not found, is it one of '
                '`terran.face.detection.FACE_DETECTION_MODELS`?'
            )
        self.checkpoint = checkpoint
        self.detection_cls = FACE_DETECTION_MODELS[self.checkpoint]

        # Load the model into memory unless we have the lazy loading set.
        self.model = (
            self.detection_cls(device=self.device) if not lazy else None
        )

        # TODO: Allow `max_size`, or by area.
        self.resize_in, self.resize_out = resize_factory(short_side=short_side)

    def __call__(self, images):
        """Performs face detection on `images`.

        Derives the actual prediction to the model the `Detection` object was
        initialized with.

        Arguments
            images (list or tuple or np.ndarray).

        Returns
            List of dictionaries containing face data for a single image, or a
            list of these entries thereof.

        """
        # TODO: Should we even accept lists? In said case, we need to pad and
        # maybe perform intelligent batching.

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
        images, scale = self.resize_in(images)
        # TODO: Move tansforming to tensor to the outside too?

        # Call the actual model on the images. If we haven't loaded the model
        # yet due to lazy loading, load it.
        if self.model is None:
            self.model = self.detection_cls(device=self.device)
        out = self.model.call(images)

        # Run generic result postprocessing, such as adjusting the coordinates
        # for rescaled images.
        out = self.resize_out(out, scale)

        return out[0] if expanded else out


# Instantiate default detector for cleaner API.
face_detection = Detection(lazy=True)
