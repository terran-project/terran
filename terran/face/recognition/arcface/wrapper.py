import numpy as np
import torch

from PIL import Image
from sklearn.preprocessing import normalize
from skimage.transform import SimilarityTransform

from terran import default_device
from terran.checkpoint import get_checkpoint_path
from terran.face.recognition.arcface.model import FaceResNet100


def load_model():
    model = FaceResNet100()
    model.load_state_dict(torch.load(
        get_checkpoint_path('terran.face.recognition.arcface.ArcFace')
    ))
    model.eval()
    return model


def preprocess_face(image, landmark, image_size=(112, 112)):
    """Prepares the face image for the recognition model.

    Uses the detected landmarks from the face detection stage, then aligns
    the image, pads to `image_side`x`image_side`, and turns it into the BGR
    CxHxW format.

    Parameters
    ----------
    image : np.ndarray of size HxWxC.
        Image containing a face to preprocess.
    landmark : np.ndarray of size (5, 2).
        Landmark coordinates.

    """

    # Target location of the facial landmarks.
    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]
    ], dtype=np.float32)

    if image_size[1] == 112:
        src[:, 0] += 8.0

    dst = landmark.astype(np.float32)

    t_form = SimilarityTransform()
    t_form.estimate(dst, src)

    #
    # Do align using landmark.
    #
    # The Image.transform method requires the inverted transformation matrix,
    # without the last row, and flattened
    #
    t_matrix = np.linalg.inv(t_form.params)[0: -1, :].flatten()

    warped = Image.fromarray(image).transform(
        size=(image_size[1], image_size[0]),
        method=Image.AFFINE,
        data=t_matrix,
        resample=Image.BILINEAR,
        fillcolor=0,
    )

    warped = np.array(warped)
    return warped.transpose([2, 0, 1])[::-1, ...]


def preprocess_face_no_landmarks(image, image_side=112):
    """Preprocess a face without landmarks.

    Resize image to have side `image_side` and add padding around it.
    """
    face = Image.fromarray(image)

    scale = image_side / max(face.size[0], face.size[1])
    face = face.resize(
        (int(face.size[0] * scale), int(face.size[1] * scale))
    )

    x_min = int((image_side - face.size[0]) / 2)
    x_max = int((image_side - face.size[0]) / 2) + face.size[0]
    y_min = int((image_side - face.size[1]) / 2)
    y_max = int((image_side - face.size[1]) / 2) + face.size[1]

    preprocessed = np.zeros(
        (3, image_side, image_side), dtype=np.uint8
    )
    preprocessed[:, y_min:y_max, x_min:x_max] = (
        np.asarray(face).transpose([2, 0, 1])[::-1, ...]
    )

    return preprocessed


class ArcFace:

    def __init__(self, device=default_device, image_side=112):
        self.device = device
        self.model = load_model().to(self.device)
        self.image_side = image_side

    def call(self, images, faces_per_image=None):
        """Performs feature extraction on `images`.

        Can either receive already-cropped images for faces (when
        `faces_per_image` is `None`), or a list of images and the face
        coordinates and landmarks to crop, as returned by a `Detection`
        function. The latter is preferable, as a more precise preprocessing can
        be performed.

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
        # Since every image will have a small number of faces, we want to
        # unroll the features so we can use larger batch sizes with the model,
        # and then re-arrange them in the original structure again. For that,
        # we keep track of the indices with which to split in `splits`.
        preprocessed = []
        if faces_per_image is not None:
            for image, faces in zip(images, faces_per_image):
                for face in faces:
                    preprocessed.append(
                        preprocess_face(image, face['landmarks'])
                    )

            # A bit of magic, but gets us the index within `preprocessed` where
            # faces of each image start.
            splits = np.cumsum(list(map(len, faces_per_image)))[:-1]
        else:
            # No landmarks provided, so preprocess it manually.
            for image in images:
                preprocessed.append(
                    preprocess_face_no_landmarks(image, self.image_side)
                )

            # No splits to perform under this scenario, we pack them up as if
            # they were all faces for a single image.
            splits = []

        # No faces received, return early.
        if not preprocessed:
            # TODO: Embedding output depends on the model.
            return [
                np.empty((0, 512)) for _ in images
            ]

        preprocessed = np.stack(preprocessed, axis=0)

        # Now turn the (already preprocessed) input into a `torch.Tensor` and
        # feed through the network.
        data = torch.tensor(
            preprocessed, device=self.device, dtype=torch.float32
        )
        with torch.no_grad():
            features = self.model(data).cpu().numpy()

        features = normalize(features, axis=1)

        features_per_image = np.split(features, splits, axis=0)

        if faces_per_image is None:
            # If no per-image separation, just return the features.
            features_per_image = features_per_image[0]

        return features_per_image
