import cv2
import numpy as np
import os
import torch

from PIL import Image
from sklearn.preprocessing import normalize
from skimage.transform import SimilarityTransform

from terran import default_device
from terran.face.recognition.arcface import FaceResNet100


def load_model():
    model = FaceResNet100()
    model.load_state_dict(torch.load(
        os.path.expanduser('~/.terran/checkpoints/arcface-resnet100.pth')
    ))
    model.eval()
    return model


def preprocess_face(
    img, bbox=None, landmark=None, image_size=(112, 112), margin=44
):
    """Preprocess an image by aligning the face contained in them."""
    M = None

    if landmark is not None:
        # Target location of the facial landmarks.
        src = np.array(
          [
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]
          ],
          dtype=np.float32
        )

        if image_size[1] == 112:
            src[:, 0] += 8.0

        dst = landmark.astype(np.float32)

        tform = SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2, :]

    if M is None:
        if bbox is None:  # Use center crop.
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1]*0.0625)
            det[1] = int(img.shape[0]*0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox

        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
        bb[3] = np.minimum(det[3]+margin/2, img.shape[0])

        ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
        ret = cv2.resize(ret, (image_size[1], image_size[0]))

        return ret
    else:
        # Do align using landmark.
        warped = cv2.warpAffine(
          img, M, (image_size[1], image_size[0]), borderValue=0.0
        )
        return warped


class ArcFace:

    def __init__(self, device=default_device, threshold=1.24, image_side=112):
        self.device = device
        self.model = load_model().to(self.device)

        self.det_threshold = [0.6, 0.7, 0.8]
        self.image_side = image_side
        self.threshold = threshold

    def get_input(self, image, face):
        """Prepares the face image for the recognition model.

        Uses the detected landmarks from the face detection stage, then aligns
        the image, pads to `image_side`x`image_side`, and turns it into the BGR
        CxHxW format.

        TODO: If no face.

        Parameters
        ----------
        image : np.ndarray of size HxWxC.

        """
        bbox = face['bbox']
        points = face['landmarks']

        # TODO: Move all this to `preprocess_face` and remove this function.
        processed = preprocess_face(image, bbox, points)
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(processed, (2, 0, 1))

        return aligned

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
                    preprocessed.append(self.get_input(image, face))

            # A bit of magic, but gets us the index within `preprocessed` where
            # faces of each image start.
            splits = np.cumsum(list(map(len, faces_per_image)))[:-1]
        else:
            # No landmarks provided, so preprocess it manually: resize
            # image to `image_size` and add padding around it.
            for image in images:
                face = Image.fromarray(image)

                scale = self.image_side / max(face.size[0], face.size[1])
                face = face.resize(
                    (int(face.size[0] * scale), int(face.size[1] * scale))
                )

                x_min = int((self.image_side - face.size[0]) / 2)
                x_max = int(
                    (self.image_side - face.size[0]) / 2
                ) + face.size[0]
                y_min = int((self.image_side - face.size[1]) / 2)
                y_max = int(
                    (self.image_side - face.size[1]) / 2
                ) + face.size[1]

                curr_preprocessed = np.zeros(
                    (3, self.image_side, self.image_side), dtype=np.uint8
                )
                curr_preprocessed[:, y_min:y_max, x_min:x_max] = (
                    np.asarray(face).transpose([2, 0, 1])[::-1, ...]
                )

                preprocessed.append(curr_preprocessed)

            # No splits to perform under this scenario, we pack them up as if
            # they were all faces for a single image.
            splits = []

        # No faces received, return early.
        if not preprocessed:
            # TODO: Embedding output depends on the model.
            if faces_per_image is not None:
                return np.empty((0, 512))
            else:
                return [
                    np.empty((0, 512)) for _ in images
                ]

        preprocessed = np.stack(preprocessed, axis=0)

        # Now turn the (already preprocessed) input into a `torch.Tensor` and
        # feed through the network.
        data = torch.tensor(
            preprocessed, device=self.device, dtype=torch.float32
        )
        features = self.model(data).detach().to('cpu').numpy()
        features = normalize(features, axis=1)

        features_per_image = np.split(features, splits, axis=0)

        if faces_per_image is None:
            # If no per-image separation, just return the features.
            features_per_image = features_per_image[0]

        return features_per_image
