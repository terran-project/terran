import cv2
import numpy as np
import os
import torch

from sklearn.preprocessing import normalize
from skimage.transform import SimilarityTransform

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


class FaceModel:

    def __init__(self, threshold=1.24, image_size=(112, 112)):
        self.model = load_model().to(torch.device('cuda:0'))

        self.det_threshold = [0.6, 0.7, 0.8]
        self.image_size = image_size
        self.threshold = threshold

    def get_input(self, image, face):
        """Prepares the face image for the recognition model.

        Uses the detected landmarks from the face detection stage, then aligns
        the image, pads to 112x112, and turns it into the BGR CxHxW format.

        TODO: If no face.

        Parameters
        ----------
        image : np.ndarray of size HxWxC.

        """
        bbox = face['bbox']
        points = face['landmarks']

        processed = preprocess_face(image, bbox, points)
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(processed, (2, 0, 1))

        return aligned

    def get_feature(self, images):
        expanded = False
        if len(images.shape) == 3:
            expanded = True
            images = np.expand_dims(images, axis=0)

        # Turn the (already preprocessed) input into a `torch.Tensor` and feed
        # through the network.
        data = torch.tensor(
            images, device=torch.device('cuda:0'), dtype=torch.float32
        )
        features = self.model(data).detach().to('cpu').numpy()
        features = normalize(features, axis=1)

        if expanded:
            features = features.flatten()

        return features
