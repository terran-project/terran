import mxnet as mx
import numpy as np
import os

from PIL import Image

from terran.face.recognition.face_model import FaceModel


os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
FACE_RECOGNITION_PATH = os.environ.get('FACE_RECOGNITION_PATH')
MTCNN_PATH = os.environ.get('MTCNN_PATH')
CTX = mx.gpu() if os.environ.get('MXNET_USE_GPU') else mx.cpu()
MODEL = None
IMAGE_SIZE = 112


def _get_face_recognition_model():
    global MODEL
    if MODEL is None:
        if FACE_RECOGNITION_PATH is None:
            raise ValueError(
                '`FACE_RECOGNITION_PATH` environment variable not set. Point '
                'it to the face recognition checkpoint location.'
            )
        MODEL = FaceModel(
            os.path.expanduser(FACE_RECOGNITION_PATH),
            os.path.expanduser(MTCNN_PATH),
            ctx=CTX,
            threshold=1.24,
            image_size=(IMAGE_SIZE, IMAGE_SIZE),
            det=0,
        )
    return MODEL


def extract_features(faces):
    # If `faces` is a single `np.ndarray`, turn into a list.
    expanded = False
    if (
        not (isinstance(faces, list) or isinstance(faces, tuple))
        and len(faces.shape) == 3
    ):
        expanded = True
        faces = [faces]

    model = _get_face_recognition_model()

    preprocessed = []
    for face in faces:
        curr_preprocessed = model.get_input(face)
        if curr_preprocessed is None:
            # MTCNN preprocessing failed, do it manually: resize image to
            # `IMAGE_SIZExIMAGE_SIZE` and add padding around it.
            face = Image.fromarray(face)

            scale = IMAGE_SIZE / max(face.size[0], face.size[1])
            face = face.resize(
                (int(face.size[0] * scale), int(face.size[1] * scale))
            )

            x_min = int((IMAGE_SIZE - face.size[0]) / 2)
            x_max = int((IMAGE_SIZE - face.size[0]) / 2) + face.size[0]
            y_min = int((IMAGE_SIZE - face.size[1]) / 2)
            y_max = int((IMAGE_SIZE - face.size[1]) / 2) + face.size[1]

            curr_preprocessed = np.zeros(
                (3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8
            )
            curr_preprocessed[:, y_min:y_max, x_min:x_max] = (
                np.asarray(face).transpose([2, 0, 1])[::-1, ...]
            )

        preprocessed.append(curr_preprocessed)

    preps = np.stack(preprocessed, axis=0)
    features = model.get_feature(preps)

    return features[0] if expanded else features
