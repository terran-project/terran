import cv2
import mxnet as mx
import numpy as np

from sklearn.preprocessing import normalize
from skimage.transform import SimilarityTransform

from terran.face.recognition.mtcnn_detector import MtcnnDetector


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


def get_model(model_path, ctx, image_size, layer, batch_size=1):
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_path, 0)
    all_layers = sym.get_internals()
    sym = all_layers[f'{layer}_output']
    model = mx.mod.Module(
        symbol=sym,
        context=ctx,
        label_names=None
    )

    model.bind(
        data_shapes=[
            ('data', (batch_size, 3, image_size[0], image_size[1]))
        ],
        # label_shapes=[
        #     ('softmax_label', (batch_size,))
        # ]
    )

    # model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)

    return model


class FaceModel:

    def __init__(
        self, model_path, mtcnn_path, ctx=mx.gpu(),
        threshold=1.24, image_size=(112, 112), det=0,
    ):
        self.model = get_model(model_path, ctx, image_size, 'fc1')

        self.det_threshold = [0.6, 0.7, 0.8]
        self.image_size = image_size
        self.threshold = threshold

        self.det_type = det
        # TODO: Check difference.
        if det == 0:
            self.detector = MtcnnDetector(
                model_folder=mtcnn_path,
                ctx=ctx,
                num_worker=1,
                accurate_landmark=True,
                threshold=self.det_threshold,
            )
        else:
            self.detector = MtcnnDetector(
                model_folder=mtcnn_path,
                ctx=ctx,
                num_worker=1,
                accurate_landmark=True,
                threshold=[0.0, 0.0, 0.2]
            )

    def get_input(self, image):
        """Prepares the face image for the recognition model.

        First uses MTCNN to obtain the facial landmarks, then aligns the image,
        pads to 112x112, and turns it into the BGR CxHxW format.

        Parameters
        ----------
        image : np.ndarray of size HxWxC.

        """
        result = self.detector.detect_face(
            image, det_type=self.det_type
        )
        if result is None:
            return

        bbox, points = result
        if bbox.shape[0] == 0:
            return

        # Consider only the first (TODO: most confident?) detection.
        bbox = bbox[0, 0:4]
        points = points[0, :].reshape((2, 5)).T

        processed = preprocess_face(image, bbox, points)
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(processed, (2, 0, 1))

        return aligned

    def get_feature(self, image):
        image = np.expand_dims(image, axis=0)

        self.model.forward(
            mx.io.DataBatch(data=(
                mx.nd.array(image),
            )),
            is_train=False
        )

        embedding = self.model.get_outputs()[0].asnumpy()
        embedding = normalize(embedding).flatten()

        return embedding
