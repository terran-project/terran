import torch

from terran.face.detection.retinaface import RetinaFace as RetinaFaceModel


def load_model(path):
    model = RetinaFaceModel()
    model.load_state_dict(torch.load(
        '/home/agustin/dev/terran/checkpoints/retinaface-mnet.pth'
    ))
    model.eval()
    return model
