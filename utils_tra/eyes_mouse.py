import torchlm
from torchlm.tools import faceboxesv2
from torchlm.models import pipnet
import cv2

def get_eye_landmarks(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    torchlm.runtime.bind(
        faceboxesv2(device="cpu")
    )  # set device="cuda" if you want to run with CUDA
    # set map_location="cuda" if you want to run with CUDA
    torchlm.runtime.bind(
        pipnet(
            backbone="resnet101",
            pretrained=True,
            num_nb=10,
            num_lms=98, #98
            net_stride=32,
            input_size=256,
            meanface_type="wflw",
            map_location="cpu",
            checkpoint=None,
        )
    ) # will auto download pretrained weights from latest release if pretrained=True

    landmarks, bboxes = torchlm.runtime.forward(image)
    # landmarks = landmarks[:,60:61]         # 左眼角
    landmarks = landmarks[:,72:73]         # 右眼角
    landmarks = landmarks.squeeze(1)

    return landmarks

def get_mouse_landmarks(image):
    torchlm.runtime.bind(
        faceboxesv2(device="cpu")
    )  # set device="cuda" if you want to run with CUDA
    # set map_location="cuda" if you want to run with CUDA
    torchlm.runtime.bind(
        pipnet(
            backbone="resnet101",
            pretrained=True,
            num_nb=10,
            num_lms=98, #98
            net_stride=32,
            input_size=256,
            meanface_type="wflw",
            map_location="cpu",
            checkpoint=None,
        )
    ) # will auto download pretrained weights from latest release if pretrained=True

    landmarks, bboxes = torchlm.runtime.forward(image)
    landmarks = landmarks[:,76:77]         # 左嘴角
    # landmarks = landmarks[:,82:83]         # 右嘴角
    landmarks = landmarks.squeeze(1)

    return landmarks

