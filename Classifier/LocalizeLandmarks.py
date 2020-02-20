import sys
sys.path.append("../DeepAlignmentNetwork/")
from FaceAlignment import FaceAlignment
import numpy as np
import cv2
import utils
import DetectFace
from matplotlib import pyplot as plt

model_file = "../data2/network_00060_2020-01-10-13-46.npz"
img_file = "../data/images/thermal_downscaled/irface_sub001_seq02_frm00055.jpg_lfb.png"

def load_model(model_f = model_file):
    model = FaceAlignment(112, 112, 1, 2, False)
    model.loadNetwork(model_f)

    return model
def localize(img_file, model = load_model()):
    if type(img_file) is not str:
        cv2.imwrite("img.png", img_file)
        img_file = "img.png"

    face_rect = DetectFace.detect(img_file)
    initLandmarks = utils.bestFitRect(None, model.initLandmarks, face_rect)
    img = cv2.imread(img_file)
    if len(img.shape) > 2:
        img = np.mean(img, axis=2).astype(np.uint8)
    else:
        img = img.astype(np.uint8)


    landmarks = model.processImg(img[np.newaxis], initLandmarks)

    return landmarks

if __name__ == '__main__':
    landmarks = localize(img_file)
    img = cv2.imread(img_file)
    for i in range(landmarks.shape[0]):
        cv2.circle(img, (int(landmarks[i,0]), int(landmarks[i, 1])), 1, (0, 255, 0), -1)

    plt.imshow(img)
    plt.show()


