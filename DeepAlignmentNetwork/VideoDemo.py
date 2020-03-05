from FaceAlignment import FaceAlignment
# import tifffile as tiff
import cv2
import utils
from matplotlib import pyplot as plt
from menpo import io as mio
import menpodetect
import numpy as np
import dlib
import os


def scaled_detector(img, face_detector):
    for scale in range(2, 6):
        new_img = cv2.resize(img, (scale * 160, scale * 120))
        cv2.imwrite("temp.png", new_img)
        img = mio.import_image('temp.png')
        bb = face_detector(img)
        if bb is not None and len(bb) != 0:
            return [bb, scale]

    print("No face detected at any scale!")
    return [None, None]


video = cv2.VideoCapture("../data/newfile.avi")
# video = tiff.imread("../data/frontal_view_front_angle.tiff")
model = FaceAlignment(112, 112, 1, 2, False)
# model.loadNetwork("../networks/network_00045_2020-01-10-12-14.npz")
model.loadNetwork("../data2/network_00055_2020-01-10-11-26.npz")

face_detector = menpodetect.DlibDetector(dlib.simple_object_detector("../data/hog_detector.svm"))
out = cv2.VideoWriter("output2.avi", cv2.VideoWriter_fourcc(*'XVID'), 9, (160, 120), 0)
face_bb = None

ret, img = video.read()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
if face_bb is None:
    cv2.imwrite("temp.png", img)
    img2 = mio.import_image('temp.png')
    face_bb = face_detector(img2)

try:
    bb = face_bb[0]
except IndexError:
    [bb, scale] = scaled_detector(img, face_detector)
    if bb is None or len(bb) == 0:
        print("no face detected")
    else:
        # print("face detectd at scale {}!".format(scale))
        bb = bb[0]
    # continue
box = bb.as_vector()
# scale = scale*scale
tl_x = int(box[1] / scale)
tl_y = int(box[0] / scale)
br_x = int(box[5] / scale)
br_y = int(box[4] / scale)

landmarks = None

while ret is True:
    scale = 1
    # print(img2.shape)

    if landmarks is not None:
        landmarks = model.processImg(img[np.newaxis], landmarks)
    else:
        cv2.rectangle(img, (tl_x, tl_y), (br_x, br_y), (255, 0, 0))
        landmarks = utils.bestFitRect(None, model.initLandmarks, [tl_x, tl_y, br_x, br_y])
        landmarks = model.processImg(img[np.newaxis], landmarks)

    landmarks = landmarks.astype(np.int32)
    for i in range(landmarks.shape[0]):
        cv2.circle(img, (landmarks[i, 0], landmarks[i, 1]), 1, (0, 255, 0))
    out.write(img)
    ret, img = video.read()
out.release()


