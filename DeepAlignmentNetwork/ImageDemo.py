from FaceAlignment import FaceAlignment
import numpy as np
import cv2
import utils
from matplotlib import pyplot as plt
from menpo import io as mio
import menpodetect
import pickle
import dlib
import os
def scaled_detector(img, face_detector):
    for scale in range(2,6):
        new_img = cv2.resize(img, (scale*160, scale*120))
        cv2.imwrite("temp.png", new_img)
        img = mio.import_image('temp.png')
        bb = face_detector(img)
        if bb is not None:
            return [bb, scale]
    return [[[100, 99, 112, 124]], 1]
# model.loadNetwork("../data2/network_00042_2020-01-10-05-36.npz")
# color_img = cv2.imread("../data/jk.png")
face_detector = menpodetect.DlibDetector(dlib.simple_object_detector("../data/hog_detector.svm"))
# color_img = cv2.imread("../data/images/thermal_detected/irface_sub001_seq02_frm00055.jpg_lfb.png")

# print(gray_img.shape)
img_file = "../testImages/s8_1.jpg"
out_dir = "../testResults"

# img_file = "../data/jk.png"
img = mio.import_image(img_file) #"../data/images/thermal_detected/irface_sub001_seq02_frm00055.jpg_lfb.png")
# reset = True
landmarks = None

# if reset:
# rects = cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=3, minSize=(50, 50))

face_bb = face_detector(img)
color_img = cv2.imread(img_file) #"../data/images/thermal_detected/irface_sub001_seq02_frm00055.jpg_lfb.png")
if len(color_img.shape) > 2:
    gray_img = np.mean(color_img, axis=2).astype(np.uint8)
else:
    gray_img = color_img.astype(np.uint8)
# gray_img = gray_img.astype(np.uint8)
print(gray_img.shape)
# for rect in rects:
scale = 1
if face_bb is None:
    [face_bb, scale] = scaled_detector(gray_img, face_detector)
if face_bb is None:
    print("None")
else:
    face_bb = [[95, 51, 135, 121]]
    print(face_bb)
    model = FaceAlignment(112, 112, 1, 2, False)
    model.loadNetwork("../data2/networks/network_00061_2020-01-10-14-00.npz")
for bb in face_bb:
    tl_x = bb[0]
    tl_y = bb[1]
    br_x = bb[0]+bb[2]
    br_y = bb[1]+bb[3]
    print("tl_x:{} tl_y:{} br_x:{} br_y:{}".format(tl_x, tl_y, br_x, br_y))

    initLandmarks = utils.bestFitRect(None, model.initLandmarks, [tl_x, tl_y, br_x, br_y])

    if model.confidenceLayer:
        landmarks, confidence = model.processImg(gray_img[np.newaxis], initLandmarks)
        if confidence < 0.1:
            reset = True
    else:
        landmarks = model.processImg(gray_img[np.newaxis], initLandmarks)

    cv2.rectangle(gray_img, (tl_x, tl_y), (br_x, br_y), (255, 0, 0))
    landmarks = landmarks.astype(np.int32)
    for i in range(landmarks.shape[0]):
        cv2.circle(gray_img, (landmarks[i, 0], landmarks[i, 1]), 3, 1, -1)
        print("{}\t{}".format(landmarks[i][0], landmarks[i][1]))

    plt.imshow(gray_img, cmap='gray')
    plt.savefig(os.path.join(out_dir, img_file), dpi='figure', bbox_inches='tight')
    plt.clf()
    plt.close()
