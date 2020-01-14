from FaceAlignment import FaceAlignment
import cv2
import utils
from matplotlib import pyplot as plt
from menpo import io as mio
import menpodetect
import numpy as np
import pickle
import dlib
import os
def scaled_detector(img, face_detector):
    for scale in range(2,6):
        new_img = cv2.resize(img, (scale*160, scale*120))
        #                                160        120
        cv2.imwrite("temp.png", new_img)
        mio_img = mio.import_image('temp.png')
        bb = face_detector(mio_img)
        if len(bb) != 0:
            return [bb, scale]
        else:
            print("No face detected at scale: {}".format(scale))
    return [None, None]

model = FaceAlignment(112, 112, 1, 2, False)
# model.loadNetwork("../data2/network_00055_2020-01-10-11-26.npz")
model.loadNetwork("../network_00044_2020-01-10-12-11.npz")
# color_img = cv2.imread("../data/jk.png")
face_detector = menpodetect.DlibDetector(dlib.simple_object_detector("../data/hog_detector.svm"))
# color_img = cv2.imread("../data/images/thermal_detected/irface_sub001_seq02_frm00055.jpg_lfb.png")

# print(gray_img.shape)
# img_file = "../data/images/thermal_detected/irface_sub001_seq02_frm00373.jpg_lfb.png"
img_dir = "../testImages"
out_dir = "../testResults"

for img_file in os.listdir(img_dir):

# if img_file.endswith(".p")
# img_file = "../data/jk.png"
    for extend in [-5, -10, -15]: #range( 50):
        img = mio.import_image(os.path.join(img_dir, img_file)) #"../data/images/thermal_detected/irface_sub001_seq02_frm00055.jpg_lfb.png")
        # reset = True
        landmarks = None

        # if reset:
        # rects = cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=3, minSize=(50, 50))

        face_bb = face_detector(img)
        color_img = cv2.imread(os.path.join(img_dir, img_file)) #"../data/images/thermal_detected/irface_sub001_seq02_frm00055.jpg_lfb.png")
        if len(color_img.shape) > 2:
            gray_img = np.mean(color_img, axis=2).astype(np.uint8)
        else:
            gray_img = color_img.astype(np.uint8)
        # gray_img = gray_img.astype(np.uint8)
        print(gray_img.shape)
        # for rect in rects:
        if len(face_bb) == 0:
            [face_bb, scale] = scaled_detector(gray_img, face_detector)
            if face_bb is None:
                continue
        else:
            scale = 1
        # print(len(face_bb))
        # for bb in face_bb:
        for _ in range(1):
            bb = face_bb[0]
            # tl_x = rect[0]
            # tl_y = rect[1]
            # br_x = tl_x + rect[2]
            # br_y = tl_y + rect[3]
            box = bb.as_vector()
            # extend = -10
            tl_x = int(box[1]/scale) - extend
            tl_y = int(box[0]/scale) - extend
            br_x = int(box[5]/scale) + extend
            br_y = int(box[4]/scale) + extend
            print("tl_x:{} tl_y:{} br_x:{} br_y:{}".format(tl_x, tl_y, br_x, br_y))
            cv2.rectangle(gray_img, (tl_x, tl_y), (br_x, br_y), (255, 0, 0))

            initLandmarks = utils.bestFitRect(None, model.initLandmarks, [tl_x, tl_y, br_x, br_y])

            if model.confidenceLayer:
                landmarks, confidence = model.processImg(gray_img[np.newaxis], initLandmarks)
                if confidence < 0.1:
                    reset = True
            else:
                landmarks = model.processImg(gray_img[np.newaxis], initLandmarks)

            landmarks = landmarks.astype(np.int32)
            for i in range(landmarks.shape[0]):
                # if i >= 27 and i <= 35:
                #     cv2.circle(gray_img, (landmarks[i, 0], landmarks[i, 1]), 3, (255, 255, 0), -1)
                # else:
                print("{} {}".format(landmarks[i,0], landmarks[i, 1]))
                cv2.circle(gray_img, (landmarks[i, 0], landmarks[i, 1]), 1, (0, 255, 0), -1)
        plt.imshow(gray_img, cmap='gray')
        plt.savefig(os.path.join(out_dir, str(extend) + "_" + img_file), dpi='figure', bbox_inches='tight')
        plt.clf()
        plt.close()
    #cv2.imshow("image", color_img)

#key = cv2.waitKey(0)
