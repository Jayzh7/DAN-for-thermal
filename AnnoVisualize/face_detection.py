import os
import menpodetect
from menpo import io as mio
import numpy as np
import dlib
from matplotlib import pyplot as plt
import cv2
img_dir = "../data/images/thermal_detected"
old_img_dir = "../data/images/thermal_all"
saved_model = "../data/hog_detector.svm"

face_detector = menpodetect.DlibDetector(dlib.simple_object_detector(saved_model))

nr = 0
for file in os.listdir(img_dir):
    if file.endswith('.png'):
        # img = dlib.load_grayscale_image(os.path.join(img_dir, file))
        # dets = detector(img)
        test_img = mio.import_image(os.path.join(img_dir, file))
        face_bb = face_detector(test_img)
        try:
            # print(dets[1])
            # print(test_img)
            box = face_bb[0].as_vector()
            # dict[file] = np.array([box[0], box[1], box[4], box[5]])
            if box[0] < 0 or box[1] < 0 or box[2] < 0 or box[3] < 0:
                pts_file = file[:-3]+'pts'
                print("File {} no face detected ".format(file))
                print(box)
            else:
                img = cv2.imread(os.path.join(img_dir, file))
                print(box)
                cv2.rectangle(img, ((int)(box[5]), (int)(box[4])), ((int)(box[1]), (int)(box[0])), (255, 0, 0))
                cv2.imwrite(file.split('.')[0]+'.png', img)
                nr += 1
            if nr == 100:
                break
            # success += 1
            # break
            # if success == 100:
            #     break
        except IndexError:
            print("No face detected!")
            fail += 1
