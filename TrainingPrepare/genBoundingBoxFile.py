# generate bounding boxes for all thermal images
# a cascade model is used for this.
import sys
sys.path.append("../DeepAlignmentNetwork")
import os
import dlib
import numpy as np
from menpo import io as mio
import menpodetect
import pickle
import utils
import shutil

img_dir = "../data/images/thermal_downscaled/"
# old_img_dir = "../data/images/thermal_all"
saved_model = "../data/hog_detector.svm"

face_detector = menpodetect.DlibDetector(dlib.simple_object_detector(saved_model))
# detector = dlib.simple_object_detector(saved_model)
success = 0
fail = 0
dict = {}
for file in os.listdir(img_dir):
    if file.endswith('.png'):
        # img = dlib.load_grayscale_image(os.path.join(img_dir, file))
        # dets = detector(img)
        test_img = mio.import_image(os.path.join(img_dir, file))
        face_bb = face_detector(test_img)
        try:
            # print(dets[1])
            box = face_bb[0].as_vector()
            dict[file] = np.array([box[1], box[0], box[5], box[4]])
            if box[0] < 0 or box[1] < 0 or box[2] < 0 or box[3] < 0:
                pts_file = file[:-3]+'pts'
            success += 1
            # break
            # if success == 100:
            #     break
        except IndexError:
            print("No face detected for image {}, using landmarks for bounding box".format(file))
            landmarks = utils.loadFromPts(os.path.join(img_dir, file[:-3]+"pts"))
            dict[file] = np.array([np.min(landmarks[:,0]), np.min(landmarks[:,1]), np.max(landmarks[:,0]), np.max(landmarks[:,1])])
            fail += 1
with open("boxesThermalDownscaledAll.pkl", "wb") as fp:
    pickle.dump(dict, fp)
print("Detection rate: {}".format(success/(success+fail)))
