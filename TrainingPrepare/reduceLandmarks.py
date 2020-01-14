import sys

sys.path.append("../DeepAlignmentNetwork")
import numpy as np
import utils
import os

lm_dir = "../data/images/landmarks"
            #---------------nose--------------#-orbits-#-mouth-#
nr_to_add = [27, 28, 29, 30, 31, 32, 33, 34, 35, 39, 42, 48, 54]
# idx = 0
for file in os.listdir(lm_dir):
    landmarks = utils.loadFromPts(os.path.join(lm_dir, file))
    # for landmark in landmarks:
    #     if idx in nr_to_add:
    #         print("{}: {}".format(idx, landmark))
    #     idx = idx + 1
    # print("--------")
    newLandmarks = np.array(landmarks[nr_to_add[0]])
    for i in range(1,len(nr_to_add)):
        newLandmarks = np.vstack((newLandmarks, landmarks[nr_to_add[i]]))
    # print(newLandmarks)
    utils.saveToPts("../data/images/thermal_downscaled/"+file, newLandmarks)


