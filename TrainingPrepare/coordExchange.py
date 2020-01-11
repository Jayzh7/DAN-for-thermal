# exchange X, Y coordinates

import os
import sys
sys.path.append("../DeepAlignmentNetwork")
import utils
img_dir = "../data/images/thermal_all/"

for file in os.listdir(img_dir):
    if file.endswith('.pts'):
        pts = utils.loadFromPts(os.path.join(img_dir, file))
        for pt in pts:
            pt[0], pt[1] = pt[1], pt[0]

        with open(os.path.join(img_dir, file), "w") as fp:
            utils.saveToPts(os.path.join(img_dir, file), pts)

        # print(file)
        # break

