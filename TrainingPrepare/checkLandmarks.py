from matplotlib import pyplot as plt
import sys
sys.path.append("../DeepAlignmentNetwork")
import matplotlib.patches as patches
import pickle
import utils
import cv2
import os

img_dir = "../data/images/thermal_downscaled/"
# img_dir = "/home/320077119/Downloads/300W/01_Indoor/"
# img_file= "irface_sub002_seq02_frm00344.jpg_lfb.png"
# img_file = "irface_sub017_seq03_frm00662.jpg_lfb.png"
# img_file = "indoor_001.png"
img_file = "irface_sub055_seq07_frm00992.jpg_lfb.png"
pts_file = img_file[:-3]+'pts'

img = cv2.imread(os.path.join(img_dir, img_file))

landmarks = utils.loadFromPts(os.path.join(img_dir, pts_file))

for landmark in landmarks:
    cv2.circle(img, (int(landmark[0]), int(landmark[1])), 1, (255, 0,  0), -1)

plt.imshow(img)
plt.show()




