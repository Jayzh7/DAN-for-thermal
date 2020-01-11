import sys
sys.path.append("../DeepAlignmentNetwork")
import os
import utils
import cv2
from tqdm import tqdm

img_dir = "../data/images/thermal_detected"

out_dir = "../data/images/thermal_downscaled/"

for img_file in tqdm(os.listdir(img_dir)):
    if img_file.endswith("png"):
        pts_file = img_file[:-3]+"pts"
        img = cv2.imread(os.path.join(img_dir, img_file), cv2.IMREAD_GRAYSCALE)
        # print(img.shape)
        # downsize image
        d_img = cv2.resize(img, (int(1/6*img.shape[1]), int(1/6*img.shape[0])))
        # print(d_img.shape)
        cv2.imwrite((out_dir+ img_file), d_img)
        # downsize pts
        pts = utils.loadFromPts(os.path.join(img_dir, pts_file))
        # print("before: {}, {}".format(pts[0,0], pts[0,1]))
        pts = pts*1/6
        # print("after: {}, {}".format(pts[0, 0], pts[0, 1]))
        utils.saveToPts(os.path.join(out_dir, pts_file), pts)
        # break

