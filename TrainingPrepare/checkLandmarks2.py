from matplotlib import pyplot as plt
import sys
sys.path.append("../DeepAlignmentNetwork")
import cv2
import os
import numpy as np
# file1 = "../data/dataset_nimgs=25850_perturbations=[0.2, 0.2, 20, 0.25]_size=[112, 112].npz"
file1 = "../data2/dataset_nimgs=24850_perturbations=[0.2, 0.2, 20, 0.25]_size=[112, 112].npz"
file2 = "../data/dataset_nimgs=100_perturbations=[]_size=[112, 112].npz"

a = np.load(file1)
b = np.load(file2)
# print(a.files)
# print(b.files)
nr = 12
img = a['imgs'][nr][0]
lm = a['gtLandmarks'][nr]
# lm = a['gtLandmarks'][nr]
print(a['boundingBoxes'][nr])
for p in lm:
    cv2.circle(img, (int(p[0]), int(p[1])), 1, 1, -1)

plt.imshow(img)
plt.show()
