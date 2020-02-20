from matplotlib import pyplot as plt
import matplotlib.patches as patches
import pickle
import os
import numpy as np
import cv2

file1 = "../data2/dataset_nimgs=24850_perturbations=[0.2, 0.2, 20, 0.25]_size=[112, 112].npz"
file2 = "../data2/dataset_nimgs=200_perturbations=[]_size=[112, 112].npz"
a = np.load(file1)
b = np.load(file2)

nr = 0
img = cv2.imread(a['filenames'][nr])
bb = a['boundingBoxes'][nr]

# for img_file in dict:
for i in range(1):

    #
    fig, ax = plt.subplots(1)
    print(bb)
    rect = patches.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], edgecolor='r', linewidth=3, facecolor='none')

    ax.imshow(img)
    ax.add_patch(rect)

    plt.show()


