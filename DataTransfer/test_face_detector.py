import numpy as np
import hdf5storage
import sys
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from heuri_face_detector import threshold_face_detector as tfd

mat = hdf5storage.loadmat("videos/s3_dc.mat")
dc = mat['dc']

img = dc[:,:,0]

face_bb = tfd(img)

# face_bb[3] -= 30
print(face_bb)

# plt.imshow(img)

fig, ax = plt.subplots(1)
rect = patches.Rectangle((face_bb[0], face_bb[1]), face_bb[2]-face_bb[0], face_bb[3]-face_bb[1], edgecolor='r', linewidth=3, facecolor='none')

ax.imshow(img)
ax.add_patch(rect)

plt.show()
