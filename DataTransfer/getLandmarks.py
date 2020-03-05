import hdf5storage
import sys
import scipy.io as sio
from tqdm import tqdm
sys.path.append("../DeepAlignmentNetwork/")
from FaceAlignment import FaceAlignment
# from heuri_face_detector import threshold_face_detector
import numpy as np
import cv2
import utils
from matplotlib import pyplot as plt
from menpo import io as mio
import menpodetect
import pickle
import dlib
import os
from heuri_face_detector import threshold_face_detector as tfd

print("Loading video...")

subject = "s3"

mat = hdf5storage.loadmat('videos/' + subject + '_dc.mat')
dc = mat['dc']

model = FaceAlignment(112, 112, 1, 2, False)
model.loadNetwork("../data2/networks/network_00061_2020-01-10-14-00.npz")

# face_bb = [95, 72, 224, 179]

# nose_dc = np.zeros()

nr_frame = dc.shape[2]
all_landmarks = np.zeros([68, 2, len(range(0, nr_frame, 5))])


for i in tqdm(range(nr_frame//5)):
# for i in range(1):
    curr_frame = dc[:,:,i*5]
    curr_frame = np.subtract(curr_frame, np.min(curr_frame))
    curr_frame = np.divide(curr_frame, np.max(curr_frame))
    curr_frame = np.multiply(curr_frame, 256)
    curr_frame = curr_frame.astype(np.uint8)
    # curr_frame = np.multiply(np.divide(np.subtract(curr_frame, np.min(curr_frame)), np.max(curr_frame)), 256).astype(np.uint8)

    face_bb = tfd(dc[:,:,i*5])
    # face_bb[3] -= 30
    print(face_bb)
    initLandmarks = utils.bestFitRect(None, model.initLandmarks, face_bb)
    all_landmarks[:,:,i] = model.processImg(curr_frame[np.newaxis], initLandmarks)

# plt.imshow(curr_frame)
# plt.show()
sio.savemat(subject + '_landmarks.mat', {'landmarks': all_landmarks})
