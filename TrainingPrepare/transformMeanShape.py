import numpy as np

a = np.load("../data/meanFaceShape.npz")
shape = a['meanShape']

nr_to_add = [27, 28, 29, 30, 31, 32, 33, 34, 35, 39, 42, 48, 54]

newShape = np.array(shape[nr_to_add[0]])
# print(newShape)
for k in range(1, len(nr_to_add)):
    # print(k)
    newShape = np.vstack((newShape, shape[nr_to_add[k]]))
# print(newShape)
np.savez("../data/reducedMeanShape.npz", meanShape=newShape)

a = np.load("../data/reducedMeanShape.npz")
# print(a['meanShapei.png'])
