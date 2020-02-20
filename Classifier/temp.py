import tiffcapture as tc
import matplotlib.pyplot as plt
import cv2
import numpy as np
tiff = tc.opentiff("../data/frontal_view_front_angle.tiff")

canvas = None
valid, img = tiff.read()
i = 0
while valid:
    dst = np.zeros(img.shape, np.double)
    dst = cv2.normalize(img, dst, 255.0, 0.0, cv2.NORM_MINMAX)
    if canvas is None:
        canvas = plt.imshow(dst)
    else:
        canvas.set_data(dst)
    plt.pause(.1)
    plt.draw()
    # cv2.imshow("video", dst)
    valid, img = tiff.read()
    # print(np.mean(dst, axis=(0,1)))
    # cv2.waitKey(50)

cv2.destroyAllWindows()
