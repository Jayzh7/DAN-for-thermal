# input a image of np array type and return a rectangle
import numpy as np
import dlib
import menpodetect
import cv2
from menpo import io as mio
from matplotlib import pyplot as plt
import matplotlib.patches as patches

test_img = "../data/images/thermal_downscaled/irface_sub001_seq02_frm00055.jpg_lfb.png"
fd_model = "../data/hog_detector.svm"

def scale_detector(img, face_detector):
    [h, w] = img.shape[:2]
    for scale in range(2,6):
        new_img = cv2.resize(img, (scale*w, scale*h))
        cv2.imwrite("temp.png", new_img)
        img = mio.import_image('temp.png')
        bb = face_detector(img)
        if bb is not None:
            return [bb, scale]
        print("No face detected at scale {}".format(scale))
    return [None, None]

def load_detector(model=fd_model):
    face_detector = menpodetect.DlibDetector(dlib.simple_object_detector(model))
    return face_detector

def detect(img_file, detector = load_detector(fd_model)):
    # img = mio.import_image(img_file)
    img = cv2.imread(img_file)
    scale = 1
    face_bb = detector(mio.import_image(img_file))
    if face_bb is None or len(face_bb) == 0:
        if len(img.shape) > 2:
            gray_img = np.mean(img, axis=2).astype(np.uint8)
        else:
            gray_img = img.astype(np.uint8)
        [face_bb, scale] = scale_detector(gray_img, detector)
    if face_bb is None or len(face_bb) == 0:
        return [None, None]
    else:
        bb = face_bb[0].as_vector()
        rect = [bb[1]/scale, bb[0]/scale, (bb[5])/scale, (bb[4])/scale]

        return rect

if __name__ == '__main__':
    # [face_bb, scale] = detect(test_img)
    rect = detect(test_img)
    img = cv2.imread(test_img)

    fig, ax = plt.subplots(1)
    # for bb in face_bb:
    #     bb = bb.as_vector()
    #     rect = patches.Rectangle((bb[1]/scale, bb[0]/scale), (bb[5] - bb[1])/scale, (bb[4] - bb[0])/scale, edgecolor='r', linewidth=3, facecolor='none')
    #     print(bb[:4])
    rect = patches.Rectangle((rect[0], rect[1]), rect[2]-rect[0], rect[3]-rect[1])
    ax.imshow(img)
    ax.add_patch(rect)

    plt.show()
