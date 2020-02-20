import tiffcapture as tc
import numpy as np
import LocalizeLandmarks
from matplotlib import pyplot as plt
import cv2
# nose  32-36
# mouth 49-68


class BreathAnalyzer:
    def __init__(self, video_file = None):
        self.video_file = video_file
        self.video_tiff = None
        self.ac = None
        self.dc = None
        self.landmarks = None
        self.landmarkFrame = None # avoid duplicate load

        # s=
    def load_video(self, video_file=None):
        if self.video_file is None and video_file is None:
            raise Exception("No video to be loaded!")
        # try:
        if self.video_file:
            self.video_tiff = tc.opentiff(self.video_file)
        else:
            self.video_tiff = tc.opentiff(video_file)

        valid, img = self.video_tiff.read()
        img = cv2.normalize(img, img, 255.0, 0.0, cv2.NORM_MINMAX)

        self.dc = np.zeros(img.shape, np.double)
        self.dc = self.dc[np.newaxis]
        while valid:
            self.dc = np.vstack((self.dc, img[np.newaxis]))
            # dst = np.zeros(img.shape, np.double)
            valid, img = self.video_tiff.read()
            img = cv2.normalize(img, img, 255.0, 0.0, cv2.NORM_MINMAX)
        print("Video loaded! Video size: {}".format(self.dc.shape))#self.dc.shape[0],self.dc.shape[1], self.dc.shape[2]))

    def get_nose_rect(self, frame=1):
        if self.landmarkFrame != frame:
            self.localize_landmarks(frame)

        nose_x_min = np.min(self.landmarks[31:36,0], axis=0)
        nose_x_max = np.max(self.landmarks[31:36,0], axis=0)
        nose_y_min = np.min(self.landmarks[30:36,1], axis=0)
        nose_y_max = np.max(self.landmarks[30:36,1], axis=0)

        return [nose_x_min, nose_x_max, nose_y_min, nose_y_max]

    def get_mouth_rect(self, frame=1):
        if self.landmarkFrame != frame:
            self.localize_landmarks(frame)

        mouth_x_min = np.min(self.landmarks[48:68,0])
        mouth_x_max = np.max(self.landmarks[48:68,0])
        mouth_y_min = np.min(self.landmarks[48:68,1])
        mouth_y_max = np.max(self.landmarks[48:68,1])

        return [mouth_x_min, mouth_x_max, mouth_y_min, mouth_y_max]


    def localize_landmarks(self, nr=1):
        if self.video_file is None or self.video_tiff is None:
            raise Exception("Load video first!")
        self.landmarks = LocalizeLandmarks.localize(self.dc[nr,:,:])
        self.landmarkFrame = nr
        return self.landmarks

    def test_extraction(self):
        nose = self.get_nose_rect()
        mouth = self.get_mouth_rect()
        if self.dc is None:
            self.load_video()
        img = self.dc[0,:,:]

        cv2.rectangle(img, (int(nose[0]), int(nose[2])), (int(nose[1]), int(nose[3])), 1, 2)
        cv2.rectangle(img, (mouth[0], int(mouth[2])), (int(mouth[1]), int(mouth[3])), 1, 2)

        plt.imshow(img)
        plt.show()



