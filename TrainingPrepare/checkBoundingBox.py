from matplotlib import pyplot as plt
import matplotlib.patches as patches
import pickle
import os
with open("../TrainingPrepare/boxesThermalDownscaledAll.pkl", "rb") as fp:
    dict = pickle.load(fp)

img_dir = "../data/images/thermal_downscaled/"
# for img_file in dict:
for i in range(1):
    img_file = "irface_sub017_seq03_frm00662.jpg_lfb.png";

    img = plt.imread(os.path.join(img_dir, img_file))

    bb = dict[img_file]
    #
    fig, ax = plt.subplots(1)
    print(bb)
    rect = patches.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], edgecolor='r', linewidth=3, facecolor='none')

    ax.imshow(img)
    ax.add_patch(rect)

    plt.show()


