from matplotlib import pyplot as plt
import matplotlib.patches as patches
import pickle
import os
with open("../TrainingPrepare/boxesThermal.pkl", "rb") as fp:
    dict = pickle.load(fp)

img_dir = "../data/images/thermal_detected/"
for img_file in dict:
    # img_file = "irface_sub093_seq07_frm00447.jpg_lfb.png";

    img = plt.imread(os.path.join(img_dir, img_file))

    bb = dict[img_file]
    #
    fig, ax = plt.subplots(1)

    rect = patches.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], edgecolor='r', linewidth=3, facecolor='none')

    ax.imshow(img)
    ax.add_patch(rect)

    plt.show()


