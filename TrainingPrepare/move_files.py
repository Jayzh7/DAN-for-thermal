import pickle
import os
import shutil
with open("../data/boxesThermal.pkl", "rb") as fp:
    dict = pickle.load(fp)
img_dir = "../data/images/thermal_all"
img_new_dir =  "../data/images/thermal_detected"
for key in dict:
    # shutil.move(os.path.join(img_dir, key), os.path.join(img_new_dir, key))
    pts_file = key[:-3]+'pts'
    shutil.move(os.path.join(img_dir, pts_file), os.path.join(img_new_dir, pts_file))
