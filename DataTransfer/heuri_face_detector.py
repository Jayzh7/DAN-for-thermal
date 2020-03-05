import numpy as np

def threshold_face_detector(img):
    th_v = 0.6
    th_h = 0.7
    img = img.astype(np.float64)

    vp = np.mean(img, axis=0)
    vp = np.subtract(vp, np.min(vp))
    hp = np.mean(img, axis=1)
    hp = np.subtract(hp, np.min(hp))

    max_vp = 0
    idx_vp = 0
    max_hp = 0
    idx_hp = 0
    for k in range(len(vp)):
        if vp[k] > max_vp:
            max_vp = vp[k]
            idx_vp = k
    # print("max_vp idx, {}:{}".format(idx_vp, max_vp))
    # print("{}:{}".format(np.max(vp), np.argmax(vp)))
    for k in range(len(hp)):
        if hp[k] > max_hp:
            max_hp = hp[k]
            idx_hp = k
    # print("max_hp idx, {}".format(idx_hp))
    # max_vp -= np.min(img, axis=0)
    # center dispersal
    tl_x = idx_vp
    br_x = idx_vp

    while tl_x >= 0 and vp[tl_x] >= max_vp*th_v:
        tl_x -= 1
    while br_x < len(vp) and vp[br_x] >= max_vp*th_v:
        br_x += 1

    tl_y = idx_hp
    br_y = idx_hp

    while tl_y >= 0 and hp[tl_y] >= max_hp*th_h:
        tl_y -= 1
    while br_y < len(hp) and hp[br_y] >= max_hp*th_h:
        br_y += 1

    return [tl_x, tl_y, br_x, br_y]