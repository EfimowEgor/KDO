import os

import cv2

base_dir = "./data/test_set"
out_path = "./data/test_set"

# Rename images
# for index, name in enumerate(os.listdir(base_dir)):
#     path = os.path.join(base_dir, name)
#     desired_name = os.path.join(base_dir, "image" + str(index) + ".JPG")
#     os.rename(path, desired_name)

# Resize images
for index, image in enumerate(os.listdir(base_dir)):
    path = os.path.join(base_dir, image)
    opath = os.path.join(out_path, image)
    mat = cv2.imread(path)
    mat = cv2.resize(mat, (640, 640))
    cv2.imwrite(opath, mat)
