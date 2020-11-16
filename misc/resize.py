import cv2
import os

path = "./dataset"
wpath = "../dataset/"
for filename in os.listdir(path):
    img = cv2.imread(os.path.join(path,filename))
    img = cv2.resize(img,(64,64))
    cv2.imwrite(os.path.join(wpath,filename), img)
