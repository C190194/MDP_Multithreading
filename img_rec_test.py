#importing libraries
import socket
import cv2
import pickle
import struct
# import imutils
#from signal import signal, SIGPIPE, SIG_DFL
from ultralytics import YOLO

# Config
class_label = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'Bullseye', 'C', 'D', 'Down', 'E', 'F', 'G', 'H', 'Left', 'Right', 'S', 'Stop', 'T', 'U', 'Up', 'V', 'W', 'X', 'Y', 'Z']
Bullseye_Index = class_label.index("Bullseye")
S_Index = class_label.index("S")
G_index = class_label.index("G")
Image_Width_Center = 320 / 2

# Load Yolo v8 model
model = YOLO("./best_v8s.pt")

path = "./8/img_17.jpg"
img = cv2.imread(path)
img= cv2.resize(img, (640,640))
for i in range(1):
    results = model.predict(show = False,source=img, save=False, save_txt=False)
    print("--- Results ---")
    for box in results[0].boxes:
        print(class_label[int(box.cls.tolist()[0])], box.conf.tolist()[0])