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
model = YOLO("./best.pt")

img = "./T_result/img_79.jpg"
for i in range(5):
    results = model.predict(show = False,source=img, save=False, save_txt=False)
    print("--- Results ---")
    for box in results[0].boxes:
        print(class_label[int(box.cls.tolist()[0])], box.conf.tolist()[0])