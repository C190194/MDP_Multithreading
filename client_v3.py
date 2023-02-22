#importing libraries
import socket
import cv2
import pickle
import struct
import math
import time
import os
import glob
# import imutils
#from signal import signal, SIGPIPE, SIG_DFL
from ultralytics import YOLO


# for name in glob.glob("./T/*"):
#     img = cv2.imread(name)
#     img = cv2.resize(img, (640,640))
#     cv2.imwrite(name, img)
# input("done")

# l = [1,2,3]
# for n in reversed(range(len(l))):
#     if l[n] == 1 or l[n] ==2:
#         l.pop(n)
# input(l)

# Config
class_label = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'Bullseye', 'C', 'D', 'Down', 'E', 'F', 'G', 'H', 'Left', 'Right', 'S', 'Stop', 'T', 'U', 'Up', 'V', 'W', 'X', 'Y', 'Z']
class_dcp = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", \
            "Alphabet A", "Alphabet B", "Bullseye", "Alphabet C", "Alphabet D", \
            "down arrow", "Alphabet E", "Alphabet F", "Alphabet G", "Alphabet H", \
            "left arrow", "right arrow", "Alphabet S", "Stop", "Alphabet T", \
            "Alphabet U", "up arrow", "Alphabet V", "Alphabet W", "Alphabet X", \
            "Alphabet Y", "Alphabet Z"]
class_id = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, -1, 22, 23, 37, 24, 25, 26, 27, 39, 38, 28, 40, 29, \
            30, 36, 31, 32, 33, 34, 35] # -1 for Bullseye
Bullseye_Index = class_label.index("Bullseye")
S_Index = class_label.index("S")
G_Index = class_label.index("G")
Image_Width_Center = 320 / 2
Image_Height_Change_Ratio = 320 / 240

def process_single_img(result_list):
    box_list = []
    box_position = []
    for box in result_list[0][0].boxes:
        if box.cls == Bullseye_Index:
            continue
        box_list.append(box)
        box_position.append(abs(Image_Width_Center - box.xywh.tolist()[0][0])) # box.conf [box.xywh, box.cls]
    if not box_position: # Nothing detected => need to move a bit to take another picture
        return -1
    if len(box_list) > 1:
        print("Multiple letters detected")
        # remove "S" and "G" if multiple letters are detected
        for i in reversed(range(len(box_list))):
            box = box_list[i]
            cls_idx = int(box.cls.tolist()[0])
            if cls_idx == S_Index or cls_idx == G_Index:
                box_list.pop(i)
                box_position.pop(i)
    # ** TBD: see if distant cards will be captured
    box_idx = box_position.index(min(box_position))
    box = box_list[box_idx]
    
    bbox_xyxy = box.xyxy.tolist()[0]
    # bbox_xyxy[1] = bbox_xyxy[1] / Image_Height_Change_Ratio
    # bbox_xyxy[3] = bbox_xyxy[3] / Image_Height_Change_Ratio
    label_idx = int(box.cls.tolist()[0])
    return [bbox_xyxy, label_idx]

# TEST
# model = YOLO("./best.pt")
# path = "./T/img_79.jpg"
# results = model.predict(show = False,source=path, save=False, save_txt=False)
# print("--- Results ---")
# res = process_single_img([results])
# img = cv2.imread(path)
# tl = (int(res[0][0]),int(res[0][1]))
# br = (int(res[0][2]),int(res[0][3]))
# cv2.rectangle(img,tl,br,(0,255,0),2)
# text = 
# cv2.putText(img,res[1],(int(res[0][2])+5,int(res[0][3])),0,0.7,(0,255,0))
# cv2.imshow("Show",img)
# key = cv2.waitKey()
# if key  == 27: # press "ESC" to end connection
#     input("dd")


#signal(SIGPIPE,SIG_DFL)
# Client socket
# create an INET, STREAMing socket : 
client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
# host_ip = '<localhost>'# Standard loopback interface address (localhost)
host_ip = '192.168.3.3' # put the ip of the server here
port = 10050 # Port to listen on (non-privileged ports are > 1023)
# now connect to the web server on the specified port number
client_socket.connect((host_ip,port)) 

# Load Yolo v8 model
model = YOLO("./best_v8s_new.pt")

'''
# Receive obstacle info and calculate path command from RPi
while True:
    data = client_socket.recv(4*1024)
    if not data:
        continue
    s = data.decode('UTF-8').strip()
    print("Received from RPi:", s)
    if s == "path":
        break

# Send the car path to RPi
path_object = ["some path"]
a = pickle.dumps(path_object)
message = struct.pack(">L",len(a))+a
client_socket.sendall(message)
print("Car path sent")

'''
# Wait for images from RPi 
#client_socket.settimeout(5)
while True:
    #'b' or 'B'produces an instance of the bytes type instead of the str type
    #used in handling binary data from network connections
    data = b""
    # Q: unsigned long long integer(8 bytes)
    payload_size = struct.calcsize(">L")

    # Receive stream frames
    result_list = []
    for i in range(1):
        print('Waiting for img', i)
        while len(data) < payload_size:
            packet = client_socket.recv(4*1024)
            if not packet: break
            data+=packet
        # t1 = time.time()
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L",packed_msg_size)[0]
        while len(data) < msg_size:
            data += client_socket.recv(4*1024)
        frame_data = data[:msg_size]
        data  = data[msg_size:]
        frame = pickle.loads(frame_data,fix_imports=True,encoding="bytes")
        frame = cv2.imdecode(frame,cv2.IMREAD_COLOR)
        resized_frame = cv2.resize(frame, (640,640))
        # cv2.imwrite('image_T_'+str(21)+'.jpg', resized_frame)
        # t2 = time.time()
        # print("Time taken to receive the image:", t2-t1)
        results = model.predict(show=False, source=resized_frame, save=False, save_txt=False, device="cpu")
        result_list.append(results)
    # process single image
    res = process_single_img(result_list)
    if res == -1:
        final_result = res
        # input("hold")
    else:
        tl = (int(res[0][0]),int(res[0][1]/Image_Height_Change_Ratio))
        br = (int(res[0][2]),int(res[0][3]/Image_Height_Change_Ratio))
        cv2.rectangle(frame,tl,br,(0,255,0),2)
        text1 = class_dcp[res[1]]
        text2 = "Image ID: " + str(class_id[res[1]])
        tx = int(res[0][2])
        if tx+175 > 639:
            tx = int(res[0][0]-175)
            if tx < 0:
                tx = 0
        ty = int((res[0][1]+res[0][3])/2/Image_Height_Change_Ratio)
        # frame = cv2.resize(frame, (640,480))
        cv2.rectangle(frame,(tx+5,ty-25),(tx+170,ty+25),(255,255,255),-1)
        cv2.putText(frame,text1,(tx+5,ty),0,0.8,(0,255,0),1)
        cv2.putText(frame,text2,(tx+5,ty+20),0,0.8,(0,255,0),1)
        cv2.imwrite('image'+str(1)+'.jpg', frame)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY),90]
        result, frame = cv2.imencode('.jpg', frame, encode_param)
        final_result = [frame, class_dcp[res[1]]]
        
    # send result
    a = pickle.dumps(final_result)
    message = struct.pack(">L",len(a))+a
    client_socket.sendall(message)
    print("Results sent:")
    print(final_result)
    # client_socket.close()
    # break
        # key = cv2.waitKey(10) # -1 will be returned if no key is pressed
        # if key  == 27: # press "ESC" to end connection
        #     break

client_socket.close()

# Load the images
image1 = cv2.imread('image1.jpg')
image2 = cv2.imread('image2.jpg')
image3 = cv2.imread('image3.jpg')
image4 = cv2.imread('image4.jpg')
image5 = cv2.imread('image5.jpg')
image6 = cv2.imread('image6.jpg')

# Stack the images vertically
row1 = np.hstack((image1, image2,image3))
row2 = np.hstack((image4, image5,image6))

# Stack the rows vertically
stacked_image = np.vstack((row1, row2))

# Display the combined image
cv2.imshow("Stitched Image", stacked_image)
cv2.waitKey('A')
cv2.destroyAllWindows()


# def maj_vote(result_list):
#     img_box = [] # 1 box for 1 image
#     img_label = [] # 1 label for 1 image
#     for results in result_list:
#         # in 1 image, get the bbox that is closest to the center
#         box_position = []
#         for box in results[0].boxes:
#             if box.cls == Bullseye_Index:
#                 box_position.append(1000)
#             box_position.append(abs(Image_Width_Center - box.xywh[2])) # box.conf [box.xywh, box.cls]
#         box_idx = box_position.index(min(box_position))
#         box = results[0].boxes[box_idx]
#         img_box.append(box)
#         img_label.append(box.cls)
    
#     highest_freq = 0
#     label_idx = 0
#     for l in set(img_label):
#         if l == Bullseye_Index:
#             continue
#         num = img_label.count(l)
#         if num > highest_freq:
#             highest_freq = num
#             label_idx = l
#     if highest_freq == 0: # Nothing detected
#         return -1
#     img_idx = img_label.index(label_idx)
#     bbox_xyxy = img_box[img_idx].xyxy
#     label = class_label[label_idx]
#     return [img_idx, bbox_xyxy, label]
