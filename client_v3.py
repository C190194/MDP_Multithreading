#importing libraries
import socket
import cv2
import pickle
import struct
import math
# import imutils
#from signal import signal, SIGPIPE, SIG_DFL
from ultralytics import YOLO

# Config
class_label = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'Bullseye', 'C', 'D', 'Down', 'E', 'F', 'G', 'H', 'Left', 'Right', 'S', 'Stop', 'T', 'U', 'Up', 'V', 'W', 'X', 'Y', 'Z']
Bullseye_Index = class_label.index("Bullseye")
Image_Width_Center = 640 / 2

#signal(SIGPIPE,SIG_DFL)
# Client socket
# create an INET, STREAMing socket : 
client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
# host_ip = '<localhost>'# Standard loopback interface address (localhost)
host_ip = '192.168.3.3' # put the ip of the server here
port = 10050 # Port to listen on (non-privileged ports are > 1023)
# now connect to the web server on the specified port number
client_socket.connect((host_ip,port)) 
#'b' or 'B'produces an instance of the bytes type instead of the str type
#used in handling binary data from network connections
data = b""
# Q: unsigned long long integer(8 bytes)
payload_size = struct.calcsize(">L")
#print("payload_size: {}".format(payload_size))
# Load Yolo v8 model
model = YOLO("./best.pt")



def maj_vote(result_list):
    img_box = [] # 1 box for 1 image
    img_label = [] # 1 label for 1 image
    for results in result_list:
        # in 1 image, get the bbox that is closest to the center
        box_position = []
        for box in results[0].boxes:
            if box.cls == Bullseye_Index:
                box_position.append(math.inf)
            box_position.append(abs(Image_Width_Center - box.xywh[2])) # box.conf [box.xywh, box.cls]
        box_idx = box_position.index(min(box_position))
        box = results[0].boxes[box_idx]
        img_box.append(box)
        img_label.append(box.cls)
    
    highest_freq = 0
    label_idx = 0
    for l in set(img_label):
        if l == Bullseye_Index:
            continue
        num = img_label.count(l)
        if num > highest_freq:
            highest_freq = num
            label_idx = l
    if highest_freq == 0: # Nothing detected
        return -1
    img_idx = img_label.index(label_idx)
    bbox_xyxy = img_box[img_idx].xyxy
    label = class_label[label_idx]
    return [img_idx, bbox_xyxy, label]

while True:
    # Receive stream frames
    result_list = []
    for i in range(5):
        print('Waiting for img', i)
        while len(data) < payload_size:
            packet = client_socket.recv(4*1024)
            if not packet: break
            data+=packet
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L",packed_msg_size)[0]
        while len(data) < msg_size:
            data += client_socket.recv(4*1024)
        frame_data = data[:msg_size]
        data  = data[msg_size:]
        frame = pickle.loads(frame_data,fix_imports=True,encoding="bytes")
        frame = cv2.imdecode(frame,cv2.IMREAD_COLOR)
        cv2.imshow("Receiving...",frame)
        results = model.predict(show=True, source=frame, save=False, save_txt=False, device="cpu")
        result_list.append(results)
        # do majority vote
        vote_result = maj_vote(result_list)
        # send result
        a = pickle.dumps(vote_result)
        message = struct.pack(">L",len(a))+a
        client_socket.sendall(message)
        print("Results sent:")
        print(vote_result)
        # key = cv2.waitKey(10) # -1 will be returned if no key is pressed
        # if key  == 27: # press "ESC" to end connection
        #     break
client_socket.close()