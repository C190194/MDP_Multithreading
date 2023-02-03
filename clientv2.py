#importing libraries
import socket
import cv2
import pickle
import struct
# import imutils
#from signal import signal, SIGPIPE, SIG_DFL
from ultralytics import YOLO

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

# Receive stream frames
while True:
    # print('testing')
    while len(data) < payload_size:
        packet = client_socket.recv(4*1024)
        if not packet: break
        data+=packet
    # print(len(data))
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
    cv2.waitKey(1)
    results = model.predict(show = True,source=frame, save=False, save_txt=False, device="cpu")
    #results = 1
    if results:
        #input(results)
        result_list = []
        
        for box in results[0].boxes:
            box_info = [box.xywh, box.conf, box.cls]
            result_list.append(box_info)
        if len(result_list) == 0:
            result_list = "Nothing detected!"
        
        a = pickle.dumps(result_list)
        message = struct.pack(">L",len(a))+a
        # print(message)
        client_socket.sendall(message)
        print("Results sent:")
        print(result_list)
    key = cv2.waitKey(10) # -1 will be returned if no key is pressed
    if key  == 27: # press "ESC" to end connection
        break
client_socket.close()