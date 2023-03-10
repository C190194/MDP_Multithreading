import socket
import cv2
import pickle
import struct
import math
import time
import os
import glob

server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # set the socket addr to be reusable
host_name  = socket.gethostname()
host_ip = socket.gethostbyname(host_name)
print('RPi Wifi HOST IP:',host_ip)
port = 10050
socket_address = (host_ip,port)
print('RPi Wifi Server Socket created')
# bind the socket to the host. 
#The values passed to bind() depend on the address family of the socket
server_socket.bind(socket_address)
print('Socket bind complete')
#listen() enables a server to accept() connections
#listen() has a backlog parameter. 
#It specifies the number of unaccepted connections that the system will allow before refusing new connections.
server_socket.listen(5)
print('Wifi Socket now listening\n')

try:
    while True:
        client_socket,addr = server_socket.accept()
        client_socket.settimeout(5)
        print('Connection from:',addr)
        while True:
            client_socket.send(b"Hi from server")
            # msg = client_socket.recv(1024).decode("utf-8")
            #print(msg)
            time.sleep(3)
            
except Exception as e:
    print(e)
    client_socket.close()
    server_socket.close()
    
    # if client_socket:
    #     vid = cv2.VideoCapture(0)
    #     while(vid.isOpened()):
    #         img,frame = vid.read()
    #         a = pickle.dumps(frame)
    #         message = struct.pack("Q",len(a))+a
    #         client_socket.sendall(message)
    #         cv2.imshow('Sending...',frame)
    #         key = cv2.waitKey(10) # -1 will be returned if no key is pressed
    #         if key  == 27: # press "ESC" to end connection
    #             client_socket.close()