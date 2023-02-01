import socket
import threading
import cv2
import pickle
import struct
import time

# Start server
class Connect_PC_Client(threading.Thread):

    def __init__(self):
        self.server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.host_name  = socket.gethostname()
        self.host_ip = socket.gethostbyname(self.host_name)
        print('RPi Wifi HOST IP:',self.host_ip)
        self.port = 10050
        self.socket_address = (self.host_ip,self.port)
        print('RPi Wifi Server Socket created')
        # bind the socket to the host. 
        #The values passed to bind() depend on the address family of the socket
        self.server_socket.bind(self.socket_address)
        print('Socket bind complete')
        #listen() enables a server to accept() connections
        #listen() has a backlog parameter. 
        #It specifies the number of unaccepted connections that the system will allow before refusing new connections.
        self.server_socket.listen(5)
        print('Socket now listening')

    def run(self):
        while True:
            client_socket,addr = self.server_socket.accept()
            print('Connection from:',addr)
            if client_socket:
                # create the thread for streaming
                stream_thread = Start_Stream()
                # create the thread for receiving img recog results
                img_result_thread = Receive_Img_Results()

                # start threads
                stream_thread.start()
                print('Waiting for the stream thread to finish ...')
                img_result_thread.start()
                print('Waiting for the img result thread to finish ...')

                stream_thread.join()
                img_result_thread.join()

class Start_Stream(threading.Thread):

    def __init__(self, client_socket):
        self.client_socket = client_socket

    def run(self):
        vid = cv2.VideoCapture(0)
        while(vid.isOpened()):
            img,frame = vid.read()
            a = pickle.dumps(frame)
            message = struct.pack("Q",len(a))+a
            self.client_socket.sendall(message)
            cv2.imshow('Sending...',frame)
            key = cv2.waitKey(10) 
            # print(key) # -1 will be returned if no key is pressed
            if key == 27: # "ESC"
                self.client_socket.close()

class Receive_Img_Results(threading.Thread):

    def __init__(self, client_socket):
        self.client_socket = client_socket

    def run(self):
        self.results = self.client_socket.recv(1024)
                
if __name__ == '__main__':
    PC_thread = Connect_PC_Client()
    PC_thread.start()
    print("Waiting for PC thread to finish...")
    while Receive_Img_Results.is_alive():
        img_results = Receive_Img_Results.results
        print(img_results)
        time.sleep(0.1)
    PC_thread.join()