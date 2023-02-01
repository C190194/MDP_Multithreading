import socket
import threading
import cv2
import pickle
import struct
import time

# Start server
class Connect_PC_Client(threading.Thread):

    def __init__(self, name):
        super(Connect_PC_Client, self).__init__()
        self.name = name
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
                stream_thread = Start_Stream("Stream Thread", client_socket)
                # create the thread for receiving img recog results
                img_result_thread = Receive_Img_Results("Img Result Thread", client_socket)

                # start threads
                stream_thread.start()
                print('Waiting for the stream thread to finish ...')
                img_result_thread.start()
                print('Waiting for the img result thread to finish ...')

                # keep updating img recog results
                while img_result_thread.is_alive():
                    self.img_results = img_result_thread.results
                    print(self.img_results)
                    time.sleep(0.1)

                stream_thread.join()
                img_result_thread.join()

class Start_Stream(threading.Thread):

    def __init__(self, name, client_socket):
        super(Start_Stream, self).__init__()
        self.name = name
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

    def __init__(self, name, client_socket):
        super(Receive_Img_Results, self).__init__()
        self.name = name
        self.client_socket = client_socket

    def run(self):
        #'b' or 'B'produces an instance of the bytes type instead of the str type
        #used in handling binary data from network connections
        data = b""
        # Q: unsigned long long integer(8 bytes)
        payload_size = struct.calcsize("Q")
        while True:
            while len(data) < payload_size:
                packet = self.client_socket.recv(1024)
                if not packet: break
                data+=packet
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("Q",packed_msg_size)[0]
            while len(data) < msg_size:
                data += self.client_socket.recv(1024)
            result_data = data[:msg_size]
            data  = data[msg_size:]
            self.results = pickle.loads(result_data)
                
if __name__ == '__main__':
    PC_thread = Connect_PC_Client(name="PC Thread")
    PC_thread.start()
    print("Waiting for PC thread to finish...")
    # show current img recog results
    while PC_thread.is_alive():
        print(PC_thread.img_results)
        time.sleep(0.1)
    PC_thread.join()