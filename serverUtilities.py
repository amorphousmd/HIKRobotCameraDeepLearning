import socket
import threading
import time

HEADER = 64
PORT = 5050
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
conn = None
addr = None
connected = False

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)

def createData(coords_list):
    list1 = [elem for coords in coords_list for elem in coords]
    msg = ""
    for i in list1:
        msg += ","
        msg += str(i)
    msg = str(len(coords_list)) + msg
    return msg


def zeroExtend(inputList):
    output = []
    for tup in inputList:
        tup = (*tup, 0)
        output.append(tup)
    return output


def establish_connection():
    global connected
    global conn
    global addr
    if not connected:
        server.listen()
        print(f"[LISTENING] Server is listening on {SERVER}")
        thread = threading.Thread(target=accept_client_nonblocking)
        thread.start()
        connected = True
    else:
        # In case you need to restart code on the client
        thread = threading.Thread(target=accept_client_nonblocking)
        thread.start()

def accept_client_nonblocking():
    global conn
    global addr
    conn, addr = server.accept()

def accept_client_again():
    global conn
    global addr
    conn.close()
    conn = None
    addr = None
    server.listen()
    print(f"[LISTENING] Server is listening on {SERVER}")
    conn, addr = server.accept()


def send_data_thread(data):
    if not data:
        # print('Sent')
        return
    def send(msg):
        FORMAT = 'utf-8'
        message = msg.encode(FORMAT)
        # conn will be redefined
        conn.send(message)
    send(createData(zeroExtend(data)))
