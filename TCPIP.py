import socket
import threading


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


def clientUtilities(data):
    PORT = 5050
    # SERVER = "192.168.1.1"
    SERVER = "127.0.0.1"
    ADDR = (SERVER, PORT)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(ADDR)

    def send(msg):
        # HEADER = 64
        FORMAT = 'utf-8'
        message = msg.encode(FORMAT)
        # msg_length = len(message)
        # send_length = str(msg_length).encode(FORMAT)
        # send_length += b' ' * (HEADER - len(send_length))
        # client.send(send_length)
        client.send(message)

    send(createData(zeroExtend(data)))
    client.close()


def sendData(dataList):
    try:
        assert type(dataList) is list
    except AssertionError:
        print('[ERROR] Wrong data format. Check your data again.')
        dataList = []
    finally:
        thread = threading.Thread(target=clientUtilities, args=[dataList])
        thread.start()
        print('Data Sent.')
