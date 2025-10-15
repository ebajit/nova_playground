from socket import *
import threading

def dm_msg(msg, receiver):
    user_database[receiver][1].send(msg.encode())

def user_sent_msgs(receiver):
    while True:
        try:
            msg = user_database[receiver][1].recv(1024).decode()
            if (msg):
                print(msg)
        except:
            continue

def user_joined_msgs(socket):
    while True:
        connectionSocket, addr = server_sock.accept()
        print(f"{addr}")
        # try:
        connectionSocket.send('1'.encode())
        username = connectionSocket.recv(1024).decode()
        print(username)
        user_database[username] = [username, connectionSocket, addr]
        join_msg = f"{user_database[username][0]} connected."
        print(join_msg)
        dm_msg(join_msg, username)
        user_msgs_thread = threading.Thread(target=user_sent_msgs, args=(username,))
        user_msgs_thread.daemon = True
        user_msgs_thread.start()
        # except:
        # connectionSocket.send('-1'.encode())
        # connectionSocket.close()

user_database = {}
hostname = '192.168.0.100'
port = 61965

server_sock = socket(AF_INET, SOCK_STREAM)
server_sock.bind((hostname, port))
print('listening on 192.168.0.100:63811/tcp')
server_sock.listen(1)
print("Server started on part {}. Accepting connections".format(port))
join_thread = threading.Thread(target=user_joined_msgs, args=(server_sock,))
join_thread.start()

