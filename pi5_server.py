from socket import *
import threading
import json
import time
import queue

def dm_msg(msg, receiver):
    """Send a message to a registered user.

    If msg is a dict, it will be JSON-encoded. Strings are sent as-is.
    On failure the user is removed from `user_database`.
    """
    try:
        sock = user_database[receiver][1]
    except KeyError:
        return False

    try:
        if isinstance(msg, dict):
            payload = json.dumps(msg).encode('utf-8')
        elif isinstance(msg, bytes):
            payload = msg
        else:
            payload = str(msg).encode('utf-8')
        # add newline delimiter for simple receivers
        if not payload.endswith(b"\n"):
            payload += b"\n"
        sock.sendall(payload)
        return True
    except Exception:
        # remove dead connection
        try:
            sock.close()
        except Exception:
            pass
        for k, v in list(user_database.items()):
            if v[1] is sock:
                try:
                    del user_database[k]
                except Exception:
                    pass
        return False


# Incoming command queues per-user (populated by user_sent_msgs)
incoming_commands = {}

# Command handler registry - other modules (e.g. aicam.py) can register
# a callback: handler(msg_dict, sender_name). Handlers are invoked in their
# own thread so socket handling is not blocked.
command_handlers = []

def register_command_handler(handler):
    """Register a callback to be invoked for each incoming JSON command.

    Handler signature: handler(msg: dict, sender: str)
    """
    if callable(handler):
        command_handlers.append(handler)

def user_sent_msgs(receiver):
    # Per-user incoming command queues
    incoming_commands.setdefault(receiver, queue.Queue())
    sock = user_database[receiver][1]
    while True:
        try:
            data = sock.recv(4096)
            if not data:
                # connection closed
                try:
                    sock.close()
                except Exception:
                    pass
                try:
                    del user_database[receiver]
                except Exception:
                    pass
                break

            text = data.decode(errors='ignore').strip()
            if not text:
                continue

            try:
                msg = json.loads(text)
                print(msg)
                incoming_commands[receiver].put(msg)
                # invoke registered handlers in their own threads
                for h in list(command_handlers):
                    try:
                        t = threading.Thread(target=h, args=(msg, receiver), daemon=True)
                        t.start()
                    except Exception:
                        # ignore handler errors to keep socket loop running
                        continue
            except Exception:
                # legacy plain-text message
                print(f"[{receiver}] {text}")
        except Exception:
            time.sleep(0.1)
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

