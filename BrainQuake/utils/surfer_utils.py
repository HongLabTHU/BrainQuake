#! /usr/bin/python3.7
# -- coding: utf-8 -- **

import socket
import time
import pickle
import os
import tqdm

HEADERSIZE = 10
SEPARATOR = '<SEPARATOR>'
BUFFER_SIZE = 4096
# host = '166.111.152.123'
# port = 6669
Filepath = '.'

def create_socket(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # s.connect((socket.gethostname(), 1241))
    s.connect((host, port))
    return s

def text_send(socket, msg):
    msg = pickle.dumps(msg)
    # print(len(msg))
    msg = bytes(f'{len(msg):<{HEADERSIZE}}', 'utf-8') + msg
    # print(len(msg))
    if len(msg) < BUFFER_SIZE:
        msg += pickle.dumps('0' * (BUFFER_SIZE-len(msg)-10))
    elif len(msg) > BUFFER_SIZE:
        len_need = BUFFER_SIZE - len(msg)%BUFFER_SIZE - 10
        msg += pickle.dumps('0' * len_need)
    # print(len(msg))
    socket.send(msg)

def text_recv(socket):
## receive a text and print it out in the terminal
    full_msg = b''
    new_msg = True
    while True:
        msg = socket.recv(BUFFER_SIZE)
        if new_msg:
            # print("New message length:", msg[:HEADERSIZE])
            msglen = int(msg[:HEADERSIZE])
            new_msg = False
        
        full_msg += msg

        if len(full_msg) == (msglen//BUFFER_SIZE+1)*BUFFER_SIZE:
            # print("Full message received!")
            txt_recv = pickle.loads(full_msg[HEADERSIZE:msglen+HEADERSIZE])
            # print(pickle.loads(full_msg[HEADERSIZE:]))
            new_msg = True
            full_msg = b''
            break
    return txt_recv

def file_send(socket, pat_name):
## send a file and its file info to the server
    pat_filepath = os.path.join('data', 'send', pat_name+'.nii.gz')
    filename = pat_name+'.nii.gz'
    filesize = os.path.getsize(pat_filepath)
    socket.send(f'{filename}{SEPARATOR}{filesize}'.encode())

    progress = tqdm.tqdm(range(filesize), f'Sending {filename}', unit="B", unit_scale=True, unit_divisor=1024)
    with open(pat_filepath, "rb") as f:
        for _ in progress:
            # read the bytes from the file
            bytes_read = f.read(BUFFER_SIZE)
            if not bytes_read:
                # file transmitting is done
                # print(f"transmission completed")
                break
            # we use sendall to assure transimission in 
            # busy networks
            socket.sendall(bytes_read)
            # update the progress bar
            progress.update(len(bytes_read))
            # int(len(bytes_read)/filesize)
    socket.close()

def file_recv(socket):
## receive a gifti file from the server
    # receive the file info
    received = socket.recv(BUFFER_SIZE).decode()
    filename, filesize = received.split(SEPARATOR)
    # remove absolute path if there is
    filename = os.path.basename(filename)
    filepath = os.path.join(Filepath, 'data', 'down', filename)
    # convert to integer
    filesize = int(filesize)

    # start receiving the file from the socket and writing to the file stream
    progress = tqdm.tqdm(range(filesize), f'Receiving {filename}', unit="B", unit_scale=True, unit_divisor=1024)
    with open(filepath, "wb") as f:
        for _ in progress:
            # read bytes from the socket (receive)
            bytes_read = socket.recv(BUFFER_SIZE)
            if not bytes_read:    
                # nothing is received
                # file transmitting is done
                # print(f"transmission completed")
                break
            # write to the file the bytes we just received
            f.write(bytes_read)
            # update the progress bar
            progress.update(len(bytes_read))
    socket.close()
