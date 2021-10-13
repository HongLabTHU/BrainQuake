#! /usr/bin/python3.7
# -- coding: utf-8 -- **

import sys
import socket
import time
import pickle
import os
import tqdm
import utils

HEADERSIZE = 10
SEPARATOR = '<SEPARATOR>'
BUFFER_SIZE = 4096
host = '0.0.0.0'
port = 6669
FILEPATH = '/home/hello/reconModule_test/testCS/data/recv'

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
    print(len(msg))
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
            new_msg = True
            full_msg = b''
            break
    return txt_recv

def file_recv(socket, reconType='recon-all'):
## receive a nifti file from the client
    # receive the file info
    received = socket.recv(BUFFER_SIZE).decode()
    filename, filesize = received.split(SEPARATOR)
    filename = os.path.basename(filename)
    # filepath = os.path.join('data', 'recv', filename)
    filesize = int(filesize)
    pat_name = filename.split('.')[0].split('T')[0]
    if os.path.exists(f"{FILEPATH}/{pat_name}"):
        filepath = os.path.join('data', 'recv', pat_name, filename)
    else:
        os.system(f"mkdir {FILEPATH}/{pat_name}")
        filepath = os.path.join('data', 'recv', pat_name, filename)

    # start receiving the file from the socket and writing to the file stream
    with open(filepath, "wb") as f:
        while True:
            # read bytes from the socket (receive)
            bytes_read = socket.recv(BUFFER_SIZE)
            if not bytes_read:    
                break
            f.write(bytes_read)
    f.close()
    number = utils.add_a_log(name=pat_name, hospital='Yuquan', reconType=reconType)
    socket.close()
    return number

def file_recvCT(socket):
    received = socket.recv(BUFFER_SIZE).decode()
    filename, filesize = received.split(SEPARATOR)
    filename = os.path.basename(filename)
    # filepath = os.path.join('data', 'recv', filename)
    filesize = int(filesize)
    pat_name = filename.split('.')[0].split('C')[0]
    if os.path.exists(f"{FILEPATH}/{pat_name}"):
        os.system(f"mkdir {FILEPATH}/{pat_name}/fslresults")
        filepath = os.path.join('data', 'recv', pat_name, filename)
    else:
        os.system(f"mkdir {FILEPATH}/{pat_name}")
        os.system(f"mkdir {FILEPATH}/{pat_name}/fslresults")
        filepath = os.path.join('data', 'recv', pat_name, filename)

    with open(filepath, "wb") as f:
        while True:
            bytes_read = socket.recv(BUFFER_SIZE)
            if not bytes_read:
                break
            f.write(bytes_read)
    f.close()
    socket.close()
    return pat_name

def file_send(filepath, socket):
    filename = filepath.split('/')[-1]
    filesize = os.path.getsize(filepath)
    socket.send(f'{filename}{SEPARATOR}{filesize}'.encode())

    # progress = tqdm.tqdm(range(filesize), f'Sending {filename}', unit="B", unit_scale=True, unit_divisor=1024)
    time.sleep(1)
    with open(filepath, "rb") as f:
        # for _ in progress:
        while True:
            # read the bytes from the file
            bytes_read = f.read(BUFFER_SIZE)
            if not bytes_read:
                break
            # we use sendall to assure transimission in 
            # busy networks
            socket.sendall(bytes_read)
            # update the progress bar
            # progress.update(len(bytes_read))
    socket.close()
