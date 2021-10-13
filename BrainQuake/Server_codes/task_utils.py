#! /usr/bin/python3.7
# -- coding: utf-8 -- **

import sys
import socket
import time
import pickle
import os
import tqdm
import multiprocessing
import utils
import utils_scs

HEADERSIZE = 10
SEPARATOR = '<SEPARATOR>'
BUFFER_SIZE = 4096
host = '0.0.0.0'
port = 6669
FILEPATH = '/usr/local/freesurfer/subjects'
FILEPATH1 = '/home/hello/reconModule_test/testCS/data/recv'

def recv_a_t1(clientsocket, task):
    task_flag = 1 # a task starts here
    fs_flag = 0 # a freesurfer recon task has not been completed
    # receive a T1 file
    if task == '10':
        reconType = f"recon-all"
        number = utils_scs.file_recv(clientsocket, reconType)
    elif task == '11':
        reconType = f"fast-surfer"
        number = utils_scs.file_recv(clientsocket, reconType)
    print('T1 file received')
    # here we read the log
    log, i = utils.read_a_log(num=number)
    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # s.bind((socket.gethostname(), 1241))
    s.bind((host, 6666))
    s.listen(5)
    clientsocket, address = s.accept()
    print(f'Connection from {address} has been established.')
    time.sleep(1)
    utils_scs.text_send(clientsocket, log)
    clientsocket.close()
    s.close()
    fs_flag = 1 # a freesurfer recon task has been completed
    return

def recv_a_ct(clientsocket):
    name = utils_scs.file_recvCT(clientsocket)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # s.bind((socket.gethostname(), 1241))
    s.bind((host, 6667))
    s.listen(5)
    clientsocket, address = s.accept()
    print(f'Connection from {address} has been established.')
    time.sleep(1)
    utils_scs.text_send(clientsocket, 'Uploaded!')
    clientsocket.close()
    s.close()
    print("Start registering...")
    p = multiprocessing.Process(target=utils.registerrun,args=(name,))
    p.start()
    p.join()
    return

def send_fsls(clientsocket):
    clientsocket.close()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # s.bind((socket.gethostname(), 1241))
    s.bind((host, 6668))
    s.listen(5)
    clientsocket, address = s.accept()
    print(f'Connection from {address} has been established.')
    time.sleep(1)
    patName = utils_scs.text_recv(clientsocket)
    print(patName)
    filepath = f"{FILEPATH1}/{patName}/fslresults/{patName}intracranial.nii.gz"
    print('sending...')
    utils_scs.file_send(filepath, clientsocket)
    print('Sent!')
    clientsocket.close()
    s.close()
    return

def check_recon(clientsocket):
    clientsocket.close()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # s.bind((socket.gethostname(), 1241))
    s.bind((host, 6665))
    s.listen(5)
    clientsocket, address = s.accept()
    print(f'Connection from {address} has been established.')
    time.sleep(1)
    check_log = utils_scs.text_recv(clientsocket)
    print(check_log)
    [num, name, hospital, state, info] = check_log.split(' ')
    if num == 'None':
        num = None
    if name == 'None':
        name = None
    logs, i = utils.task_log(req='client', num=num, name=name, hospital=hospital)
    print(logs)
    time.sleep(1)
    utils_scs.text_send(clientsocket, logs)
    time.sleep(2)
    clientsocket.close()
    s.close()
    return

def send_recon(clientsocket):
    clientsocket.close()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # s.bind((socket.gethostname(), 1241))
    s.bind((host, 6664))
    s.listen(5)
    clientsocket, address = s.accept()
    print(f'Connection from {address} has been established.')
    time.sleep(1)
    down_log = utils_scs.text_recv(clientsocket)
    print(down_log)
    for log in down_log:
        [num, name, hospital, reconType, state, info] = log.split(' ')
        filepath = f"{FILEPATH}/{name}.zip"
        print('sending...')
        utils_scs.file_send(filepath, clientsocket)
        print('Sent!')
    clientsocket.close()
    s.close()
    return