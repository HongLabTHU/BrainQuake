#! /usr/bin/python3.7
# -- coding: utf-8 -- **

import sys
import socket
import os
import multiprocessing
import task_utils
import utils_scs

HEADERSIZE = 10
SEPARATOR = '<SEPARATOR>'
BUFFER_SIZE = 4096
host = '0.0.0.0'
port = 6669
FILEPATH = '/usr/local/freesurfer/subjects'
FILEPATH1 = '/home/hello/reconModule_test/testCS/data/recv/fslresults'

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.bind((socket.gethostname(), 1241))
s.bind((host, port))
s.listen(5)
task_flag = 0 # whether a task is on
fs_flag = 0 # whether a task is done

while True:
    # Now we are listening on port 6669.
    clientsocket, address = s.accept()
    print(f'Connection from {address} has been established.')
    
    while True:
        # receive a task request: new task or check
        task = utils_scs.text_recv(clientsocket)
        print(f'task: {task}')

        if task == '10' or task == '11': # task 1: reconstruction
            p1 = multiprocessing.Process(target=task_utils.recv_a_t1, args=(clientsocket, task,))
            p1.start()
            # p1.join()
            break
        elif task == '12': # task 12: CT file upload
            p2 = multiprocessing.Process(target=task_utils.recv_a_ct, args=(clientsocket,))
            p2.start()
            # p2.join()
            break
        elif task == '13': # task 13: download fslresults
            p3 = multiprocessing.Process(target=task_utils.send_fsls, args=(clientsocket,))
            p3.start()
            # p3.join()
            break
        elif task == '2': # task 2: check
            p4 = multiprocessing.Process(target=task_utils.check_recon, args=(clientsocket,))
            p4.start()
            # p4.join()
            break
        elif task == '3': # task 3: download recon
            p5 = multiprocessing.Process(target=task_utils.send_recon, args=(clientsocket,))
            p5.start()
            # p5.join()
            break