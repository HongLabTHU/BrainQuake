#! /usr/bin/python3.7
# -- coding: utf-8 -- **

import os
import time
import multiprocessing
import eePipeline

Filename1 = 'task_log.txt'
Filename2 = 'task_done.txt'
FILEPATH = os.path.join(os.getcwd(), 'data', 'recv') # '/home/hello/reconModule_test/testCS/data/recv/'
FILEPATH2 = os.path.join(os.getcwd(), 'data', 'recon') # '/home/hello/reconModule_test/testCS/data/recon/'
SUBJECTS_DIR = os.getenv('SUBJECTS_DIR') # '/usr/local/freesurfer/subjects'
FASTPATH = '/home/hello/Downloads/labServer/FastSurfer-master'
CHECKTIME = 10

Filepath = os.path.join(FILEPATH, Filename1) # '/home/hello/reconModule_test/testCS/data/recv/task_log.txt'
Filepath2 = os.path.join(FILEPATH, Filename2) # '/home/hello/reconModule_test/testCS/data/recv/task_done.txt'

def task_log(req, num=None, name=None, hospital=None, reconType=None, state=None, info=None):
    """A function to analysis a request either from a client, 
       a freesurfer task or polling, assigning the following tasks for them.
    
        Parameter
        ---------
        req: str
            a complete log of request, 
            looks like: f"{#xxxx} {Name} {Hospital} {reconType} {state} {info}"
        num : str, optional
            to carry the object number '#xxxx' of the request
        name : str, optional
            the patient name
        hospital : str, optional
            the hospital name, where the patient comes from
        reconType: str, optional
            reconstruction type, either 'recon-all' or 'fast-surfer'
        state: str, optional
            'wait', 'running' or 'done'
        info: float, optional
            progress, from 0 to 1
        
        Output
        ------
        logs : str
            the log line(s) picked
        i : int
            how many logs are picked
        new_flag : int
            whether a new task is found
        log : str
            the log line picked
    """
    if req == "client":
        logs, i = read_a_log(num, name, hospital)
        if i > 0: # we've found sth.
            print(f"Here are what we've checked!")
            print(logs)
            return logs, i
        else: # find nothing
            ## here we should ask for the patient's data from either client or dataset.
            print(f"Patient {num} {name} from {hospital} has not been found! Check out your input or please upload the patient's data! [y/n]")
            logs = f"Patient {num} {name} from {hospital} has not been found! Check out your input or please upload the patient's data!"
            # reply = input()
            # if reply == "y":
            #     add_a_log(name, hospital)
            #     logs, i = read_a_log(name, hospital)
            #     print(f"Patient {name} from {hospital} has been added to the task line!")
            #     print(f"Here are what we've checked!")
            #     print(logs)
            return logs, i
    
    elif req == "freesurfer":
        if state == "finished":
            write_a_log(num, name, hospital, reconType, state, info)
            print(f"{req} has finished a task {num}!")
            # new_flag, log = pick_a_log(num)
            # if new_flag:
            #     print(f"A task {log} will be sent to {req} program!")
            #     ## call recon.py !!!
            #     num_next = log.split(" ")[0]
            #     write_a_log(num_next, state="running", info=0)
            # else:
            #     print(f"No new task is available!")
            # return new_flag, log
            return
        else: # state == "running"
            write_a_log(num, name, hospital, reconType, state, info)
            print(f"Log {num} has been updated!")
            return
    
    elif req == "polling":
        new_flag, log = pick_a_log(num)
        if new_flag:
            print(f"A task {log} has been detected by {req}!")
            # Here we should call a freesurfer program!
            return new_flag, log
        else:
            print(f"No task has been detected by {req}!")
            return new_flag, log
    
    else:
        raise IOError('Error: An unidentified request!')
    
def read_a_log(num=None, name=None, hospital=None):
    """A function to open the task log and read the log content

    Parameter
    ---------
    num : str, optional
        to carry the object number '#xxxx' of the request
    name : str, optional
        the patient name
    hospital : str, optional
        the hospital name, where the patient comes from
    
    Output
    ------
    log_read : str
        the log line(s) picked
    i : int
        how many logs are returned
    """
    if (num == None and name == None and hospital == None):
        raise IOError('Please type in at least one piece of info about the patient(s) you ask!')

    else:
        f = open(Filepath, 'r')
        lines = f.readlines()
        log_read = ""
        i = 0
        for line in lines:
            if (hospital == None) or (hospital in line):
                if (name == None) or (name in line):
                    if (num == None) or (num in line):
                        log_read += line
                        i = i + 1
        f.close()
    # print(log_read)
    return log_read, i
    
def write_a_log(num=None, name=None, hospital=None, reconType=None, state=None, info=None):
    """A function to write or update a task log content
    
    Parameter
    ---------
    num : str, optional
        to carry the object number '#xxxx' of the request
    name : str, optional
        the patient name
    hospital : str, optional
        the hospital name, where the patient comes from
    state : str, optional
        the tast state, "finished", "running" or "wait"
    info : float, optional
        a number btw 0 and 1 to carry the progress rate of the request
    """
    if (num == None and name == None):
        raise IOError('Please at least type in a patient\'s name or number!')

    else:
        f = open(Filepath, 'r+')
        lines = f.readlines()
        # print(lines)
        i = 0
        for line in lines:
            i = i + 1
            if (name == None) or (name in line):
                if (num == None) or (num in line):
                    parts = line.split(' ')
                    parts[3] = reconType
                    parts[4] = state
                    parts[5] = str(info)+"\n"
                    lines[i-1] = " ".join(parts)
        # print(lines)
        f.seek(0)
        f.truncate() # Here we need an improvement!
        f.writelines(lines)
        f.close()
    return

def add_a_log(name, hospital, reconType):
    """A function to add a new task to the log

    Parameter
    ---------
    name : str
        a new patient name
    hospital : str
        the hospital where the patient comes from
    reconType: str
        recon-all or fast-surfer
    """
    if (name == None or hospital == None):
        raise IOError('Please type in both the patient name and the hospital name!')

    else:
        f = open(Filepath, 'r+')
        number_max = f.readlines()[-1].split(' ')[0]
        # print(number_max)
        num_val = int(number_max[1:]) + 1
        # print(num)
        number_new = "#" + "%04d" % num_val
        # print(number_new)
        line_new = "\n" + " ".join([number_new, name, hospital, reconType, "wait", "0"])
        # print(line_new)
        f.write(line_new)
        f.close()
    return number_new

def pick_a_log(num):
    """A function to check polling and if available, pick the first waiting task and return its number

    Parameter
    ---------
    num : str
        to carry the object number '#xxxx' of the request, after which we check the following states

    Output
    ------
    log_read : str
        if available, the next task to start
    new_flag : bool
        whether there is a new task to start

    """
    num_val = int(num[1:]) + 1
    number_next = "#" + "%04d" % num_val

    f = open(Filepath, 'r+')
    lines = f.readlines()
    i = 0
    for line in lines:
        if not (number_next in line): # the finished task is already the last one
            new_flag = 0
            log_read = ""
        else: # the next task is in the line
            parts = line.split(" ")
            if parts[4] == "wait":
                if i < 3:
                    new_flag = 1
                    log_read = line
                    break
                else:
                    new_flag = 0
                    log_read = ""
                    break
            else:
                if parts[4] == "running":
                    i = i + 1
                num_val = int(number_next[1:]) + 1
                number_next = "#" + "%04d" % num_val
    return new_flag, log_read

def write_to_done(req, num, name, hospital, reconType, state, info):
    """A function to write or update a task log content
    
    Parameter
    ---------
    num : str, optional
        to carry the object number '#xxxx' of the request
    name : str, optional
        the patient name
    hospital : str, optional
        the hospital name, where the patient comes from
    state : str, optional
        the tast state, "finished", "running" or "wait"
    info : float, optional
        a number btw 0 and 1 to carry the progress rate of the request
    """
    if (num == None and name == None):
        raise IOError('Please at least type in a patient\'s name or number!')

    else:
        f = open(Filepath2, 'a+')
        line = '\n' + num + ' ' + name + ' ' + hospital + ' ' + reconType + ' ' + state + ' ' + str(info)
        f.write(line)
        f.close()
    return

def divide_a_log(log):
    parts = log.split(" ")
    num = parts[0]
    name = parts[1]
    hospital = parts[2]
    reconType = parts[3]
    state = parts[4]
    info = parts[5]
    if num == "None":
        num = None
    if name == "None":
        name = None
    if hospital == "None":
        hospital = None
    if reconType == "None":
        reconType = None
    if state == "None":
        state = None
    if info == "None":
        info = None
    return num, name, hospital, reconType, state, info

def write_a_fastcmd(log):
    num, name, hospital, reconType, state, info = divide_a_log(log)
    filename = str(name)
    filepath = os.path.join(FILEPATH, filename, f"{filename}T1.nii.gz") # FILEPATH + str(name) + 'T1.nii.gz'
    cmd = f"cd {FASTPATH} && ./run_fastsurfer.sh --t1 {filepath} --sid {filename}fast --sd $SUBJECTS_DIR --parallel --threads 8 --py python3.7 --surfreg >{filename}fast.log 2>&1"
    # cmd = f"python test.py"
    return cmd
    
def write_a_freecmd(log):
    num, name, hospital, reconType, state, info = divide_a_log(log)
    filename = str(name)
    filepath = os.path.join(FILEPATH, filename, f"{filename}T1.nii.gz") # FILEPATH + str(name) + 'T1.nii.gz'
    cmd = f"recon-all -i {filepath} -s {filename} -all -parallel -openmp 8 >{filename}.log 2>&1"
    # cmd = f"python test.py"
    return cmd
    
def write_a_infantcmd(log):
    num, name, hospital, reconType, state, info = divide_a_log(log)
    filename = str(name)
    filepath = os.path.join(FILEPATH, filename, f"{filename}T1.nii.gz") # FILEPATH + str(name) + 'T1.nii.gz'
    # infant_age = str(age)
    cmd = f"infant_recon_all --s {filename}"
    return cmd

def reconrun(cmd, num, name, hospital, reconType):
    """
    Print the command and execute a command string on the shell (on bash).
    
    Parameters
    ----------
    cmd : str
        Command to be sent to the shell.
    """
    assert reconType=='recon-all', 'Wrong reconType!'
    
    # unzip cmd
    cdir = os.path.join(os.getcwd(), 'data', 'recv', name)
    if os.path.isfile(os.path.join(cdir, f"{name}.zip")):
        if not os.path.isfile(os.path.join(cdir, f"{name}CT.nii.gz")):
            cmd_unzip = f"unzip {cdir}/{name}.zip -d {cdir}"
            print(cmd_unzip)
            os.system(cmd_unzip)
    fdir = os.path.join(cdir, 'fslresults')
    if not os.path.isdir(fdir):
        os.system(f"mkdir {fdir}")
    
    # run recon-all
    print(f"Running shell command: {cmd}")
    task_log(req='freesurfer', num=num, reconType='recon-all', state='running', info=0)
    os.system(cmd)
    
    # run supplementary cmds
    # cmd3 = f"mris_convert --combinesurfs /usr/local/freesurfer/subjects/{name}/surf/lh.pial /usr/local/freesurfer/subjects/{name}/surf/rh.pial /usr/local/freesurfer/subjects/{name}/{name}.stl"
    # print(cmd3)
    # os.system(cmd3)
    # cmd4 = f"mne watershed_bem -s {name} -o"
    # print(cmd4)
    # os.system(cmd4)
    cmd_mri_convert = f"mri_convert {SUBJECTS_DIR}/{name}/mri/orig.mgz {SUBJECTS_DIR}/{name}/mri/orig.nii.gz"
    print(cmd_mri_convert)
    os.system(cmd_mri_convert)

    cmd_mri_binarize = f"mri_binarize --i {SUBJECTS_DIR}/{name}/mri/brainmask.mgz --o {SUBJECTS_DIR}/{name}/mri/mask.mgz --min 1"
    print(cmd_mri_binarize)
    os.system(cmd_mri_binarize)
    
    cmd_label_convert_rh = f"mri_annotation2label --subject {name} --hemi rh --outdir {SUBJECTS_DIR}/{name}/label"
    print(cmd_label_convert_rh)
    os.system(cmd_label_convert_rh)

    cmd_label_convert_lh = f"mri_annotation2label --subject {name} --hemi lh --outdir {SUBJECTS_DIR}/{name}/label"
    print(cmd_label_convert_lh)
    os.system(cmd_label_convert_lh)
    
    cmd_fslfolder = f"mkdir {SUBJECTS_DIR}/{name}/fslresults"
    print(cmd_fslfolder)
    os.system(cmd_fslfolder)
    
    cmd_register = f"flirt -in {cdir}/{name}CT.nii.gz -ref {SUBJECTS_DIR}/{name}/mri/orig.nii.gz -out {SUBJECTS_DIR}/{name}/fslresults/{name}CT_Reg.nii.gz -cost normmi -dof 12"
    print(cmd_register)
    os.system(cmd_register)
    
    task_log(req='freesurfer', num=num, reconType='recon-all', state='finished', info=1)
    write_to_done(req='freesurfer', num=num, name=name, hospital=hospital, reconType='recon-all', state='finished', info=1)
    cmd1 = f"cd {SUBJECTS_DIR} && zip -rq {name}.zip {name}"
    print(cmd1)
    os.system(cmd1)
    # cmd2 = f"scp {SUBJECTS_DIR}/{name}.zip {FILEPATH2}"
    # os.system(cmd2)
    return
    
def fastrun(cmd, num, name, hospital, reconType):
    """
    Print the fast-recon command and execute a command string on the shell (on bash).
    
    Parameters
    ----------
    cmd : str
        Command to be sent to the shell.
    """
    assert reconType=='fast-surfer', 'Wrong reconType!'
    print(f"Running shell command: {cmd}")
    task_log(req='freesurfer', num=num, reconType='fast-surfer', state='running', info=0)
    # os.system(cmd)
    
    task_log(req='freesurfer', num=num, reconType='fast-surfer', state='finished', info=1)
    write_to_done(req='freesurfer', num=num, name=name, hospital=hospital, reconType='fast-surfer', state='finished', info=1)
    cmd1 = f"cd {SUBJECTS_DIR} && zip -r {name}fast.zip {name}fast"
    print(cmd1)
    # os.system(cmd1)
    return
    
def infantrun(cmd, num, name, hospital, reconType):
    print('infant function to be waiting...')

    task_log(req='freesurfer', num=num, reconType='infant-surfer', state='running', info=0)
    # os.system(cmd)

    task_log(req='freesurfer', num=num, reconType='infant-surfer', state='finished', info=1)
    write_to_done(req='freesurfer', num=num, name=name, hospital=hospital, reconType='infant-surfer', state='finished', info=1)
    cmd1 = f"cd {SUBJECTS_DIR} && zip -r {name}fast.zip {name}fast"
    print(cmd1)
    # os.system(cmd1)
    return

#def estimate(num, name, hospital, state, info):
#    while True:
#        time.sleep(2*CHECKTIME)
#        # receive a signal from a starting recon.py
#        f = open('recon-all-status.log', 'r')
#        lines = f.readlines()
#        if (lines[-1].split(' ')[3] == 'finished'):
#            # Here the task has finished
#            fin = 1
#            task_log(req='freesurfer', num=num, state='finished', info=1)
#            write_to_done(req='freesurfer', num=num, name=name, hospital=hospital, state='finished', info=1)
#            break
#        else:
#            start_time = lines[1].split(' ')[-5:-1]
#            last_time = lines[-1].split(' ')[-5:-1]
#            print(start_time)
#            print(last_time)
#            # calculate the time rest
#            info = 0.88
#            task_log(req='freesurfer', num=num, state='running', info=info)
#
#    return

def write_a_registercmd(name):
    # num, name, hospital, reconType, state, info = divide_a_log(log)
    filename = str(name)
    filepath1 = f"{FILEPATH}{name}/" + str(name) + 'T1.nii.gz'
    filepath2 = f"{FILEPATH}{name}/" + str(name) + 'CT.nii.gz'
    cmd = f"nohup recon-all -i {filepath1} -s {filename} -all -parallel -openmp 8 >{filename}.log 2>&1"
    # cmd = f"python test.py"
    return cmd

def registerrun(name):
    eePipeline.eep(name)
    return
