import subprocess
import shlex
import time

def run_multiple_commands(commands, logpaths, metalogpath):
    assert len(commands) == len(logpaths), 'Number of commands must match number of log paths'
    logfiles = []
    processes = []
    try:
        metalogfile = open(metalogpath, 'w', buffering=1)
        for command, logpath in zip(commands, logpaths):
            logfile = open(logpath, 'w', buffering=1)
            logfiles.append(logfile)
            p = subprocess.Popen(shlex.split(command), stdout=logfile)
            processes.append(p)
            print(f'{command}\n=> {logpath}', file=metalogfile)

        while len(processes) > 0:
            terminated = []
            for i in range(len(processes)):
                return_code = processes[i].poll()
                if return_code is not None:
                    terminated.append(i)
            for i in reversed(terminated):
                print('TERMINATED: {}'.format(commands[i]), file=metalogfile)
                del processes[i]
                del commands[i]
                del logpaths[i]
                logfiles[i].close()
                del logfiles[i]
            time.sleep(5)
        print('All processes have terminated. Quitting...')
    finally:
        for file in logfiles:
            file.close()
