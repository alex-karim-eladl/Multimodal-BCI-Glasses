import subprocess
import time

blueberry1 = '3E8847D9-3D52-7A2F-B913-2FBD9F63ECF3'
blueberry2 = 'CAC8BEEB-CDC7-ED2F-04D7-48A3A88C2614'
blueberry3 = '737E96F8-345B-0B18-66D0-B72FF7301397'

subject = 'nat'
runname = '0'
try:
    p1 = subprocess.Popen('python3 record.py -a ' + blueberry1 + ' -u ' + subject + ' -r ' + runname, shell=True)
    p2 = subprocess.Popen('python3 record.py -a ' + blueberry2 + ' -u ' + subject + ' -r ' + runname, shell=True)
    p3 = subprocess.Popen('python3 record.py -a ' + blueberry3 + ' -u ' + subject + ' -r ' + runname, shell=True)
    # (output, err) = p.communicate()  This makes the wait possible     p_status = p.wait()
    time.sleep(60)
except KeyboardInterrupt:
    raise Exception('User stopped program.')

# while True:
#     pass
