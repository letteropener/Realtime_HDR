from multiprocessing import Process, Queue
import cv2
from run_CCRF_mono import parallel_CCRF
import numpy as np
import time

######
# A simple script showing how multiprocessing.Queue works
# 
#The main process captures frames using default camera, 
#then send the frame to child process using a queue(name 'to_proc' in main process)
#the child process gets a frame, and convert it to grayscale before sending it back to the main process 
#using another queue (named 'from_proc' in main process)
#
#The main process then displays the grayscale frame and handles ui events
#
#When the capture is stopped by key 'q', the main process send a `None` to the child process to indicate "It's time to stop"
#
#************** #IMPORTANT# *********************
# Before joining the child process, make sure all queues are properly emptied. Otherwise it will cause a DEADLOCK. In this example, the main process gets
# every item left in the 'from_proc' queue using a while loop. Once the queues are empty, it is safe to let all process join.

def f(in_queue,out_queue):
    while True:
        try:
            v=in_queue.get()
            f1 = np.zeros((300,400),np.uint8)
            f2 = np.zeros((300,400),np.uint8)
            time =
            result = parallel_CCRF(f1,f2)
            print("child block:\n",result)
            if v is None:
                break
            #print(result)
            #process v
            v=cv2.cvtColor(v,cv2.COLOR_BGR2GRAY)
            out_queue.put(v)
        except:
            continue

if __name__ == '__main__':
    to_proc = Queue()
    from_proc=Queue()
    p = Process(target=f, args=(to_proc,from_proc))
    p.start()
    cap=cv2.VideoCapture(0)
    
    while True:
        _,img=cap.read()
        to_proc.put(img)
        print("entered while loop")
        try:
            #print("main proc",from_proc.get())
            frame=from_proc.get()
        except:
            continue
        cv2.imshow("12",frame)
        if cv2.waitKey(1)&0xff==ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()
    to_proc.put(None)
    while from_proc.empty() is False:
        print("in loop")
        try:
            print(from_proc.get())
        except:
            continue

    p.join()