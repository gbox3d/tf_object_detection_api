#python3 faceCam_dnn.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel
import cv2 as cv
import sys 
import time

import numpy as np
import argparse

print(cv.__version__)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

# cam ver1 size : 2592,1944
ap.add_argument("-v", "--videodevice", type=int,default=0, help="video device id default(0)")
ap.add_argument("--videoWidth", type=int,default=192*4, help="video width")
ap.add_argument("--videoHeight", type=int,default=108*4, help="video height")
# ap.add_argument("-p", "--prototxt", required=True, help="path to Caffee 'deploy' prototxt file")
# ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
# ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")

args = vars(ap.parse_args())

# load our serialized model from disk
# print("[INFO] loading model...")
# net = cv.dnn.readNetFromCaffe(args["prototxt"], args["model"])

cap = cv.VideoCapture(args["videodevice"])

if cap.get(3) < 10 : 
    print('not found cam')
    exit()
else : 
    print(f'found cam : {cap.get(3),{cap.get(4)}}') 
    cap.set(3,args["videoWidth"])
    cap.set(4,args["videoHeight"])
    time.sleep(1)
    print(f'change resolution : {cap.get(3),{cap.get(4)}}') 


while(True) :
    # time.sleep(1)

    ret,frame = cap.read()
    
    cv.imshow("Face detector from camera stream", frame)

    _k = cv.waitKey(1) & 0xff
    if _k == 27 : break

cap.release()
cv.destroyAllWindows()
