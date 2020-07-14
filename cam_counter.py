# %% python3 faceCam_dnn.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import argparse
import time
from threading import Thread

import socket
from struct import *

import argparse

parser = argparse.ArgumentParser(description="camera counter sample")

parser.add_argument('--port', type=int, 
    default=20105,
    help='help : udp port')
parser.add_argument('--model_path', type=str, 
    default='./',
    help='help : model data path')
parser.add_argument('--width', type=int, 
    default=1296,
    help='help : camera width')
parser.add_argument('--height', type=int, 
    default=972,
    help='help : camera height')


_args = parser.parse_args()

_port = _args.port
_model_path = _args.model_path
_camW = _args.width
_camH = _args.height


g_personCount = 1

def personCounter(_w,_h,model_path) :
    global g_personCount

    print(f'start counter thred cap size :{_w},{_h} , model path : {model_path}')

    import tensorflow as tf
    import zipfile

    from collections import defaultdict
    from io import StringIO

    from picamera.array import PiRGBArray
    from picamera import PiCamera

    from object_detection.utils import visualization_utils as vis_util
    from object_detection.utils import label_map_util
    from object_detection.utils import ops as utils_ops

    # print(f'cv version {cv.__version__}')
    print(f'tf 버전: {tf.__version__}')
    print("즉시 실행 모드: ", tf.executing_eagerly())
    print("GPU ", "사용 가능" if tf.config.experimental.list_physical_devices(
        "GPU") else "사용 불가능")
    print('util modeules load ok')

    model_name = 'ssdlite_mobilenet_v2_coco_2018_05_09'
    # model_name = 'ssd_mobilenet_v2_oid_v4_2018_12_12'

    print(f'start load model : {model_name}')
    start_tick = time.time()

    model_dir = f"{model_path}{model_name}/saved_model"
    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']
    print(f'{model_name} load ok , time to load {time.time() - start_tick}')
    # print(model.inputs)
    category_index = label_map_util.create_category_index_from_labelmap(f'{model_path}mscoco_label_map.pbtxt', use_display_name=True)
    # print(category_index)
    print('label load ok')

    # %%
    # 2592x1944
    IM_WIDTH = _w
    IM_HEIGHT = _h
    # Initialize Picamera and grab reference to the raw capture
    camera = PiCamera()
    camera.resolution = (IM_WIDTH, IM_HEIGHT)
    camera.framerate = 2
    rawCapture = PiRGBArray(camera, size=(IM_WIDTH, IM_HEIGHT))
    rawCapture.truncate(0)


    for frame1 in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

        # t1 = cv.getTickCount()

        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        image_np = np.copy(frame1.array)

        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image_np)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis,...]

        # Run inference
        # print('start inference')
        # start_tick = time.time()
        output_dict = model(input_tensor)
        # print(f'end inference { time.time() - start_tick}')

        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key:value[0, :num_detections].numpy() for key,value in output_dict.items()}
        output_dict['num_detections'] = num_detections

        # print( f'num detection {num_detections}' )

        # detection_classes should be ints.
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

        _result = [ _v for _v in zip(output_dict['detection_classes'],output_dict['detection_scores']) if _v[0] == 1   ]

        g_personCount = len(_result)
        time.sleep(0.25)

        # print( _result )
        rawCapture.truncate(0)

    camera.close()

def __testCounter(_w,_h) :
    global g_personCount

    while True :
        g_personCount += 1
        time.sleep(3)


if __name__ == '__main__':

    _th_counter = Thread(target=personCounter, args=(_camW,_camH,_model_path))
    _th_counter.daemon = True
    _th_counter.start()


    udp_socket = socket.socket(
        socket.AF_INET, #internet
        socket.SOCK_DGRAM # udp
        )
    
    udp_socket.bind(('',_port))

    print(f'start port : {_port}')

    while True:
        # print('wait packet...')
        _data, _rinfo = udp_socket.recvfrom(1024) # buffer size is 1024 bytes
        _packet = unpack("<BBBB",_data)
        # print(_packet)
        _header = _packet[0]
        if _header == 0x7f: # 인식된 인원수 요청
            _res = pack('<BBBB',0x7f,_packet[1],0,0)
            if _packet[1] == 10 :
                _res = pack('<BBBB',0x7f,_packet[1],g_personCount,0)
            udp_socket.sendto(_res,(_rinfo[0],_rinfo[1]))


