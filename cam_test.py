#%% python3 faceCam_dnn.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import argparse
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
import time

import cv2 as cv

# print(f'cv version {cv.__version__}')
print(f'tf 버전: {tf.__version__}')
print("즉시 실행 모드: ", tf.executing_eagerly())
print("GPU ", "사용 가능" if tf.config.experimental.list_physical_devices("GPU") else "사용 불가능")

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

print('util modeules load ok')


# %%
print('start load model')
start_tick = time.time()
# model_name = 'ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
model_name = 'ssdlite_mobilenet_v2_coco_2018_05_09'
# model_name = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
model_dir = f"./data/{model_name}/saved_model"
model = tf.saved_model.load(str(model_dir))
model = model.signatures['serving_default']
print(f'{model_name} load ok , time to load {time.time() - start_tick}')
# print(model.inputs)
category_index = label_map_util.create_category_index_from_labelmap('./data/mscoco_label_map.pbtxt', use_display_name=True)
# print(category_index)
print('label load ok')


#%%
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

# cam ver1 size : 2592,1944
ap.add_argument("-v", "--videodevice", type=int,default=0, help="video device id default(0)")
ap.add_argument("--videoWidth", type=int,default=192*4, help="video width")
ap.add_argument("--videoHeight", type=int,default=108*4, help="video height")
# ap.add_argument("--videoWidth", type=int,default=320, help="video width")
# ap.add_argument("--videoHeight", type=int,default=240, help="video height")

args = vars(ap.parse_args())

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

#%%
while(True) :
    # time.sleep(1)

    ret,image_np = cap.read()
    time.sleep(1)

    # image_path = '../../tf_api/test.png'
    # print('load image')
    # image_np = np.array(Image.open(image_path))
    # print(image_np.shape)
    # print(type( image_np))
    # print(f'{image_path} load ok')

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]

    # Run inference
    print('start inference')
    start_tick = time.time()
    output_dict = model(input_tensor)
    print(f'end inference { time.time() - start_tick}')

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    print( f'num detection {num_detections}' )

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    
    _img_temp = image_np.copy()
    vis_util.visualize_boxes_and_labels_on_image_array(
        _img_temp,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        min_score_thresh = 0.3,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)
        
    cv.imshow("Face detector from camera stream", _img_temp)

    _k = cv.waitKey(1) & 0xff
    if _k == 27 : break

    time.sleep(1)


cap.release()
cv.destroyAllWindows()


# %%
