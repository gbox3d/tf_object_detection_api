# %% python3 faceCam_dnn.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel
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
# from matplotlib import pyplot as plt
# from PIL import Image
# from IPython.display import display
import time

# import cv2 as cv

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

#%%
model_name = 'ssdlite_mobilenet_v2_coco_2018_05_09'
# model_name = 'ssd_mobilenet_v2_oid_v4_2018_12_12'

print(f'start load model : {model_name}')
start_tick = time.time()

model_dir = f"./data/{model_name}/saved_model"
model = tf.saved_model.load(str(model_dir))
model = model.signatures['serving_default']
print(f'{model_name} load ok , time to load {time.time() - start_tick}')
# print(model.inputs)
category_index = label_map_util.create_category_index_from_labelmap('./data/mscoco_label_map.pbtxt', use_display_name=True)
# print(category_index)
print('label load ok')

# %%
# 2592x1944
IM_WIDTH = 2592
IM_HEIGHT = 1944
# Initialize Picamera and grab reference to the raw capture
camera = PiCamera()
camera.resolution = (IM_WIDTH, IM_HEIGHT)
camera.framerate = 2
rawCapture = PiRGBArray(camera, size=(IM_WIDTH, IM_HEIGHT))
rawCapture.truncate(0)

# Initialize frame rate calculation
# frame_rate_calc = 1
# freq = cv.getTickFrequency()
# font = cv.FONT_HERSHEY_SIMPLEX

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
    print('start inference')
    start_tick = time.time()
    output_dict = model(input_tensor)
    print(f'end inference { time.time() - start_tick}')

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # print( f'num detection {num_detections}' )

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    _result = [ _v for _v in zip(output_dict['detection_classes'],output_dict['detection_scores']) if _v[0] == 1   ]

    print( _result )



    
    # # _img_temp = image_np.copy()
    # vis_util.visualize_boxes_and_labels_on_image_array(
    #     image_np,
    #     output_dict['detection_boxes'],
    #     output_dict['detection_classes'],
    #     output_dict['detection_scores'],
    #     category_index,
    #     min_score_thresh = 0.3,
    #     instance_masks=output_dict.get('detection_masks_reframed', None),
    #     use_normalized_coordinates=True,
    #     line_thickness=8)
    # frame.setflags(write=1)
    # frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # frame_expanded = np.expand_dims(frame_rgb, axis=0)

    # cv.putText(image_np, "FPS: {0:.2f}".format(
    #     frame_rate_calc), (30, 50), font, 1, (255, 255, 0), 2, cv.LINE_AA)

    # image_np.resize

    
    # cv.imshow("Face detector from camera stream", cv.resize(image_np,dsize=(320,240)))

    # # All the results have been drawn on the frame, so it's time to display it.
    # cv.imshow('Object detector', frame)

    # t2 = cv.getTickCount()
    # time1 = (t2-t1)/freq
    # frame_rate_calc = 1/time1

    # Press 'q' to quit
    # if cv.waitKey(1) == ord('q'):
            # break

    rawCapture.truncate(0)

camera.close()
# cv.destroyAllWindows()


# %%
