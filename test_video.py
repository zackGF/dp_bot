
import time
from datetime import datetime

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import csv
import glob
import cv2 

ROOT_DIR = os.path.abspath("/home/zsy/mask_rcnn/Mask_RCNN")
sys.path.append(ROOT_DIR)
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn.config import Config
from mrcnn import visualize 

sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
from pycocotools.coco import COCO 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
print(COCO_MODEL_PATH)

#if not os.path.exists(COCO_MODEL_PATH):
#    utils.download_trained_weights(COCO_MODEL_PATH)
# Directory of images to run detection on
VIDEO_DIR = os.path.abspath("/home/zsy/mask_rcnn/videos")
VIDEO_SAVE_DIR = os.path.join(VIDEO_DIR, "out")


batch_size=1



class CocoConfig(Config):
    NAME = "coco"
    IMAGES_PER_GPU=1
    NUM_CLASSES = 81

class InferenceConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
config = InferenceConfig()
config.display()



#create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush']



def random_colors(N):
    np.random.seed(1)
    colors=[tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors

def apply_mask(image,mask,color,alpha=0.5):
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image

def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 1)

    return image



def make_video(outvid, images=None, fps=30, size=None,
               is_color=True, format="FMP4"):
    from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    #vid.release()
    return vid


capture= cv2.VideoCapture(os.path.join(VIDEO_DIR, 'test1.mp4'))
try:
  if not os.path.exists(VIDEO_SAVE_DIR):
    os.makedirs(VIDEO_SAVE_DIR)
except OSError:
  print ('Error: Creating directory of data')
frames = []
frame_count = 0
# these 2 lines can be removed if you dont have a 1080p camera.
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


while True:
  ret, frame = capture.read()
  # Bail out when the video file ends
  if not ret:
    break
        
  # Save each frame of the video to a list
  frame_count += 1
  frames.append(frame)
  print('frame_count :{0}'.format(frame_count))
  if len(frames) == batch_size:
    results = model.detect(frames, verbose=0)
    print('Predicted')
    for i, item in enumerate(zip(frames, results)):
      start=datetime.utcnow()

      frame = item[0]
      r = item[1]
      frame = display_instances(
          frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
      )
      name = '{0}.jpg'.format(frame_count + i - batch_size)
      name = os.path.join(VIDEO_SAVE_DIR, name)
      cv2.imwrite(name, frame)
      
      end = datetime.utcnow()
      c = end-start
      print('writing to file:{0}'.format(name), c.microseconds)
   # Clear the frames array to start the next batch
  frames = []

capture.release()
print('detect video Ok .................')

#images = list(glob.iglob(os.path.join(VIDEO_SAVE_DIR, '*.*')))
# Sort the images by integer index
#print(images)
#images = sorted(images, key=lambda x: float(os.path.split(x)[1][:-3]))

#outvid = os.path.join(VIDEO_DIR, "out.mp4")
#make_video(outvid, images, fps=30)
#print('make video success............')
