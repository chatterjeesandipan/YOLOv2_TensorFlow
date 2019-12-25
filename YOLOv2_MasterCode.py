from __future__ import absolute_import, print_function
from __future__ import unicode_literals, division
import warnings 
warnings.filterwarnings("ignore")

import os, sys, random, re
import math, glob, copy, cv2
import imgaug as ia
import time, datetime
import matplotlib.pyplot as plt
from easydict import EasyDict as edict

import numpy as np
import tensorflow as tf
print("Using Tensorflow version: ", tf.__version__)

### Importing local packages in the folder
from config import *
from Image_Label_Utils import *
from Dataset_Utils import *
from Data_Augmentation import *
from Process_YOLO_format import *
from Loss_function import *
from Create_Model import *
from Train_Function import *


### Bring in the model configuration file
mc = build_config()
os.chdir(mc.maindir)

### Even though we have a config file, still we can have some argument parsing to provide 
### some control to the user
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--project_name", required=False, default="Road", type=str, \
    help='Name for this project and dataset')
parser.add_argument("--image_width", required=False, default=512, type=int, \
    help='Width of the images used for training')
parser.add_argument("--image_height", required=False, default=512, type=int, \
    help='Height of the images used for training')
parser.add_argument("--grid_width", required=False, default=16, type=int, \
    help='Final width of the images after model ops')
parser.add_argument("--grid_height", required=False, default=16, type=int, \
    help='Final width of the images after model ops')
parser.add_argument("--epochs", required=False, default=100, type=int, \
    help='Number of epochs for training the model')
parser.add_argument("--train_batch_size", required=False, default=32, type=int, \
    help='Batch size for training')
parser.add_argument("--val_batch_size", required=False, default=16, type=int, \
    help='Batch size for validation')
parser.add_argument("--score_threshold", required=False, default=0.5, type=float, \
    help='Threshold for reporting a detected object class')
parser.add_argument("--iou_threshold", required=False, default=0.4, type=float, \
    help='Threshold for considering the box overlap during object localization')
parser.add_argument("--loss_coef_noobject", required=False, default=10, type=int, \
    help='Loss coefficient for not detecting an object in the image')
parser.add_argument("--loss_coef_object", required=False, default=10, type=int, \
    help='Loss coefficient for wrongly detecting an object in the image')
parser.add_argument("--loss_coef_class", required=False, default=5, type=int, \
    help='Loss coefficient for detecting the wrong object class in the image')
parser.add_argument("--loss_coef_coord", required=False, default=5, type=int, \
    help='Loss coefficient for mismatch between predicted and true bounding box coordinates')
parser.add_argument("--learning_rate", required=False, default=1e-4, type=float, \
    help='Initial learning rate for model training')
parser.add_argument("--model", required=False, default="YOLO", type=str, \
    help='Choose the object detection model type')
parser.add_argument("--resize_data", required=False, default="yes", type=str, \
    help='Resize image and bounding box data (yes/no), if data being reused')

args = parser.parse_args()
mc.PROJECT, mc.MODEL = args.project_name, args.model.upper()
assert mc.MODEL in ("VGG", "YOLO")
mc.IMAGE_W, mc.IMAGE_H = args.image_width, args.image_height
mc.GRID_W, mc.GRID_H = args.grid_width, args.grid_height
mc.EPOCHS, mc.LEARN_RATE = args.epochs, args.learning_rate
mc.TRAIN_BATCH_SIZE, mc.VAL_BATCH_SIZE = args.train_batch_size, args.val_batch_size
mc.LAMBDA_NOOBJECT, mc.LAMBDA_OBJECT = args.loss_coef_noobject, args.loss_coef_object
mc.LAMBDA_CLASS, mc.LAMBDA_COORD = args.loss_coef_class, args.loss_coef_coord

rawdata = [folder for folder in os.listdir() if folder == "OID"][0]
rawdata_dir = os.path.join(os.getcwd(), "OID")

### Creating the dataset for training and testing
### comment the line below if you have previously resized the training data
if args.resize_data in ("y", "yes", "Yes", "YES", "1"):
    image_bbox_resize_ops(mc, rawdata_dir)
else:
    pass

### Paths of Training and Testing images' and labels' folders
train_folder = os.path.abspath(os.path.join(mc.maindir, "Dataset", mc.PROJECT, "TRAIN"))
train_images, train_labels = os.path.join(train_folder, "Images"), os.path.join(train_folder, "Labels")
val_folder = os.path.abspath(os.path.join(mc.maindir, "Dataset", mc.PROJECT, "TEST"))
val_images, val_labels = os.path.join(val_folder, "Images"), os.path.join(val_folder, "Labels")

### Call Dataset_Utils.py to create a tensorflow dataset of training and validation data
train_dataset = None
train_dataset, labels_dict = get_dataset(train_images, train_labels, mc.TRAIN_BATCH_SIZE, name="Train")
val_dataset = None
val_dataset, labels_dict = get_dataset(val_images, val_labels, mc.VAL_BATCH_SIZE, name="Validation")

### Details about the labels in this training exercise
mc.LABELS = list(labels_dict.keys())
mc.CLASSES = len(mc.LABELS)
print("\n=================================================\n")
print("Current configuration: ")
for key, val in mc.items():
    print(key, "\t", val)
print("\n=================================================\n")

### Sample visualization from the training dataset
# visualize_image_bbox_from_dataset(train_dataset, labels_dict)

### data augmentation for training datasets (only)
aug_train_dataset = augment_dataset(mc, train_dataset)

#### convert data to YOLO format and initiate a training and testing generator
train_gen = ground_truth_generator(mc, aug_train_dataset, labels_dict)
val_gen = ground_truth_generator(mc, val_dataset, labels_dict)

### Define the object detection model
if mc.MODEL == "VGG":
    model = make_VGG_model(mc)
elif mc.MODEL == "YOLO":
    model = make_YOLO_model(mc)
else:
    raise TypeError("Only VGG and YOLO are valid inputs")
    sys.exit()

print(model.summary())

### TRAINING THE YOLOv2 MODEL
### Ensuring that each training run is unique with the help of a time tag
train_logname = "TRAIN_" + datetime.datetime.now().strftime("%b_%d_%Y_%H_%M")
LOGDIR = os.path.abspath(os.path.join(mc.maindir, "LOGS", mc.PROJECT, train_logname))
os.makedirs(LOGDIR, exist_ok=True)
results = train_fn(model, mc, train_gen, val_gen, train_folder, val_folder, train_logname, LOGDIR)

