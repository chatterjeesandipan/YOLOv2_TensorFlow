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
from tensorflow import keras
from tensorflow.keras.layers import Concatenate, concatenate, Dropout, Dense
from tensorflow.keras.layers import Reshape, Activation, Flatten
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, BatchNormalization

### Importing local packages in the folder
from config import *
from Image_Label_Utils import *
from Dataset_Utils import *
from Data_Augmentation import *
from Process_YOLO_format import *
from Loss_function import *
from Train_Function import *

### Define leakyReLu layer for Convolutions
LeakyReLu = keras.layers.LeakyReLU
K = keras.backend

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

args = parser.parse_args()
mc.PROJECT = args.project_name
mc.IMAGE_W, mc.IMAGE_H = args.image_width, args.image_height
mc.GRID_W, mc.GRID_H = args.grid_width, args.grid_height
mc.EPOCHS = args.epochs
mc.TRAIN_BATCH_SIZE, mc.VAL_BATCH_SIZE = args.train_batch_size, args.val_batch_size
mc.LAMBDA_NOOBJECT, mc.LAMBDA_OBJECT = args.loss_coef_noobject, args.loss_coef_object
mc.LAMBDA_CLASS, mc.LAMBDA_COORD = args.loss_coef_class, args.loss_coef_coord

rawdata = [folder for folder in os.listdir() if folder == "OID"][0]
rawdata_dir = os.path.join(os.getcwd(), "OID")

### Creating the dataset for training and testing
### comment the line below if you have previously resized the training data
image_bbox_resize_ops(mc, rawdata_dir)

### Paths of Training and Testing images' and labels' folders
train_folder = os.path.abspath(os.path.join(mc.maindir, "Dataset", mc.PROJECT, "TRAIN"))
train_images, train_labels = os.path.join(train_folder, "IMAGES"), os.path.join(train_folder, "LABELS")
val_folder = os.path.abspath(os.path.join(mc.maindir, "Dataset", mc.PROJECT, "TEST"))
val_images, val_labels = os.path.join(val_folder, "IMAGES"), os.path.join(val_folder, "LABELS")

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
train_gen = ground_truth_generator(config, aug_train_dataset, labels_dict)
val_gen = ground_truth_generator(config, val_dataset, labels_dict)

#### DEFINE the YOLOv2 model
class SpaceToDepth(keras.layers.Layer):

    def __init__(self, block_size, **kwargs):
        self.block_size = block_size
        super(SpaceToDepth, self).__init__(**kwargs)

    def call(self, inputs):
        x = inputs
        batch, height, width, depth = K.int_shape(x)
        batch = -1
        reduced_height = height // self.block_size
        reduced_width = width // self.block_size
        y = K.reshape(x, (batch, reduced_height, self.block_size,\
            reduced_width, self.block_size, depth))
        z = K.permute_dimensions(y, (0, 1, 3, 2, 4, 5))
        t = K.reshape(z, (batch, reduced_height, reduced_width, depth*self.block_size**2))
        return t

    def compute_output_shape(self, input_shape):
        shape = (input_shape[0], input_shape[1]//self.block_size, input_shape[2]//self.block_size,\
            input_shape[3]*self.block_size**2)
        return tf.TensorShape(shape)

def make_model():
## YOLO v2 MODEL'S Layers
    input_image = tf.keras.layers.Input((mc.IMAGE_W, mc.IMAGE_H, 3), dtype="float32")
    ## Layer 1
    x = Conv2D(32, (3,3), strides=(1,1), padding="same", name="conv_1", use_bias=False)(input_image)
    x = BatchNormalization(name="norm1")(x)
    x = LeakyReLu(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    ## Layer 2
    x = Conv2D(64, (3,3), strides=(1,1), padding="same", name="conv_2", use_bias=False)(x)
    x = BatchNormalization(name="norm2")(x)
    x = LeakyReLu(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    ## Layer 3
    x = Conv2D(128, (3,3), strides=(1,1), padding="same", name="conv_3", use_bias=False)(x)
    x = BatchNormalization(name="norm3")(x)
    x = LeakyReLu(alpha=0.1)(x)
    ## Layer 4
    x = Conv2D(64, (1,1), strides=(1,1), padding="same", name="conv_4", use_bias=False)(x)
    x = BatchNormalization(name="norm4")(x)
    x = LeakyReLu(alpha=0.1)(x)
    ## Layer 5
    x = Conv2D(128, (3,3), strides=(1,1), padding="same", name="conv_5", use_bias=False)(x)
    x = BatchNormalization(name="norm5")(x)
    x = LeakyReLu(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    ## Layer 6
    x = Conv2D(256, (3,3), strides=(1,1), padding="same", name="conv_6", use_bias=False)(x)
    x = BatchNormalization(name="norm6")(x)
    x = LeakyReLu(alpha=0.1)(x)
    ## Layer 7
    x = Conv2D(128, (1,1), strides=(1,1), padding="same", name="conv_7", use_bias=False)(x)
    x = BatchNormalization(name="norm7")(x)
    x = LeakyReLu(alpha=0.1)(x)
    ## Layer 8
    x = Conv2D(256, (3,3), strides=(1,1), padding="same", name="conv_8", use_bias=False)(x)
    x = BatchNormalization(name="norm8")(x)
    x = LeakyReLu(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    ## Layer 9
    x = Conv2D(512, (3,3), strides=(1,1), padding="same", name="conv_9", use_bias=False)(x)
    x = BatchNormalization(name="norm9")(x)
    x = LeakyReLu(alpha=0.1)(x)
    ## Layer 10
    x = Conv2D(256, (1,1), strides=(1,1), padding="same", name="conv_10", use_bias=False)(x)
    x = BatchNormalization(name="norm10")(x)
    x = LeakyReLu(alpha=0.1)(x)
    ## Layer 11
    x = Conv2D(512, (3,3), strides=(1,1), padding="same", name="conv_11", use_bias=False)(x)
    x = BatchNormalization(name="norm11")(x)
    x = LeakyReLu(alpha=0.1)(x)
    ## Layer 12
    x = Conv2D(256, (1,1), strides=(1,1), padding="same", name="conv_12", use_bias=False)(x)
    x = BatchNormalization(name="norm12")(x)
    x = LeakyReLu(alpha=0.1)(x)
    ## Layer 13
    x = Conv2D(512, (3,3), strides=(1,1), padding="same", name="conv_13", use_bias=False)(x)
    x = BatchNormalization(name="norm13")(x)
    x = LeakyReLu(alpha=0.1)(x)

    skip_connection = x
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)

    ## Layer 14
    x = Conv2D(1024, (3,3), strides=(1,1), padding="same", name="conv_14", use_bias=False)(x)
    x = BatchNormalization(name="norm14")(x)
    x = LeakyReLu(alpha=0.1)(x)
    ## Layer 15
    x = Conv2D(512, (1,1), strides=(1,1), padding="same", name="conv_15", use_bias=False)(x)
    x = BatchNormalization(name="norm15")(x)
    x = LeakyReLu(alpha=0.1)(x)
    ## Layer 16
    x = Conv2D(1024, (3,3), strides=(1,1), padding="same", name="conv_16", use_bias=False)(x)
    x = BatchNormalization(name="norm16")(x)
    x = LeakyReLu(alpha=0.1)(x)
    ## Layer 17
    x = Conv2D(512, (1,1), strides=(1,1), padding="same", name="conv_17", use_bias=False)(x)
    x = BatchNormalization(name="norm17")(x)
    x = LeakyReLu(alpha=0.1)(x)
    ## Layer 18
    x = Conv2D(1024, (3,3), strides=(1,1), padding="same", name="conv_18", use_bias=False)(x)
    x = BatchNormalization(name="norm18")(x)
    x = LeakyReLu(alpha=0.1)(x)
    ## Layer 19
    x = Conv2D(1024, (3,3), strides=(1,1), padding="same", name="conv_19", use_bias=False)(x)
    x = BatchNormalization(name="norm19")(x)
    x = LeakyReLu(alpha=0.1)(x)
    ## Layer 20
    x = Conv2D(1024, (3,3), strides=(1,1), padding="same", name="conv_20", use_bias=False)(x)
    x = BatchNormalization(name="norm20")(x)
    x = LeakyReLu(alpha=0.1)(x)
    # Layer 21
    skip_connection = Conv2D(64, (1,1), strides=(1,1), padding="same", name="conv_21", use_bias=False)(skip_connection)
    skip_connection = BatchNormalization(name="norm21")(skip_connection)
    skip_connection = LeakyReLu(alpha=0.1)(skip_connection)

    skip_connection = SpaceToDepth(block_size=2)(skip_connection)
    x = concatenate([skip_connection, x])
    ##Layer 22
    x = Conv2D(1024, (3,3), strides=(1,1), padding="same", name="conv_22", use_bias=False)(x)
    x = BatchNormalization(name="norm22")(x)
    x = LeakyReLu(alpha=0.1)(x)
    x = Dropout(0.5)(x)
    ##Layer 23
    x = Conv2D(mc.BOX * (4 + 1 + mc.CLASSES), (1,1), strides=(1,1), padding="same", name="conv_23")(x)
    output = Reshape((mc.GRID_W, mc.GRID_H, mc.BOX, 4 + 1 + mc.CLASSES))(x)

    model = keras.models.Model(input_image, output)
    return model

model = make_model()
print(model.summary())

### TRAINING THE YOLOv2 MODEL
### Ensuring that each training run is unique with the help of a time tag
train_logname = "TRAIN_" + datetime.datetime.now().strftime("%b_%d_%Y_%H_%M")
LOGDIR = os.path.abspath(os.path.join(mc.maindir, "LOGS", mc.PROJECT, train_logname))
os.makedirs(LOGDIR, exist_ok=True)
results = train_fn(model, mc, train_gen, val_gen, train_folder, val_folder, train_logname, LOGDIR)

