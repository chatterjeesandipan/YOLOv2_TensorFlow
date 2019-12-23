import warnings
warnings.filterwarnings("ignore")

import os, sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
K = keras.backend

from config import *
mc = build_config()

def process_true_boxes(true_boxes):
    """
    Build image ground truth in YOLO format from the true boxes and anchors
    """
    scale = mc.IMAGE_W / mc.GRID_W
    anchor_count = len(mc.ANCHORS) // 2
    anchors = np.array(mc.ANCHORS)
    anchors = anchors.reshape(len(anchors)//2, 2)

    detector_mask = np.zeros((mc.GRID_W, mc.GRID_H, anchor_count, 1))
    match_true_boxes = np.zeros((mc.GRID_W, mc.GRID_H, anchor_count, 5))

    ### convert true boxes numpy array to Tensor
    true_boxes = true_boxes.numpy()
    true_boxes_grid = np.zeros(true_boxes.shape)

    for i, box in enumerate(true_boxes):
        ### convert box coords x, y, w, h and convert to grid coords
        w = (box[2] - box[0]) / scale
        h = (box[3] - box[1]) / scale
        x = ((box[0] + box[2]) / 2) / scale
        y = ((box[1] + box[3]) / 2) / scale

        true_boxes_grid[i, ...] = np.array([x, y, w, h, box[4]])
        if w * h > 0 and box[4] != 0: ## we have a box
            best_iou = 0
            best_anchor = 0
            for i in range(anchor_count):
                intersect = np.minimum(w, anchors[i,0]) * np.minimum(h, anchors[i, 1])
                union = (anchors[i, 0] * anchors[i, 1]) + (w * h) - intersect
                iou = intersect / union
                ### select the anchor with the best iou
                if iou > best_iou:
                    best_iou, best_anchor  = iou, i
            if best_iou > 0:
                x_coord = np.floor(x).astype("int")
                y_coord = np.floor(y).astype("int")
                detector_mask[y_coord, x_coord, best_anchor] = 1
                yolo_box = np.array([x, y, w, h, box[4]])
                match_true_boxes[y_coord, x_coord, best_anchor] = yolo_box
    return match_true_boxes, detector_mask, true_boxes_grid


def ground_truth_generator(dataset, labels_dict):
    """
    convert the true bounding box data to YOLO format; for use with 
    predictions in loss functions
    """
    for batch in dataset:
        imgs, true_boxes = batch[0], batch[1]
        ### YOLO format bounding box data for this batch
        batch_matching_true_boxes = []
        batch_detector_mask = []
        batch_true_boxes_grid = []

        for i in range(true_boxes.shape[0]):
            one_match_true_boxes, one_detector_mask, true_boxes_grid = \
                process_true_boxes(true_boxes[i])

            batch_matching_true_boxes.append(one_match_true_boxes)
            batch_detector_mask.append(one_detector_mask)
            batch_true_boxes_grid.append(true_boxes_grid)

        detector_mask = tf.convert_to_tensor(np.array(batch_detector_mask), dtype="float32")
        match_true_boxes = tf.convert_to_tensor(np.array(batch_matching_true_boxes), dtype="float32")
        true_boxes_grid = tf.convert_to_tensor(np.array(batch_true_boxes_grid), dtype="float32")

        ### one-hot encoding of the classes
        CLASSES = len(list(labels_dict.keys()))
        match_classes = tf.cast(match_true_boxes[..., 4], dtype='int32')
        class_one_hot = tf.cast(K.one_hot(match_classes, CLASSES + 1)[ :, :, :, :, 1:], dtype='float32')

        batch = (imgs, detector_mask, match_true_boxes, class_one_hot, true_boxes_grid)
        yield batch

