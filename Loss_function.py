import warnings 
warnings.filterwarnings("ignore")

import os, sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
K = keras.backend


def iou(x1, y1, w1, h1, x2, y2, w2, h2):
    xmin1, ymin1 = x1 - 0.5*w1, y1 - 0.5*h1
    xmax1, ymax1 = x1 + 0.5*w1, y1 + 0.5*h1
    xmin2, ymin2 = x2 - 0.5*w2, y2 - 0.5*h2
    xmax2, ymax2 = x2 + 0.5*w2, y2 + 0.5*h2
    
    intersect_x = np.minimum(xmax1, xmax2) - np.maximum(xmin1, xmin2)
    intersect_y = np.minimum(ymax1, ymax2) - np.maximum(ymin1, ymin2)
    intersection = intersect_x * intersect_y
    union = w1*h1 + w2*h2 - intersection
    iou = intersection/(union + 1e-6) ## to avoid division by zero
    return iou


def yolov2_loss(config, detector_mask, match_true_boxes, class_one_hot, true_boxes_grid, y_pred, info=False):
    """
    Calculate YOLO v2 Loss from prediction (y_pred) and ground truth tensors
    Get the output as overall loss value
    Also get the loss in each sub-category: confidence, class, boxes
    """
    ### Anchors
    anchors = np.array(config.ANCHORS)
    anchors = anchors.reshape(len(anchors)//2, 2)
    ### Grid Coords Tensor
    coord_x = tf.cast(tf.reshape(tf.tile(tf.range(config.GRID_W), [config.GRID_H]), \
        (1, config.GRID_H, config.GRID_W, 1, 1)), tf.float32)
    coord_y = tf.transpose(coord_x, (0, 2, 1, 3, 4))
    coords = tf.tile(tf.concat([coord_x, coord_y], -1), [y_pred.shape[0], 1, 1, 5, 1])

    ## BOUNDING BOX LOSS
    pred_xy = K.sigmoid(y_pred[:,:,:,:,0:2])
    pred_xy = (pred_xy + coords)
    pred_wh = K.exp(y_pred[:,:,:,:,2:4]) * anchors
    nb_detector_mask = K.sum(tf.cast(detector_mask > 0.0, tf.float32))
    xy_loss = config.LAMBDA_COORD * K.sum(detector_mask * K.square(match_true_boxes[...,:2] - pred_xy))/(nb_detector_mask + 1e-6)
    wh_loss = config.LAMBDA_COORD * K.sum(detector_mask * K.square(match_true_boxes[...,2:4] - K.sqrt(pred_wh)))/(nb_detector_mask + 1e-6)
    coord_loss = xy_loss + wh_loss 

    ## CLASS LOSS
    pred_box_class = y_pred[..., 5:]
    true_box_class = tf.argmax(class_one_hot, -1)
    class_loss = K.sparse_categorical_crossentropy(target=true_box_class, output=pred_box_class, from_logits=True)
    class_loss = K.expand_dims(class_loss, -1) * detector_mask
    class_loss = config.LAMBDA_CLASS * K.sum(class_loss) / (nb_detector_mask + 1e-6)

    ## CONFIDENCE LOSS
    pred_conf = K.sigmoid(y_pred[..., 4:5])
    ### find iou between prediction and ground truth boxes
    x1, y1, w1, h1 = match_true_boxes[..., 0], match_true_boxes[..., 1], match_true_boxes[..., 2], match_true_boxes[..., 3]
    x2, y2, w2, h2 = pred_xy[...,0], pred_xy[...,1], pred_wh[...,0], pred_wh[...,1] 
    ious = iou(x1, y1, w1, h1, x2, y2, w2, h2)
    ious = K.expand_dims(ious, -1)

    ### for each detector: best ious between prediction and true boxes
    pred_xy = K.expand_dims(pred_xy, 4)
    pred_wh = K.expand_dims(pred_wh, 4)
    pred_wh_half = pred_wh/2.0
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half
    true_box_shape = K.int_shape(true_boxes_grid)
    true_boxes_grid = K.reshape(true_boxes_grid, [true_box_shape[0], 1, 1, 1, true_box_shape[1], true_box_shape[2]])
    true_xy, true_wh = true_boxes_grid[..., 0:2], true_boxes_grid[..., 2:4]
    true_wh_half = true_wh * 0.5
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half
    intersect_mins = K.maximum(pred_mins, true_mins)
    intersect_maxes = K.minimum(pred_maxes, true_maxes)

    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    pred_areas = pred_wh[...,0] * pred_wh[...,1]
    true_areas = true_wh[...,0] * true_wh[...,1]
    
    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas
    best_ious = K.max(iou_scores, axis=4)
    best_ious = K.expand_dims(best_ious)

    ## NO OBJECT CONFIDENCE LOSS
    no_object_detection = K.cast(best_ious < 0.6, K.dtype(best_ious))
    noobj_mask = no_object_detection * (1 - detector_mask)
    nb_noobj_mask = K.sum(tf.cast(noobj_mask > 0.0, tf.float32))
    noobject_loss = config.LAMBDA_NOOBJECT * K.sum(noobj_mask*K.square(-pred_conf)) / (nb_noobj_mask + 1e-6)
    ## object confidence loss
    object_loss = config.LAMBDA_OBJECT * K.sum(detector_mask * K.square(ious - pred_conf)) / (nb_detector_mask + 1e-6)
    ## total confidence loss
    conf_loss = noobject_loss + object_loss

    ### TOTAL LOSS
    loss = conf_loss + class_loss + coord_loss
    sub_loss = [conf_loss, class_loss, coord_loss]

    if info:
        print("Conf_Loss: {:.4f} Class_Loss: {:.4f} Coord_Loss: {:.4f}".format(conf_loss, class_loss, coord_loss))
        print("XY_Loss: {:.4f} WH_Loss: {:.4f}".format(xy_loss, wh_loss))
        print("Total_loss: {:.4f}".format(loss))

    return loss, sub_loss
