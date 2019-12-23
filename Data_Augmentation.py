import os, sys, random
import tensorflow as tf
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa

from config import *
mc = build_config()

def augment_dataset(dataset):
    for batch in dataset:
        img, boxes = batch[0].numpy(), batch[1].numpy()
        batch_size, width, height, channels = img.shape
        ### initiate list for boxes in augmented/modified images
        ia_boxes = []
        for i in range(batch_size):
            ia_box = [ia.BoundingBox(bb[0], bb[1], bb[2], bb[3]) for \
                bb in boxes[i] if sum(bb[:4]) > 0]
            ia_boxes.append(ia.BoundingBoxesOnImage(ia_box, shape=(mc.IMAGE_W, mc.IMAGE_H)))

        ## Initiate the data augmentation
        seq = iaa.Sequential([
            iaa.Fliplr(0.5), iaa.Flipud(0.5), 
            iaa.Multiply((0.4, 1.6)),
            iaa.SaltAndPepper(0.05),
            iaa.Affine(rotate=(-45, 45), shear=(-10, 10)),
            ], random_order=True, name="Image Aug Functions")

        seq_det = seq.to_deterministic()
        img_aug = seq_det.augment_images(img)
        img_aug = np.clip(img_aug, 0, 1)
        boxes_aug = seq_det.augment_bounding_boxes(ia_boxes)
        ### convert bounding boxes to Numpy object
        for i in range(batch_size):
            boxes_aug[i] = boxes_aug[i].remove_out_of_image().clip_out_of_image()
            for j, box in enumerate(boxes_aug[i].bounding_boxes):
                boxes[i, j, 0], boxes[i, j, 1] = box.x1, box.y1
                boxes[i, j, 2], boxes[i, j, 3] = box.x2, box.y2
        batch = (tf.convert_to_tensor(img_aug), tf.convert_to_tensor(boxes))
        
        yield batch