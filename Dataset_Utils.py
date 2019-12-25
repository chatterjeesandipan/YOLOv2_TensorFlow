import warnings
warnings.filterwarnings("ignore")

import os, sys, cv2
import tensorflow as tf
import numpy as np
import tqdm


def image_label_pairs(image_dir, label_dir, name):
    images = [file.split(".")[0] for file in os.listdir(image_dir) if file.endswith(".jpg")]
    labels = [file.split(".")[0] for file in os.listdir(label_dir) if file.endswith(".txt")]
    ### Make sure that you have all the labels corresponding to all the images
    assert len(set(images)) == len(set(labels))

    ### if the above assert passes, then go for the pairing of the image filepaths and their associated labels
    max_annot, LABELS_dict, LABELS_tag = 0, {}, 1
    annotations, imagepath = [], []
    display_string = name + " Dataset Image+Label Pairs" if name is not None else "Image+Label Pairs"

    for i in tqdm.tqdm(range(len(images)), total=len(images), desc=display_string):
        boxes, annot_count = [], 0
        filename = images[i]   ### to know that we are handling the right label file for each image
        imagepath.append(os.path.join(image_dir, filename + ".jpg"))

        labelfile = open(os.path.join(label_dir, filename + ".txt")).read().split("\n")[:-1]
        for line in labelfile:
            line = line.split(" ")
            annot_count += 1
            ### ensuring the numbers are integers, not floats or strings
            line[1:] = list(map(eval, line[1:]))
            if line[0] not in LABELS_dict.keys():
                LABELS_dict[line[0]] = LABELS_tag
                LABELS_tag += 1

            ### first 4 numbers for bounding box and last one for class label as a number {1, 2, 3,....}
            box = np.zeros((5)) ### 4 for bounding boxes and 1 for class_name
            box[:4], box[4] = line[1:], LABELS_dict[line[0]]
            boxes.append(np.asarray(box))
        
        annotations.append(np.asarray(boxes))
        if annot_count > max_annot:
            max_annot = annot_count
        
    ### Rectify annotations box: len -> max_annot
    imagepath = np.array(imagepath)
    gt_boxes = np.zeros((imagepath.shape[0], max_annot, 5))
    for idx, boxes in enumerate(annotations):
        gt_boxes[idx, :boxes.shape[0], :5] = boxes
    ### return the filepaths of the images and the ground truth boxes
    return imagepath, gt_boxes, LABELS_dict


def image_operations(imagepath, gt_boxes):
    img = tf.image.decode_image(tf.io.read_file(imagepath), channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img, gt_boxes


def get_dataset(image_dir, label_dir, BATCH_SIZE, name=None):
    imagepath, gt_boxes, labels_dict = image_label_pairs(image_dir, label_dir, name)
    dataset =  tf.data.Dataset.from_tensor_slices((imagepath, gt_boxes)).shuffle(len(imagepath)).repeat()
    dataset = dataset.map(image_operations, num_parallel_calls=6)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(32)
    print("---------------------------------------------------------------------")
    print("Dataset: ")
    print("Images count: {}".format(len(imagepath)))
    print("Steps per epoch: {}".format(len(imagepath)//BATCH_SIZE))
    print("Images per epoch: {}".format(BATCH_SIZE * (len(imagepath)//BATCH_SIZE)))
    print("---------------------------------------------------------------------")
    return dataset, labels_dict


def visualize_image_bbox_from_dataset(dataset, labels_dict):
    print(labels_dict)
    for batch in dataset:
        img = np.asarray(batch[0][0])
        bbox = batch[1][0]
        for i in range(bbox.shape[0]):
            box = bbox[i, :].numpy()
            if box[4] in list(labels_dict.values()):
                x0, y0, x1, y1 = list(map(int, box[:4]))
                ### find label name from the encoded value in labels_dict
                label = list(labels_dict.keys())[list(labels_dict.values()).index(box[4])]
                xpos, ypos = int(x0 + 0.05*(x1 - x0)), int(y0 + 0.15*(y1 - y0))
                cv2.putText(img, label, (xpos, ypos), cv2.FONT_HERSHEY_PLAIN, 2, (0, 128, 128), 2)
                cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 1, lineType=cv2.LINE_4)
        break    

    cv2.imshow("Sample_Image_BoundBox", img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    return None
