import warnings 
warnings.filterwarnings("ignore")

import os, sys, datetime
import tensorflow as tf
from tensorflow import keras

from Loss_function import *

def grad(model, config, img, detector_mask, match_true_boxes, class_one_hot, true_boxes, training=True):
    with tf.GradientTape() as tape:
        y_pred = model(img, training)
        loss, sub_loss = yolov2_loss(config, detector_mask, match_true_boxes, class_one_hot, true_boxes, y_pred)

    return loss, sub_loss, tape.gradient(loss, model.trainable_variables)

def save_best_weights(model, log_dir, val_loss_avg):
    filepath = os.path.join(log_dir, "Model_val_loss_" + str(round(val_loss_avg,2)).replace(".", "_") + ".h5")
    model.save_weights(filepath)

### add items for tracking on tensorboard
# def log_loss(loss, val_loss, step):
#     tf.summary.scalar("loss", loss, step)
#     tf.summary.scalar("val_loss", val_loss, step)


def train_fn(model, config, train_dataset, val_dataset, train_dir, val_dir, train_logname, log_dir):
    train_files = [file for file in os.listdir(os.path.join(train_dir, "Images")) if file.endswith(".jpg")]
    val_files = [file for file in os.listdir(os.path.join(val_dir, "Images")) if file.endswith(".jpg")]
    steps_per_epoch_train = len(train_files)//config.TRAIN_BATCH_SIZE
    steps_per_epoch_val = len(val_files)//config.VAL_BATCH_SIZE

    train_loss_history = []
    val_loss_history = []
    best_val_loss = 1e6

    ## optimizer
    optimizer = keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    ## Log tensorboard
    summary_writer = tf.summary.create_file_writer(os.path.join(log_dir), flush_millis=5000)
    summary_writer.set_as_default()

    ### TRAINING
    for epoch in range(config.EPOCHS):
        start_time = datetime.datetime.now()
        epoch_loss, epoch_sub_loss, epoch_val_loss, epoch_val_sub_loss = [], [], [], []
        for batch_idx in range(steps_per_epoch_train):
            img, detector_mask, match_true_boxes, class_one_hot, true_boxes = next(train_dataset)
            loss, sub_loss, grads = grad(model, config, img, detector_mask, match_true_boxes, class_one_hot, true_boxes)

            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss.append(loss)
            epoch_sub_loss.append(sub_loss)


        for batch_idx in range(steps_per_epoch_val):
            img, detector_mask, match_true_boxes, class_one_hot, true_boxes = next(val_dataset)
            loss, sub_loss, grads = grad(model, config, img, detector_mask, match_true_boxes, class_one_hot, true_boxes, training=False)
            epoch_val_loss.append(loss)
            epoch_val_sub_loss.append(sub_loss)

        loss_avg = np.mean(np.array(epoch_loss))
        sub_loss_avg = np.mean(np.array(epoch_sub_loss), axis=0)
        val_loss_avg = np.mean(np.array(epoch_val_loss))
        val_sub_loss_avg = np.mean(np.array(epoch_val_sub_loss), axis=0)
        train_loss_history.append(loss_avg)
        val_loss_history.append(val_loss_avg)


        ### log the loss on tensorboard
        tf.summary.scalar("loss", loss_avg, epoch)
        tf.summary.scalar("val_loss", val_loss_avg, epoch)
        tf.summary.scalar("Conf_loss", sub_loss_avg[0], epoch)
        tf.summary.scalar("Class_loss", sub_loss_avg[1], epoch)
        tf.summary.scalar("Box_loss", sub_loss_avg[2], epoch)
        tf.summary.scalar("Val_Conf_loss", val_sub_loss_avg[0], epoch)
        tf.summary.scalar("Val_Class_loss", val_sub_loss_avg[1], epoch)
        tf.summary.scalar("Val_Box_loss", val_sub_loss_avg[2], epoch)
        

        ### Save if model has improved
        if val_loss_avg < best_val_loss:
            save_best_weights(model, log_dir, val_loss_avg)
            best_val_loss = val_loss_avg

        time_epoch = datetime.datetime.now() - start_time
        time_epoch = round(time_epoch.total_seconds(), ndigits=3)

        print("====================================================================================")
        print("Epoch: {} \t\t Time elapsed: {}".format(epoch, time_epoch))
        print("Loss: {:.3f} Val_Loss: {:.3f}".format(loss_avg, val_loss_avg))
        print("Train_Conf: {:.3f} Train_Class: {:.3f} Train_Box: {:.3f}".format(sub_loss_avg[0], sub_loss_avg[1], sub_loss_avg[2]))
        print("Val_Conf: {:.3f} Val_Class: {:.3f} Val_Box: {:.3f}".format(val_sub_loss_avg[0], val_sub_loss_avg[1], val_sub_loss_avg[2]))
        print("====================================================================================")

    return [train_loss_history, val_loss_history]
