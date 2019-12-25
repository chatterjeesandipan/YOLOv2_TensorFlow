import os
from easydict import EasyDict as edict

def build_config():
    config = edict()
    ## DEFINE the configuration for the main code
    config.PROJECT = "Road"
    config.IMAGE_H, config.IMAGE_W = 512, 512
    config.GRID_H, config.GRID_W = 16, 16
    config.BOX = 5
    config.SCORE_THRESHOLD = 0.5
    config.IOU_THRESHOLD = 0.45
    config.ANCHORS = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
    config.TRAIN_BATCH_SIZE = 4
    config.VAL_BATCH_SIZE = 2
    config.EPOCHS = 1000
    config.LAMBDA_NOOBJECT = 1
    config.LAMBDA_OBJECT = 5
    config.LAMBDA_CLASS = 1
    config.LAMBDA_COORD = 1

    ### directory contaning all the python codes
    config.maindir = os.path.abspath(os.path.dirname(__file__))

    return config
