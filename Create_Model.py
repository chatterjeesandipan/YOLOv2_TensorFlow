import tensorflow as tf
from tensorflow import keras
K = keras.backend
from tensorflow.keras.layers import *

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

def make_YOLO_model(config):
## YOLO v2 MODEL'S Layers
    input_image = tf.keras.layers.Input((config.IMAGE_W, config.IMAGE_H, 3), dtype="float32")
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
    x = Conv2D(config.BOX * (4 + 1 + config.CLASSES), (1,1), strides=(1,1), padding="same", name="conv_23")(x)
    output = Reshape((config.GRID_W, config.GRID_H, config.BOX, 4 + 1 + config.CLASSES))(x)

    model = keras.models.Model(input_image, output)
    return model


def make_VGG_model(config):
    ### Create a VGG 16 type model
    input_image = tf.keras.layers.Input((config.IMAGE_W, config.IMAGE_H, 3), dtype="float32")
    ## Layer 1
    x = Conv2D(64, (3,3), strides=(1,1), padding="same", name="conv_1", use_bias=False)(input_image)
    x = BatchNormalization(name="norm1")(x)
    x = LeakyReLu(alpha=0.1)(x)
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
    x = Conv2D(128, (3,3), strides=(1,1), padding="same", name="conv_4", use_bias=False)(x)
    x = BatchNormalization(name="norm4")(x)
    x = LeakyReLu(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    ## Layer 5
    x = Conv2D(128, (3,3), strides=(1,1), padding="same", name="conv_5", use_bias=False)(x)
    x = BatchNormalization(name="norm5")(x)
    x = LeakyReLu(alpha=0.1)(x)
    # ## Layer 6
    x = Conv2D(128, (3,3), strides=(1,1), padding="same", name="conv_6", use_bias=False)(x)
    x = BatchNormalization(name="norm6")(x)
    x = LeakyReLu(alpha=0.1)(x)
    ## Layer 7
    x = Conv2D(128, (3,3), strides=(1,1), padding="same", name="conv_7", use_bias=False)(x)
    x = BatchNormalization(name="norm7")(x)
    x = LeakyReLu(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    ## layer 8
    x = Conv2D(256, (3,3), strides=(1,1), padding="same", name="conv_8", use_bias=False)(x)
    x = BatchNormalization(name="norm8")(x)
    x = LeakyReLu(alpha=0.1)(x)
    ## Layer 9
    x = Conv2D(256, (3,3), strides=(1,1), padding="same", name="conv_9", use_bias=False)(x)
    x = BatchNormalization(name="norm9")(x)
    x = LeakyReLu(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    ##Layer 10
    x = Conv2D(config.BOX * (4 + 1 + config.CLASSES), (1,1), strides=(1,1), padding="same", name="conv_10")(x)
    output = Reshape((config.GRID_W, config.GRID_H, config.BOX, 4 + 1 + config.CLASSES))(x)

    model = keras.models.Model(input_image, output)
    return model