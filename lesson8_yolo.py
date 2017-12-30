from __future__ import print_function
from tensorflow import keras
from tensorflow.python.keras import layers, models, datasets, callbacks, preprocessing, optimizers, applications
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, BatchNormalization, LeakyReLU, Input, Reshape
from datetime import datetime
from keras.regularizers import l2
from os.path import basename
import re
import xml.etree.ElementTree as etree
from utils import *
import itertools
from numpy.random import uniform

from tensorflow.python import debug as tf_debug
#
# sess = tf.Session()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess, dump_root="/tmp/debug/")
# sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
# keras.backend.set_session(sess)

#
# TODO here we should pass original shape and resized
#
def voc2012_get_annotation(self, path):
    img = Image.open(path)
    size_scaler = tuple(s[1] / float(s[0]) for s in zip(img.size, self.target_size))
    img = img.resize(self.target_size)
    img = np.asarray(img) / 255.

    y = np.zeros(self.grid_shape + (self.anchors_size*5 + len(self.classes),), dtype=K.floatx())
    y_grid = np.zeros(self.grid_shape, dtype=np.int) # Count

    features_path = "VOCdevkit/VOC2012/Annotations/" + re.sub(".jpg", ".xml", basename(path))
    xml = etree.parse(features_path)
    for node in xml.findall(".//object"):
        y_class = self.classes.index(node.find("./name").text)

        # Read information about bounding box and rescale it to target size
        xmax, xmin, ymax, ymin = [float(el.text) for el in sorted(node.findall("./bndbox/*"), key=lambda el: el.tag)]
        bb = BBox.from_corners(xmin*size_scaler[0], xmax*size_scaler[0], ymin*size_scaler[1], ymax*size_scaler[1])
        bb_cell, bb_rel = bb.grid_position(self.target_size, self.grid_shape)

        # Check whether this grid cell is already full
        if y_grid[bb_cell] >= self.anchors_size:
            continue

        y[bb_cell[0], bb_cell[1], (y_grid[bb_cell]*5):(y_grid[bb_cell]*5+2)] = bb_rel
        y[bb_cell[0], bb_cell[1], (y_grid[bb_cell]*5+2):(y_grid[bb_cell]*5+4)] = [bb.w / self.target_size[0], bb.h / self.target_size[1]]
        y[bb_cell[0], bb_cell[1], (y_grid[bb_cell]*5+4)] = 1.
        y[bb_cell[0], bb_cell[1], 5*anchors_size + y_class] = 1
        y_grid[bb_cell] += 1

    return img, y



epochs = 128
batch_size = 128
grid_shape = (7, 7)
anchors_size=2
l2_weights = 1e-4
input_shape = (448, 448, 3)
anchors =  [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]
classes = ["aeroplane",	"bicycle", "bird", "boat", "bottle", "bus",	"car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

#
# Feature extractor model
#
input = Input(shape=input_shape)
x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape, kernel_regularizer=l2(l2_weights), padding="SAME")(input)
x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")(x)
for cnn_depth in range(1,6):
    x = Conv2D(16*(2**cnn_depth), kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=l2(l2_weights), padding="SAME")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")(x)

x = Conv2D(16*(2**cnn_depth), kernel_size=(3, 3), strides=(1, 1), activation='relu', kernel_regularizer=l2(l2_weights), padding="SAME")(x)
x = Conv2D((anchors_size*5) + len(classes), kernel_size=(1, 1), strides=(1, 1), activation='softmax', kernel_regularizer=l2(l2_weights), padding="SAME")(x)
model = models.Model(inputs=input, outputs=x)

model.compile(loss=yolo_loss,
              optimizer=optimizers.Adam(lr=0.0001),
              metrics=['accuracy'])

test_items = glob("VOCdevkit/VOC2012/JPEGImages/*.jpg")
train_items = glob("VOCdevkit/VOC2012/JPEGImages/*.jpg")
test_generator = YoloImageGenerator(classes=classes, target_size=input_shape[0:2], grid_shape=grid_shape, anchors_size=anchors_size)
train_generator = YoloImageGenerator(classes=classes, target_size=input_shape[0:2], grid_shape=grid_shape, anchors_size=anchors_size)
test_iterator = test_generator.flow_from_directory("VOCdevkit/VOC2012/JPEGImages/*.jpg", callback=voc2012_get_annotation, batch_size=batch_size)
train_iterator = train_generator.flow_from_directory("VOCdevkit/VOC2012/JPEGImages/*.jpg", callback=voc2012_get_annotation, batch_size=batch_size)

##############################
# Train model
##############################
tensorboard = callbacks.TensorBoard(log_dir="./K_YOLO/{}".format(datetime.today().strftime('%m-%d__%H-%M-%S')),
    histogram_freq=0, write_graph=True)
tensorboard.set_model(model)
model.fit_generator(
    generator=train_iterator,
    epochs=epochs,
    validation_data=test_iterator,
    validation_steps=len(test_items) // batch_size,
    callbacks=[tensorboard],
    steps_per_epoch=len(train_items) // batch_size,
    verbose=True)

model.save('yolo.h5')
