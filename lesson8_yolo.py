from __future__ import print_function
from tensorflow.python.keras import layers, models, datasets, callbacks, preprocessing, optimizers, applications
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, BatchNormalization, LeakyReLU, Input, Reshape
from datetime import datetime
from keras.regularizers import l2
from os.path import basename
import re
import cv2
import xml.etree.ElementTree as etree
import matplotlib.pyplot as plt
from utils import BBox, draw_bbox, draw_grid, draw_grid_values
from utils import *

def voc2012_get_annotation(self, path):
    grid_shape = self.grid_shape
    classes = self.classes
    classes_len = len(classes)
    target_shape = self.target_size
    bnum = self.anchors_size
    self.grid_shape = grid_shape

    img = cv2.imread(path)[..., [2, 1, 0]].copy()

    y = np.zeros(grid_shape + (bnum * 5 + classes_len,), dtype=np.float)
    grid_nbox = np.zeros(grid_shape, dtype=np.int)  # Count
    features_path = "VOCdevkit/VOC2012/Annotations/" + re.sub(".jpg", ".xml", basename(path))
    xml = etree.parse(features_path)
    for node in xml.findall(".//object"):
        y_class = classes.index(node.find("./name").text)

        # Read information about bounding box and rescale it to target size
        xmax, xmin, ymax, ymin = [float(el.text) for el in sorted(node.findall("./bndbox/*"), key=lambda el: el.tag)]
        bb = BBox.from_corners(xmin, xmax, ymin, ymax)
        bb_cell, bb_rel = position_absolute2grid(img, grid_shape, (bb.y, bb.x))

        # Check whether this grid cell is already full
        cell_nbox = grid_nbox[bb_cell[0], bb_cell[1]]
        if cell_nbox >= bnum:
            continue

        y[bb_cell[0], bb_cell[1], (cell_nbox * 5):(cell_nbox * 5 + 2)] = bb_rel
        y[bb_cell[0], bb_cell[1], (cell_nbox * 5 + 2):(cell_nbox * 5 + 4)] = [bb.h / img.shape[0], bb.w / img.shape[1]]
        y[bb_cell[0], bb_cell[1], (cell_nbox * 5 + 4)] = 1.
        y[bb_cell[0], bb_cell[1], 5 * bnum + y_class] = 1.
        grid_nbox[bb_cell[0], bb_cell[1]] += 1

        #print(bb_cell, bb_rel, [bb.h / img.shape[0], bb.w / img.shape[1]], (bb.y, bb.x, bb.h, bb.w))

    x = cv2.resize(img, dsize=target_shape) / 255.
    return x, y



epochs = 600
batch_size = 64
grid_shape = (7, 7)
anchors_size=64
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
              optimizer=optimizers.Adam(lr=1e-3),
              metrics=[])

test_items = glob("VOCdevkit/VOC2012/JPEGImages/*.jpg")
train_items = glob("VOCdevkit/VOC2012/JPEGImages/*.jpg")
test_generator = YoloImageGenerator(classes=classes, target_size=input_shape[0:2], grid_shape=grid_shape, anchors_size=anchors_size)
train_generator = YoloImageGenerator(classes=classes, target_size=input_shape[0:2], grid_shape=grid_shape, anchors_size=anchors_size)
test_iterator = test_generator.flow_from_directory("VOCdevkit/VOC2012/JPEGImages/*.jpg", callback=voc2012_get_annotation, batch_size=batch_size)
train_iterator = train_generator.flow_from_directory("VOCdevkit/VOC2012/JPEGImages/*.jpg", callback=voc2012_get_annotation, batch_size=batch_size)

##############################
# Train model
##############################
log_dir = "./K_YOLO/{}".format(datetime.today().strftime('%m-%d__%H-%M-%S'))
tensorboard = callbacks.TensorBoard(log_dir=log_dir, write_graph=True, write_grads=False, write_images=False, histogram_freq=0)
tensorboard.set_model(model)
checkpoint = callbacks.ModelCheckpoint(log_dir + "model-{epoch:04d}.hdf5", period=50)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
model.fit_generator(
    generator=train_iterator,
    epochs=epochs,
    validation_data=test_iterator,
    validation_steps=len(test_items) // batch_size,
    callbacks=[tensorboard, checkpoint, reduce_lr],
    steps_per_epoch=len(train_items) // batch_size,
    verbose=True)

model.save('yolo.h5')
