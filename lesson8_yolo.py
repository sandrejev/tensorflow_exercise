from __future__ import print_function
from tensorflow.python.keras import models, callbacks, optimizers, regularizers
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, LeakyReLU, Input, Reshape, Dense, Flatten, Dropout, Activation
from datetime import datetime
from os.path import basename
import re
import xml.etree.ElementTree as etree
from utils import *
from clr_callback import CyclicLR
import argparse

parser = argparse.ArgumentParser(description='YOLO')
parser.add_argument('name')
args = parser.parse_args()

def yolo_loss(y_true, y_pred):
    # TODO: try this on xy + wh only model
    # y_true = tf.Print(y_true, [y_true], summarize=1e6, message="\n\n\n\n\n\ny_true->\n")
    # y_pred = tf.Print(y_pred, [y_pred], summarize=1e6, message="\n\n\n\n\n\ny_pred->\n")

    # y_true = np.reshape(np.fromfile("y_true2.np", sep=" "), (2, 7, 7, 30))
    # y_true = tf.to_float(tf.reshape(y_true, (2, 7, 7, 30)))
    # y_pred = np.reshape(np.fromfile("y_pred2.np", sep=" "), (2, 7, 7, 30))
    # y_pred = tf.to_float(tf.reshape(y_pred, (2, 7, 7, 30)))

    grid_shape = np.array([7, 7]) # TODO: detect automatically
    anchors_size = 2 # TODO: detect automatically

    pred_box = tf.stack(tf.split(y_pred[..., 0:anchors_size * 5], num_or_size_splits=anchors_size, axis=3), axis=3)
    true_box = tf.stack(tf.split(y_true[..., 0:anchors_size * 5], num_or_size_splits=anchors_size, axis=3), axis=3)

    pred_box_xy = pred_box[..., 0:2]
    pred_box_wh = pred_box[..., 2:4]
    pred_box_conf = pred_box[..., 4:5]  # tf.sigmoid(
    pred_class = y_pred[..., (anchors_size * 5):]

    true_box_xy = true_box[..., 0:2]
    true_box_wh = true_box[..., 2:4]
    true_box_mask = true_box[..., 4:5]  # tf.sigmoid(
    true_class = y_true[..., (anchors_size * 5):]

    #################################
    ### adjust confidence using IoU
    #################################
    true_wh_norm = (true_box_wh**2) * grid_shape
    true_mins = true_box_xy
    true_maxes = true_box_xy + true_wh_norm

    pred_wh_norm = (pred_box_wh**2) * grid_shape
    pred_mins = pred_box_xy
    pred_maxes = pred_box_xy + pred_wh_norm

    # relative to grid_shape
    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    true_areas = true_wh_norm[..., 0] * true_wh_norm[..., 1]
    pred_areas = pred_wh_norm[..., 0] * pred_wh_norm[..., 1]
    union_areas = pred_areas + true_areas - intersect_areas + 1e-6
    iou_scores = tf.truediv(intersect_areas, union_areas)

    true_box_conf = tf.expand_dims(iou_scores, 4) * true_box_mask

    obj_threshold = 0.6
    true_box_obj = tf.to_float(true_box_mask >= obj_threshold)
    true_box_noobj = tf.to_float(true_box_mask < obj_threshold)
    true_cell_obj = tf.reduce_max(true_box_obj, reduction_indices=[3, 4])


    loss_xy = true_box_obj * tf.squared_difference(pred_box_xy, true_box_xy)
    loss_xy = tf.reduce_mean(tf.reduce_sum(loss_xy, reduction_indices=(1, 2, 3, 4)))
    loss_wh = true_box_obj * tf.squared_difference(pred_box_wh, true_box_wh)
    loss_wh = tf.reduce_mean(tf.reduce_sum(loss_wh, reduction_indices=(1, 2, 3, 4)))
    # eval((true_box_obj * true_box_xy)[0, :, :, 0, :])
    # eval((true_box_obj * pred_box_xy)[0, :, :, 0, :])

    loss_conf = true_box_obj * tf.squared_difference(pred_box_conf, true_box_conf)
    loss_conf = tf.reduce_mean(tf.reduce_sum(loss_conf, reduction_indices=(1, 2, 3, 4)))
    # eval((true_box_obj * pred_box_conf)[0,:,:,0,0])
    # eval((true_box_obj * true_box_conf)[0,:,:,0,0])

    loss_notconf = true_box_noobj * tf.squared_difference(pred_box_conf, true_box_conf)
    loss_notconf = tf.reduce_mean(tf.reduce_sum(loss_notconf, reduction_indices=(1, 2, 3, 4)))
    loss_class = tf.squared_difference(pred_class, true_class)
    loss_class = true_cell_obj * tf.reduce_mean(loss_class, reduction_indices=(3))
    loss_class = tf.reduce_mean(tf.reduce_sum(loss_class, reduction_indices=(1, 2)))
    # eval((true_cell_obj*tf.to_float(tf.arg_max(pred_class, 3)))[0,:,:])
    # eval((true_cell_obj*tf.to_float(tf.arg_max(true_class, 3)))[0,:,:])

    loss = 5.*loss_xy + 5.*loss_wh + loss_class + loss_conf + .5*loss_notconf# 5. * loss_xy + 5. * loss_wh + loss_conf + 0.5 * loss_notconf + loss_class

    # loss = tf.Print(loss, [loss], summarize=1e6, message="\n\n\n\n\n\nloss->\n")
    return loss

def voc2012_get_annotation(self, img_path):
    annotations_path = "VOCdevkit/VOC2012/Annotations/" + re.sub(".jpg", ".xml", basename(img_path))
    xml = etree.parse(annotations_path)
    height = float(xml.find(".//size/height").text)
    width = float(xml.find(".//size/width").text)
    for node in xml.findall(".//object"):
        cl = self.classes.index(node.find("./name").text)
        p = {el.tag: float(el.text)-1 for el in node.findall("./bndbox/*")}
        bb = BBox.from_corners(xmin=p['xmin'], xmax=p['xmax'], ymin=p['ymin'], ymax=p['ymax'], clip_shape=(height, width))
        yield cl, bb


enable_profiler = False
epochs = 600
batch_size = 64
grid_shape = (7, 7)
nbox=2
l2_weights = 1e-4
input_shape = (448, 448, 3)
classes = ["aeroplane",	"bicycle", "bird", "boat", "bottle",
           "bus",	"car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

#
# Feature extractor model
#
input = Input(shape=input_shape)
x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), input_shape=input_shape, padding="SAME", activation="relu")(input)
#x = LeakyReLU(alpha=0.1)(x)
x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")(x)

for fnum in [32,64,128,256]:
    x = Conv2D(fnum, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu")(x)
    #x = LeakyReLU(alpha=0.1)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")(x)
    #x = Dropout(0.1)(x)
x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")(x)

for fnum in [128, 256]:
    x = Conv2D(fnum, kernel_size=(1, 1), strides=(1, 1), padding="SAME", activation="relu")(x)
    #x = LeakyReLU(alpha=0.1)(x)

x = Flatten()(x)
x = Dense(2048, kernel_regularizer=regularizers.l2(l2_weights), activation="relu")(x)
#x = LeakyReLU(alpha=0.1)(x)
#x = Dropout(0.5)(x)

x = Dense(grid_shape[0]*grid_shape[1]*(len(classes) + nbox * 5), activation="relu")(x)
#x = LeakyReLU(alpha=0.1)(x) # TODO: Activation("linear")
#x = Dropout(0.5)(x)

x = Reshape((grid_shape[0], grid_shape[1], len(classes) + nbox * 5))(x)


model = models.Model(inputs=input, outputs=x)
#model.load_weights("K_YOLO/01-10__17-47-04_loss_40_215_215/model-0599.hdf5")
#for layer in model.layers[0:-4]:
#    layer.trainable = False

optimizer = optimizers.Adam(lr=1e-6)
model.compile(loss=yolo_loss, optimizer=optimizer, metrics=[current_learning_rate(optimizer)])
model.summary()
#exit()
#model.save("yolo_model.hdf5")
#exit()

all_images = glob("VOCdevkit/VOC2012/JPEGImages/*.jpg")
train_items = all_images[0:12000]
test_items = all_images[12000:len(all_images)]
data_generator = YoloImageGenerator(classes=classes, target_size=input_shape[0:2], grid_shape=grid_shape, nbox=nbox)
train_iterator = data_generator.flow_from_list(train_items, annotation_callback=voc2012_get_annotation, batch_size=batch_size, augument=False)
test_iterator = data_generator.flow_from_list(test_items, annotation_callback=voc2012_get_annotation, batch_size=batch_size, augument=False)


##############################
# Train model
##############################
log_dir = "./K_YOLO/{}".format(datetime.today().strftime('%m-%d__%H-%M-%S') + "_" + args.name)
tensorboard = callbacks.TensorBoard(log_dir=log_dir, write_graph=True, write_grads=False, write_images=False, histogram_freq=0)
tensorboard.set_model(model)
checkpoint = callbacks.ModelCheckpoint(log_dir + "/model-{epoch:04d}.hdf5", period=1)
terminate = callbacks.TerminateOnNaN()

def step_decay(epoch):
    initial_lr = 1e-4
    drop = 0.05
    lrate = initial_lr * 1/(1 + drop * epoch)
    return lrate

def step_increase(epoch): # designed for 0-35
    initial_lr = 1e-8
    increase = 0.5
    lrate = initial_lr * (1 + increase)**epoch
    return lrate

lr_schedule = callbacks.LearningRateScheduler(step_decay)

model.fit_generator(
    generator=train_iterator,
    epochs=epochs,
    validation_data=test_iterator,
    validation_steps=len(test_items) // batch_size,
    callbacks=[lr_schedule, tensorboard], # checkpoint, tensorboard, lr_schedule,
    steps_per_epoch=len(train_items) // batch_size, #
    verbose=True)

