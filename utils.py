from glob import glob
import numpy as np
from tensorflow.python.keras._impl.keras import backend as K
import tensorflow as tf
import cv2
import pickle
import itertools
from collections import defaultdict
from imgaug import augmenters as iaa
import imgaug as ia
import matplotlib.pyplot as plt


class BBox(object):
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    @staticmethod
    def from_corners(xmin, xmax, ymin, ymax, clip_shape=None):
        if clip_shape:
            xmin = np.clip(xmin, 0, clip_shape[1] - 1)
            xmax = np.clip(xmax, 0, clip_shape[1] - 1)
            ymin = np.clip(ymin, 0, clip_shape[0] - 1)
            ymax = np.clip(ymax, 0, clip_shape[0] - 1)

        x = (xmax + xmin)/2.
        y = (ymax + ymin)/2.
        w = xmax - xmin
        h = ymax - ymin

        return BBox(x, y, w, h)

    @property
    def center(self):
        return (self.y, self.x)

    @property
    def xmin(self):
        return self.x - self.w / 2.

    @property
    def xmax(self):
        return self.x + self.w / 2.

    @property
    def ymin(self):
        return self.y - self.h / 2.

    @property
    def ymax(self):
        return self.y + self.h / 2.

    def resize(self, scaler):
        self.x *= scaler[1]
        self.y *= scaler[0]
        self.w *= scaler[1]
        self.h *= scaler[0]
    #
    # def grid_position(self, img_size, grid_shape):
    #     cell_shape = tuple(s[1] / float(s[0]) for s in zip(grid_shape, img_size))
    #     grid_coords = tuple(sum(self.center[i] > np.linspace(-1e-10, img_size[i], num=grid_shape[i]+1))-1 for i in range(2))
    #     within_grid_coords = tuple((bb - (g+0.5)*c)/c for bb,g,c in zip(self.center, grid_coords, cell_shape))
    #
    #     return grid_coords, within_grid_coords

    def as_list(self):
        return [self.x, self.y, self.w, self.h]

    def iou(self, box2):
        intersect_w = self.interval_overlap([self.xmin, self.xmax], [box2.xmin, box2.xmax])
        intersect_h = self.interval_overlap([self.ymin, self.ymax], [box2.ymin, box2.ymax])

        intersect = intersect_w * intersect_h
        union = self.w * self.h + box2.w * box2.h - intersect

        return float(intersect) / union

class YoloImageGenerator(object):
    def __init__(self, classes, target_size, grid_shape=(7,7), nbox=2):
        self.classes = classes
        self.target_size = target_size
        self.nbox = nbox
        self.grid_shape = grid_shape
        self.augmentation_pipeline = iaa.Sequential([
            iaa.Fliplr(0.5),  # horizontally flip 50% of the images
            iaa.Sometimes(0.9,
                iaa.Sequential([
                    iaa.Sometimes(0.5, iaa.Affine(translate_px={"x": (-100, 100), "y": (-100, 100)})),
                    iaa.Sometimes(0.5, iaa.Affine(scale={"x": (0.75, 1.2), "y": (0.75, 1.2)})),
                    iaa.Sometimes(0.8, iaa.Affine(rotate=(-5., 5), scale=1.2)),
                    iaa.Crop(px=(int(self.target_size[0]*0.1), int(self.target_size[1]*0.1), int(self.target_size[0]*0.1), int(self.target_size[1]*0.1))),
                    iaa.ElasticTransformation(alpha=(0, 0.2)),
                    iaa.Add((-100,100))
                ])
            ),
            iaa.Scale({"height":self.target_size[0], "width":self.target_size[1]})
        ])
        self.scale_pipeline = iaa.Sequential([
            iaa.Scale({"height":self.target_size[0], "width":self.target_size[1]})
        ])

    def generate_yolo_grid(self, img_shape, objects):
        bbox_features = defaultdict(lambda: [])

        y = np.zeros(self.grid_shape + (self.nbox * 5 + len(self.classes),), dtype=np.float)
        for cl, bb in objects:
            bb_cell, bb_rel = position_absolute2grid(img_shape, self.grid_shape, (bb.y, bb.x))
            bb_size = np.array([bb.h, bb.w]) / img_shape
            assert bb.xmin >= 0 and bb.ymin >= 0 and bb.xmax < self.target_size[1] and bb.ymax < self.target_size[0]
            assert bb_cell[0] >= 0 and bb_cell[1] >= 0 and bb_cell[0] < self.grid_shape[0] and bb_cell[1] < self.grid_shape[1]

            #print(bb_cell, bb_rel)
            y[bb_cell[0], bb_cell[1], 5 * self.nbox + cl] = 1.
            bbox_features[tuple(bb_cell)].append(np.concatenate((bb_rel, np.sqrt(bb_size), [1.])))

        for bb_cell, features in bbox_features.items():
            for bb_i, f in enumerate(itertools.cycle(features)):
                if bb_i >= self.nbox: break
                y[bb_cell[0], bb_cell[1], (bb_i * 5):(bb_i * 5 + 5)] = f

        return y

    def flow_from_list(self, images, annotation_callback, batch_size=10, augument=True):
        batch_index = 0
        total_batches_seen = 0
        n = len(images)

        while 1:
            np.random.shuffle(images)
            current_index = (batch_index * batch_size) % n
            if n > current_index + batch_size:
                current_batch_size = batch_size
                batch_index += 1
            else:
                current_batch_size = n - current_index
                batch_index = 0
            total_batches_seen += 1

            batch_augmentation_pipeline = self.augmentation_pipeline.to_deterministic()

            batch_images_paths = images[current_index:current_index + current_batch_size]
            batch_y = np.zeros((current_batch_size,) + (self.grid_shape + (5*self.nbox + len(self.classes),)), dtype=K.floatx())
            batch_x = np.zeros((current_batch_size,) + self.target_size + (3,), dtype=K.floatx())
            for img_i, image_path in enumerate(batch_images_paths):
                img = cv2.imread(image_path)[..., [2, 1, 0]].copy()
                img_shape = np.array(img.shape[:2])
                img_annotations = list(annotation_callback(self, image_path))
                img_preprocessing = batch_augmentation_pipeline if augument else self.scale_pipeline

                img_bboxes = []
                for cl, bb in img_annotations:
                    img_bboxes.append(ia.BoundingBox(x1=bb.xmin, y1=bb.ymin, x2=bb.xmax, y2=bb.ymax))
                img_bboxes = ia.BoundingBoxesOnImage(img_bboxes, shape=img.shape)
                aug_bboxes = img_preprocessing.augment_bounding_boxes([img_bboxes])[0].cut_out_of_image()

                aug_annotations = []
                for bb_i, bb_aug in enumerate(aug_bboxes.bounding_boxes):
                    if bb_aug.x2 - bb_aug.x1 < 50 or bb_aug.y2 - bb_aug.y1 < 50:
                        continue
                    cl, _ = img_annotations[bb_i]
                    bb = BBox.from_corners(xmin=bb_aug.x1, xmax=bb_aug.x2, ymin=bb_aug.y1, ymax=bb_aug.y2, clip_shape=self.target_size)
                    aug_annotations.append((cl, bb))

                batch_x[img_i] = batch_augmentation_pipeline.augment_image(img) / 255.
                batch_y[img_i] = self.generate_yolo_grid(self.target_size, aug_annotations)


                # draw_predicted_boxes(img, self.generate_yolo_grid(img_shape, img_annotations), self.classes, conf_threshold = 0.5)
                # plt.imshow(img)
                # plt.show()
                #
                # draw_predicted_boxes(batch_x[img_i], batch_y[img_i], self.classes, conf_threshold = 0.5)
                # plt.imshow(batch_x[img_i].astype(np.uint8))
                # plt.show()
                # x = "test"

                #print("batch {}: {}".format(batch_index, np.sum(batch_y)))

            #pickle.dump((batch_x, batch_y), open("nan.pickle", "wb"))
            yield batch_x, batch_y


def current_learning_rate(optimizer):
    def _current_learning_rate(y_true, y_pred):
        return optimizer.lr / (1. + optimizer.decay * K.cast(optimizer.iterations, K.dtype(optimizer.decay)))

    return _current_learning_rate


def draw_grid(img, grid_shape):
    cell_shape = tuple(s[1] // s[0] for s in zip(grid_shape, img.shape))
    for x in range(grid_shape[1]):
        cv2.line(img, (cell_shape[1]*x, 0), (cell_shape[1]*x, img.shape[0]-1), (0,255,0), 1)
    for x in range(grid_shape[0]):
        cv2.line(img, (0, cell_shape[0]*x), (img.shape[1]-1, cell_shape[0]*x), (0,255,0), 1)


def draw_grid_values(img, grid, offset=(0.7, 0.3), fontScale=1):
    draw_grid(img, grid.shape)
    offset = np.array(offset)
    cell_shape = np.array([s[1] / s[0] for s in zip(grid.shape, img.shape)])
    for coord, cl in np.ndenumerate(grid):
        if not cl:
            continue

        coord = np.array(coord)
        pt = tuple(np.flip(coord*cell_shape + offset*cell_shape, 0).astype(int))
        cv2.putText(img, str(cl), pt, cv2.FONT_HERSHEY_DUPLEX, fontScale, color=(0,255,0), thickness=1)


def draw_bbox(img, bbox, width=1):
    cv2.line(img, (int(bbox.xmin), int(bbox.ymin)), (int(bbox.xmin), int(bbox.ymax)), (255,0,0), int(width))
    cv2.line(img, (int(bbox.xmax), int(bbox.ymin)), (int(bbox.xmax), int(bbox.ymax)), (255,0,0), int(width))
    cv2.line(img, (int(bbox.xmin), int(bbox.ymin)), (int(bbox.xmax), int(bbox.ymin)), (255,0,0), int(width))
    cv2.line(img, (int(bbox.xmin), int(bbox.ymax)), (int(bbox.xmax), int(bbox.ymax)), (255,0,0), int(width))

def position_absolute2grid(img_shape, grid_shape, yx):
    grid_shape = np.array(grid_shape)
    cell_shape = img_shape / grid_shape

    yx = np.array(yx)
    bb_cell = (yx // cell_shape).astype(int)
    bb_rel = yx / cell_shape - bb_cell

    return bb_cell, bb_rel

def draw_predicted_boxes(img, prediction, classes, conf_threshold = 0.1):
    classes_empty4 = [cl[:5] for cl in classes]
    classes_empty4.append("")

    bnum = int((prediction.shape[2] - len(classes)) / 5)
    grid_shape = np.array(prediction.shape[:2])
    cell_shape = np.array(img.shape[:2]) / np.array(grid_shape)

    # Display grid
    draw_grid(img, grid_shape)

    # Get prediction confedences and classes
    prediction_conf = prediction[..., np.arange(4, bnum * 5, 5)]
    prediction_class = np.tile(np.expand_dims(np.argmax(prediction[..., (bnum * 5):(bnum * 5 + len(classes))], 2), 2), (1, 1, bnum))

    prediction_class[prediction_conf < conf_threshold] = len(classes)
    prediction_class_str = np.array(classes_empty4)[prediction_class]

    # Get predictions display matrix
    display_matrix = np.reshape(np.repeat(": ", np.prod(prediction_class_str.shape)), prediction_class_str.shape)
    display_matrix = np.core.defchararray.add(prediction_class_str, display_matrix)
    display_matrix = np.core.defchararray.add(display_matrix, (prediction_conf * 100).astype(int).astype(str))
    display_matrix = np.core.defchararray.add(display_matrix,
                                              np.reshape(np.repeat("%", np.prod(prediction_class_str.shape)),
                                                         prediction_class_str.shape))
    display_matrix[prediction_conf < conf_threshold] = ""

    # Display predicted boxes
    prediction_yxhw = prediction[:, :,
                      np.tile(np.arange(4), (bnum, 1)) + np.reshape(np.repeat(np.arange(0, bnum * 5, 5), 4), (bnum, 4))]
    for gy, gx, b in np.ndindex(*prediction_yxhw.shape[:3]):
        if prediction_conf[gy, gx, b] < conf_threshold:
            continue

        yxhw = prediction_yxhw[gy, gx, b, :]
        yx = np.array([gy, gx]) * cell_shape + yxhw[:2] * cell_shape
        hw = yxhw[2:4]**2 * img.shape[:2]
        conf = prediction_yxhw[5]
        bb = BBox(yx[1], yx[0], hw[1], hw[0])
        draw_bbox(img, bb, width=prediction_conf[gy, gx, b] * 3)

    # Display predicted classes
    for i in range(display_matrix.shape[2]):
        draw_grid_values(img, display_matrix[:, :, i], fontScale=0.4, offset=(0.3, 0.0))