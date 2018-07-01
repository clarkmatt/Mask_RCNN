
# coding: utf-8

# # Mask R-CNN - Train on Shapes Dataset
# 
# 
# This notebook shows how to train Mask R-CNN on your own dataset. To keep things simple we use a synthetic dataset of shapes (squares, triangles, and circles) which enables fast training. You'd still need a GPU, though, because the network backbone is a Resnet101, which would be too slow to train on a CPU. On a GPU, you can start to get okay-ish results in a few minutes, and good results in less than an hour.
# 
# The code of the *Shapes* dataset is included below. It generates images on the fly, so it doesn't require downloading any data. And it can generate images of any size, so we pick a small image size to train faster. 

# In[1]:


import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import json
import skimage

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

#get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


# ## Configurations

# In[2]:


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "strawberry"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    
config = ShapesConfig()
config.display()


# ## Notebook Preferences

# In[3]:


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# ## Dataset
# 
# Create a synthetic dataset
# 
# Extend the Dataset class and add a method to load the shapes dataset, `load_shapes()`, and override the following methods:
# 
# * load_image()
# * load_mask()
# * image_reference()

# In[4]:


class ShapesDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    #def load_shapes(self, count, height, width):
    def load_shapes(self, dataset_dir, subset):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("strawberry", 1, "red_berry")
        self.add_class("strawberry", 2, "green_berry")
        self.add_class("strawberry", 3, "flower")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        self.dataset_dir = dataset_dir

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            #polygons = [r['shape_attributes'] for r in a['regions'].values()]
            polygons = a['regions']

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "strawberry",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_image(self, image_idx):
        """
        Load image with given image_idx from list of images in dataset
        """
        info = self.image_info[image_idx]
        image_path = os.path.join(self.dataset_dir, info['id'])
        image = skimage.io.imread(image_path)
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "strawberry":
            return info["strawberry"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        polygons = info['polygons']
        count = len(polygons.keys())
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, polygon in enumerate(info['polygons'].values()):
            p = polygon['shape_attributes']
            #mask[:, :, i:i+1] = self.draw_shape(mask[:, :, i:i+1].copy(),
            #                                    shape, dims, 1)
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))

        # Map class names to class IDs.
        class_ids = np.array([int(s['region_attributes']['class']) for s in polygons.values()])
        return mask.astype(np.bool), class_ids.astype(np.int32)

#    def draw_shape(self, image, shape, dims, color):
#        """Draws a shape from the given specs."""
#        # Get the center x, y and the size s
#        x, y, s = dims
#        if shape == 'square':
#            cv2.rectangle(image, (x-s, y-s), (x+s, y+s), color, -1)
#        elif shape == "circle":
#            cv2.circle(image, (x, y), s, color, -1)
#        elif shape == "triangle":
#            points = np.array([[(x, y-s),
#                                (x-s/math.sin(math.radians(60)), y+s),
#                                (x+s/math.sin(math.radians(60)), y+s),
#                                ]], dtype=np.int32)
#            cv2.fillPoly(image, points, color)
#        return image
#
#    def random_shape(self, height, width):
#        """Generates specifications of a random shape that lies within
#        the given height and width boundaries.
#        Returns a tuple of three valus:
#        * The shape name (square, circle, ...)
#        * Shape color: a tuple of 3 values, RGB.
#        * Shape dimensions: A tuple of values that define the shape size
#                            and location. Differs per shape type.
#        """
#        # Shape
#        shape = random.choice(["square", "circle", "triangle"])
#        # Color
#        color = tuple([random.randint(0, 255) for _ in range(3)])
#        # Center x, y
#        buffer = 20
#        y = random.randint(buffer, height - buffer - 1)
#        x = random.randint(buffer, width - buffer - 1)
#        # Size
#        s = random.randint(buffer, height//4)
#        return shape, color, (x, y, s)
#
#    def random_image(self, height, width):
#        """Creates random specifications of an image with multiple shapes.
#        Returns the background color of the image and a list of shape
#        specifications that can be used to draw the image.
#        """
#        # Pick random background color
#        bg_color = np.array([random.randint(0, 255) for _ in range(3)])
#        # Generate a few random shapes and record their
#        # bounding boxes
#        shapes = []
#        boxes = []
#        N = random.randint(1, 4)
#        for _ in range(N):
#            shape, color, dims = self.random_shape(height, width)
#            shapes.append((shape, color, dims))
#            x, y, s = dims
#            boxes.append([y-s, x-s, y+s, x+s])
#        # Apply non-max suppression wit 0.3 threshold to avoid
#        # shapes covering each other
#        keep_ixs = utils.non_max_suppression(np.array(boxes), np.arange(N), 0.3)
#        shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
#        return bg_color, shapes


#Parse command line arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False,
                    metavar="/path/to/balloon/dataset/",
                    help='Directory of the Balloon dataset')
args = parser.parse_args()

# Training dataset
dataset_train = ShapesDataset()
dataset_train.load_shapes(args.dataset, "train")
dataset_train.prepare()

# Validation dataset
dataset_val = ShapesDataset()
dataset_val.load_shapes(args.dataset, "val")
dataset_val.prepare()


# In[6]:


# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)


# ## Ceate Model

# In[ ]:


# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)


# In[7]:


# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)


# ## Training
# 
# Train in two stages:
# 1. Only the heads. Here we're freezing all the backbone layers and training only the randomly initialized layers (i.e. the ones that we didn't use pre-trained weights from MS COCO). To train only the head layers, pass `layers='heads'` to the `train()` function.
# 
# 2. Fine-tune all layers. For this simple example it's not necessary, but we're including it to show the process. Simply pass `layers="all` to train all layers.

# In[8]:


# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=1, 
            layers='heads')


# In[9]:


# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=2, 
            layers="all")


# In[10]:


# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
# model.keras_model.save_weights(model_path)


