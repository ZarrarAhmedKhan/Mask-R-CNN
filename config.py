from mrcnn.config import Config
import os

ROOT_DIR = os.path.abspath(os.getcwd()) # Root directory of the project
pretrained_model = "mask_rcnn_coco.h5" # Path to trained weights file

checkpoints_dir = "logs"
train_subset = 'train'
training_folder = 'datasets'

val_subset = 'val'
json_file = "via_project.json" # same name of json_file in both train and val

############################################################
#  Configurations
############################################################

class TrainConfig(Config):
	NAME = "object" # Give the configuration a recognizable name (don't change it)

	IMAGES_PER_GPU = 1

	NUM_CLASSES = 1 + 2 # Background + number of custom classes

	STEPS_PER_EPOCH = 10

	# VALIDATION_STEPS = 50 # optional

	DETECTION_MIN_CONFIDENCE = 0.9

class InferenceConfig(TrainConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    
    GPU_COUNT = 1
    
    IMAGES_PER_GPU = 1