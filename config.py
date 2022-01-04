from mrcnn.config import Config

ROOT_DIR = "root directory of this repo"
pretrained_model = "mask_rcnn_coco.h5"
checkpoints_dir = "logs"
dataset_folder = "dataset" #dataset folder path
train_subset = 'train'
val_subset = 'val'
json_file = "via_project.json" # same name of json_file in both train and val

class CustomConfig(Config):
	Name = "object" # Give the configuration a recognizable name (don't change it)
	IMAGES_PER_GPU = 1
	NUM_CLASSES = 1 + 2 # Background + number of custom classes 
	STEPS_PER_EPOCH = 10
	DETECTION_MIN_CONFIDENCE = 0.9