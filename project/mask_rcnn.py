from os import listdir
from numpy import zeros
from numpy import asarray
from numpy import mean
from numpy import expand_dims
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
import pandas as pd
import pickle
import shutil
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

img_rows, img_cols = 1200, 675
Number_Images = len(listdir('data/images'))
Fraction_Training_set = 0.66


def data_frame_to_pickle(data_frame, annotations_dir):
    img_id = next(data_frame.iterrows())[1][0]
    image_dict = {
        'filename': img_id + '.jpg',
        'bboxes':   []
    }

    # columns = ['img_id', 'x_min', 'x_max', 'y_min', 'y_max', 'label']
    for i, row in data_frame.iterrows():
        bbox = {
            'id':    i,
            'x_min': row[1],
            'x_max': row[2],
            'y_min': row[3],
            'y_max': row[4],
            'name':  row[5],
        }
        image_dict['bboxes'].append(bbox)

    with open(annotations_dir + img_id, "wb") as outfile:
        pickle.dump(image_dict, outfile)
    outfile.close()


# class that defines and loads the dataset
class HumanInVesselDangerDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True, is_validation=False):
        # define one class
        self.add_class("dataset", 1, "Dangerous")
        self.add_class("dataset", 2, "Safe")
        # define data locations
        images_dir = dataset_dir + 'images_/'
        annotations_file = dataset_dir + 'labels.csv'
        annotations_dir = dataset_dir + 'annotations/'

        # if individual pickle files for each image haven't been generated
        if is_validation:
            if len(listdir(annotations_dir)) is 0:
                row_count = 0
                # load master csv file with 'img_id', 'x_min', 'x_max', 'y_min', 'y_max' and 'label' columns
                annotations = pd.read_csv(annotations_file, usecols=[0, 2, 3, 4, 5, 8], header=0)
                # create empty data frame
                image_annotations = pd.DataFrame(columns=['img_id', 'x_min', 'x_max', 'y_min', 'y_max', 'label'])

                annotations_dic = {}

                img_id = next(annotations.iterrows())[1][0]
                for i, row in annotations.iterrows():
                    annotations_dic[str(row[0])] = 0

                    current_img_id = str(row[0])
                    # if the current row belongs to the same image as the previous row
                    if img_id is current_img_id:
                        # add current row to data frame of same image
                        image_annotations = image_annotations.append(row)
                        # final image
                        if i is len(annotations.index):
                            # create image's pickle file
                            data_frame_to_pickle(image_annotations, annotations_dir)
                    # if the current row doesn't belongs to the same image as the previous row
                    elif img_id is not current_img_id:
                        if len(image_annotations.index) > 0:
                            data_frame_to_pickle(image_annotations, annotations_dir)
                        # reset variables for next image
                        row_count = 0
                        image_annotations = pd.DataFrame(columns=['img_id', 'x_min', 'x_max', 'y_min', 'y_max', 'label'])
                        image_annotations = image_annotations.append(row)
                        img_id = current_img_id

                    row_count += 1

                if len(image_annotations) > 0:
                    # create image's pickle file
                    data_frame_to_pickle(image_annotations, annotations_dir)

                for key in annotations_dic:
                    if not is_validation:
                        shutil.copy('data/images/' + key + '.jpg', images_dir + key + '.jpg')
                    else:
                        shutil.copy('validation/images/' + key + '.jpg', images_dir + key + '.jpg')

        img_count = 0
        for filename in listdir(images_dir):
            # extract image id
            image_id = filename[:-4]
            img_count += 1
            if not is_validation:
                # we are building the train set, 90% of data
                if is_train and img_count > len(listdir(images_dir)) * Fraction_Training_set:
                    continue
                # we are building the test/val set, 10% of data
                if not is_train and img_count <= len(listdir(images_dir)) * Fraction_Training_set:
                    continue
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id
            # add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path, class_ids=[0, 1, 2])

    # extract bounding boxes from an annotation file
    def extract_boxes(self, filename):
        # load and parse the file
        with open(filename, 'rb') as f:
            img_info = pickle.load(f)
        boxes = list()

        for box in img_info["bboxes"]:
            name = int(box['name'])
            xmin = int(box['x_min'])
            ymin = int(box['y_min'])
            xmax = int(box['x_max'])
            ymax = int(box['y_max'])
            coors = [xmin, ymin, xmax, ymax, name]
            boxes.append(coors)
        f.close()
        return boxes

    # load the masks for an image
    def load_mask(self, image_id):
        h = img_rows
        w = img_cols
        image_info = self.image_info[image_id]
        boxes = self.extract_boxes(image_info['annotation'])
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1

            if box[4]:
                class_ids.append(self.class_names.index('Dangerous'))
            else:
                class_ids.append(self.class_names.index('Safe'))

        return masks, asarray(class_ids, dtype='int32')

    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


# calculate the mAP for a model on a given dataset
def evaluate_model(dataset, model, cfg):
    APs = list()
    for image_id in dataset.image_ids:
        # load image, bounding boxes and masks for the image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = expand_dims(image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=1)

        print("image_meta: ", image_meta)
        print("gt_bbox:", gt_bbox)
        print("gt_bbox shape:", gt_bbox.shape)
        print("gt_class_id: ", gt_class_id)
        print("gt_mask: ", gt_mask.shape)
        print("-----------------------------------------------")
        # extract results for first sample
        r = yhat[0]

        print(r)
        print("rois: ", r["rois"].shape)
        print("class_ids: ", r["class_ids"])
        print("scores: ",  r["scores"])
        print("masks: ", r["masks"].shape)

        # calculate statistics, including AP
        AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        # store
        APs.append(AP)
    # calculate the mean AP across all images
    mAP = mean(APs)
    return mAP

# define a configuration for the model
class VesselConfig(Config):
    # define the name of the configuration
    NAME = "Vessel_cfg"
    # number of classes (background + 2 states)
    NUM_CLASSES = 1 + 2
    # number of training steps per epoch
    STEPS_PER_EPOCH = 1000

    IMAGES_PER_GPU = 1
    GPU_COUNT = 8


class VesselEvalConfig(Config):
    # define the name of the configuration
    NAME = "Vessel_cfg"
    # number of classes (background + 2 states)
    NUM_CLASSES = 1 + 2
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    #IMAGE_MAX_DIM = 1200



# load_test = HumanInVesselDangerDataset()
# load_test.load_dataset('data/')

# test_set = HumanInVesselDangerDataset()
# test_set.load_dataset('data/', is_train=False)
# test_set.prepare()
# print('Test: %d' % len(test_set.image_ids))

# prepare train set
train_set = HumanInVesselDangerDataset()
train_set.load_dataset('data/', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
# prepare test set
test_set = HumanInVesselDangerDataset()
test_set.load_dataset('data/', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))
# prepare validation set
validation_set = HumanInVesselDangerDataset()
validation_set.load_dataset('validation/', is_validation=True)
validation_set.prepare()
print('Validation: %d' % len(validation_set.image_ids))

# prepare config
config = VesselEvalConfig()
config.display()
# # define the model
# model = MaskRCNN(mode='training', model_dir='./models/', config=config)
# # load weights (mscoco) and exclude the output layers
# model.load_weights('models/vessel_cfg20200512T2115/mask_rcnn_vessel_cfg_0004.h5', by_name=True,
#                    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
# # train weights (output layers or 'heads')
# model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=10, layers='heads')

# define the model
model = MaskRCNN(mode='inference', model_dir='./', config=config)
# load model weights
model.load_weights('models/vessel_cfg20200512T0727/mask_rcnn_vessel_cfg_0009.h5', by_name=True)
# evaluate model on training dataset
train_mAP = evaluate_model(train_set, model, config)
print("Train mAP: %.3f" % train_mAP)
# evaluate model on test dataset
test_mAP = evaluate_model(test_set, model, config)
print("Test mAP: %.3f" % test_mAP)
# evaluate model on validation dataset
val_mAP = evaluate_model(validation_set, model, config)
print("Validation mAP: %.3f" % val_mAP)

# TODO: display actual vs predicted images
