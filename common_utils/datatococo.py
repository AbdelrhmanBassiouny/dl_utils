import datetime
import os
import glob
from natsort import natsorted
import json
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
from pycocotools.coco import COCO


class COCOJsonGenerator:
    def __init__(self, images_dir, annotations, categories,
     write_to=None, new_size=None, filename='coco_data', detections=False):
        """
        images_path: a str as the directory containing images,
        annotations: dict {imgid: {clsid1: [boxes], cls_id2: [boxes]}}
                     if detections=True add a score value in each box's last index.
        categories: dict mapping class_ids to class_names {class_id: "class_name"}
        returns coco metrics (Also gives the ability to use cocoeval APIs on given data).
        """
        self.dataset = {}
        self.dataset["images"] = []
        self.dataset["annotations"] = []
        self.dataset["categories"] = []

        self.images_paths = glob.glob(os.path.join(images_dir, "*.jpg"))
        self.images_paths = natsorted(self.images_paths)
        self.all_imgs_annotations = annotations
        self.categories = categories
        self.new_size = new_size
        self.detections = detections

        self.dataset['info'] = {
            "description": "laptop_components_as_box_objects Dataset",
            "url": "",
            "version": "0.1.0",
            "year": 2021,
            "contributor": "AbdelrhmanBassiouny",
            "date_created": datetime.datetime.utcnow().isoformat(' ')
        }

        self.dataset['licenses'] = [
            {
                "id": 1,
                "name": "",
                "url": ""
            }]

        self._create_categories_info()
        self._create_images_info()
        self._create_annotations_info()

        self.coco_dataset = COCO()
        self.coco_dataset.dataset = self.dataset
        self.coco_dataset.createIndex()

        # print(json.dumps(self.dataset, indent=4))
        if write_to is not None:
            with open(os.path.join(write_to, filename+'.json'), 'w') as outfile:
                json.dump(self.dataset, outfile, indent=4)
        
    def _create_images_info(self):
        # go through each image
        for image_id, image_filename in enumerate(self.images_paths):
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            self.dataset["images"].append(image_info)
    
    def _create_categories_info(self):
        for id, name in self.categories.items():

            self.dataset['categories'].append({
                'id': id,
                'name': name,
                'supercategory': "object",
            })
    
    def _create_annotations_info(self):
        # go through each associated annotation
        ann_id = 1
        for image_id, annotations in self.all_imgs_annotations.items():
            for cls_id, boxes in annotations.items():
                if cls_id not in self.categories.keys():
                    continue
                for box in boxes:
                    category_info = {'id': cls_id, 'is_crowd': 0}
                    print
                    img_size = (self.dataset['images'][image_id]["height"],
                     self.dataset['images'][image_id]["width"])
                    binary_mask = np.zeros(img_size, dtype=np.uint8)
                    binary_mask[box[1]: box[1]+box[3], box[0]: box[0]+box[2]] = 1
                    annotation_info = pycococreatortools.create_annotation_info(
                        ann_id, image_id, category_info, binary_mask, self.new_size, tolerance=2)
                    if annotation_info is not None:
                        if self.detections:
                            annotation_info['score'] = box[4]
                        self.dataset["annotations"].append(annotation_info)
                    ann_id += 1        


if __name__ == "__main__":
    gt_ann = {
        0: {0: [[0, 1, 3, 4]], 1: [[4, 5, 7, 8]]},
        1: {0: [[5, 6, 9, 10]], 1: [[6, 7, 11, 12]]}
        }
    det_ann = {
        0: {0: [[0, 1, 3, 4, 0.5]]},
        1: {1: [[6, 7, 11, 12, 0.5]]}
        }
    test_cat = {0: "apple", 1: "pie"}
    coco_dset = COCOJsonGenerator(
        "/home/abdelrhman/TensorFlow/workspace/training_demo/images/test", det_ann, test_cat, detections=True).coco_dataset
    print(coco_dset.loadAnns([1]))
