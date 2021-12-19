from datatococo import COCOJsonGenerator
from pycocotools.cocoeval import COCOeval


def display_coco_metrics(images_dir, groundtruths, detections, categories, new_size=None, write_to=None, per_class=True, overall=True):

    def display_category_coco_metrics(categories_to_display):
        if len(categories_to_display.keys()) > 1:
            cat_name = "all classes"
        else:
            cat_name = "class {}".format(list(categories_to_display.values())[0])
        string_to_print = "Metrics for {} :".format(cat_name)
        print()
        print("="*len(string_to_print))
        print(string_to_print)
        print("="*len(string_to_print))
        print()
        cocoGt = COCOJsonGenerator(images_dir, groundtruths, categories_to_display,
                                new_size=new_size, write_to=write_to, filename='gt').coco_dataset
        cocoDt = COCOJsonGenerator(images_dir, detections, categories_to_display,
                                new_size=new_size, detections=True, write_to=write_to, filename='det').coco_dataset
        eval = COCOeval(cocoGt, cocoDt, iouType='bbox')
        eval.evaluate()
        eval.accumulate()
        eval.summarize()

    if per_class:
        for cat_id, cat_name in categories.items():
            display_category_coco_metrics({cat_id: cat_name})
    if overall:
        display_category_coco_metrics(categories)


if __name__ == "__main__":
    gt_ann = {
        0: {0: [[0, 1, 3, 4, 1], [5, 6, 9, 10, 1]], 1: [[4, 5, 7, 8, 1], [6, 7, 11, 12, 1]]},
        1: {0: [[0, 1, 3, 4, 1], [5, 6, 9, 10, 1]], 1: [[4, 5, 7, 8, 1], [6, 7, 11, 12, 1]]}
    }
    det_ann = {
        0: {0: [[0, 1, 3, 4, 0.99]], 1: [[4, 5, 7, 8, 0.99], [6, 7, 11, 12, 0.99]]},
        1: {0: [[0, 1, 3, 4, 0.99], [5, 6, 9, 10, 0.99]], 1: [[4, 5, 7, 8, 0.99], [6, 7, 11, 12, 0.99]]}
    }
    test_cat = {0: "apple", 1: "pie"}
    coco_dset = display_coco_metrics(
        "/home/abdelrhman/TensorFlow/workspace/training_demo/images/test", gt_ann, det_ann, test_cat)
