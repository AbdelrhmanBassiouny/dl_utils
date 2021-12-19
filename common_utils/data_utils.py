import collections
import os


xywh_to_xcycwh_xyxy = {'x1': lambda x: x[0], 'x2': lambda x: x[0] + x[2], 'y1': lambda x: x[1],
                       'y2': lambda x: x[1] + x[3], 'w': lambda x: x[2], 'h': lambda x: x[3],
                       'xc': lambda x: x[0] + x[2] / 2.0, 'yc': lambda x: x[1] + x[3] / 2.0}

xcycwh_to_xywh_xyxy = {'x1': lambda x: x[0] - x[2] / 2.0, 'y1': lambda x: x[1] - x[3] / 2.0,
                       'x2': lambda x: x[0] + x[2] / 2.0, 'y2': lambda x: x[1] + x[3] / 2.0,
                       'w': lambda x: x[2], 'h': lambda x: x[3]}

xyxy_to_xywh_xcycwh = {'x1': lambda x: x[0], 'w': lambda x: x[2] - x[0], 'y1': lambda x: x[1],
                       'h': lambda x: x[3] - x[1], 'xc': lambda x: (x[0] + x[2]) / 2.0,
                       'yc': lambda x: (x[1] + x[3]) / 2.0}

yxyx_to_xywh = {'x1': lambda x: x[1], 'w': lambda x: x[3] - x[1], 'y1': lambda x: x[0],
                'h': lambda x: x[2] - x[0], 'xc': lambda x: (x[1] + x[3]) / 2.0,
                'yc': lambda x: (x[0] + x[2]) / 2.0}


def convert_format(box, in_format=('x1', 'y1', 'w', 'h'), out_format=('x1', 'y1', 'x2', 'y2')):
    if in_format == ('x1', 'y1', 'x2', 'y2') and \
       (out_format == ('x1', 'y1', 'w', 'h') or
            out_format == ('xc', 'yc', 'w', 'h')):
      return [xyxy_to_xywh_xcycwh[element](box) for element in out_format]
    elif in_format == ('xc', 'yc', 'w', 'h') and \
        (out_format == ('x1', 'y1', 'w', 'h') or
         out_format == ('x1', 'y1', 'x2', 'y2')):
      return [xcycwh_to_xywh_xyxy[element](box) for element in out_format]
    elif in_format == ('x1', 'y1', 'w', 'h') and \
        (out_format == ('x1', 'y1', 'x2', 'y2') or
         out_format == ('xc', 'yc', 'w', 'h')):
      return [xywh_to_xcycwh_xyxy[element](box) for element in out_format]
    elif in_format == ('y1', 'x1', 'y2', 'x2') and \
            (out_format == ('x1', 'y1', 'w', 'h') or
             out_format == ('xc', 'yc', 'w', 'h')):
      return [yxyx_to_xywh[element](box) for element in out_format]
    elif in_format == out_format:
      return box
    else:
      raise ValueError("Wrong Conversion")

def read_labels_from_files(directory,
                           classes_names=None,
                           in_format=None,
                           out_format=('x1', 'y1', 'w', 'h'),
                           isrel=False,
                           has_conf=True,
                           add_conf=False,
                           box_formatter=None,
                           sort=True,
                           label_change_dict=None):
  if box_formatter is None:
    def box_formatter(box): return [float(x) for x in box]

  # all_imgs_labels = []
  all_imgs_labels_dict = {}
  convert = True
  if in_format is None:
      in_format = out_format
      convert = False
  idx = -1
  for filename in os.listdir(directory):
      if not filename.endswith(".txt"):
        continue
      if filename == "labels.txt":
        continue
      idx += 1
      imgname = filename.replace('.txt', '.jpg')
      # print(imgname)
      # first get all lines from file
      with open(directory + filename, 'r') as f:
          lines = f.readlines()
      # all_imgs_labels.append({"id":imgname})
      img_labels = {}
      sorted_img_labels = []
      for line in lines:
          # remove spaces
          label = line.replace('\n', '').split(sep=' ')
          # print("label = ", label)
          clabel = int(label[0])
          if label_change_dict is not None:
            clabel = label_change_dict[clabel]
          if classes_names is not None:
            cname = classes_names[clabel]
          else:
            cname = clabel
          if cname not in img_labels:
              img_labels[cname] = []

          new_box = label[1:-1] if has_conf else label[1:]
          new_box = box_formatter(new_box)
          if convert:
            new_box = convert_format(new_box,
                                     in_format=in_format,
                                     out_format=out_format)
          sorted_img_labels.append((cname, new_box))
          img_labels[cname].append(new_box)
          if add_conf:
            if has_conf:
              img_labels[cname][-1].append(float(label[-1]))
            else:
              img_labels[cname][-1].append(1.0)

      # all_imgs_labels[idx]['labels'] = img_labels
      # all_imgs_labels_dict[imgname] = img_labels
      all_imgs_labels_dict[imgname] = sorted_img_labels
  if sort:
      all_imgs_labels_dict = dict(collections.OrderedDict(
          sorted(all_imgs_labels_dict.items(), key=lambda x: int(x[0][0:-4]))))
  all_imgs_labels = [{'id': id, 'labels': labels}
                     for id, labels in all_imgs_labels_dict.items()]
  return all_imgs_labels


def get_labels_as(all_imgs_labels,
                  in_format=('x1', 'y1', 'w', 'h'),
                  out_format=('x1', 'y1', 'w', 'h'),
                  is_rel=False, relative=False,
                  dtype='float',
                  width=None, height=None,
                  cname2cid=None,
                  imgids=False):
  
  if (width is None) or (height is None):
      width, height = [1.0]*len(all_imgs_labels), [1.0]*len(all_imgs_labels)
  new_imgs_labels = {}
  img_id = 0
  for i, (img_name, annotations) in enumerate(all_imgs_labels.items()):
    img = img_name
    if imgids:
      img = img_id
    new_imgs_labels[img] = {}
    width_ratio, height_ratio = 1.0, 1.0
    if relative and not is_rel:
        width_ratio /= width[i]
        height_ratio /= height[i]
    elif not relative and is_rel:
        width_ratio *= width[i]
        height_ratio *= height[i]
    
    if type(annotations) == dict:
      zipped_annotations = []
      for cls_name, cls_boxes in annotations.items():
        for box in cls_boxes:
          zipped_annotations.append((cls_name, box))
    else:
      zipped_annotations = annotations
            
    for cls_name, box in zipped_annotations:
      if cname2cid is not None:
        id = cname2cid[cls_name]
      new_imgs_labels[img][id] = []
      new_box = []
      new_box.append(box[0] * width_ratio)
      new_box.append(box[1] * height_ratio)
      new_box.append(box[2] * width_ratio)
      new_box.append(box[3] * height_ratio)
      if len(box) > 4:
        new_box.append(box[4])
      new_box[0:4] = convert_format(new_box[0:4], in_format=in_format,
                                    out_format=out_format)
      if dtype == 'int':
        new_box[0:4] = list(map(int, new_box[0:4]))
      new_imgs_labels[img][id].append(new_box)
    img_id += 1
  return new_imgs_labels


