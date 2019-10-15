import numpy as np
import cv2
from glob import glob
import torch.utils.data as data
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm; tqdm.monitor_interval = 0  # noqa
import pandas as pd
from collections import defaultdict, Counter  # noqa
import torch as th

from albumentations import Compose, ShiftScaleRotate, RandomCrop, BboxParams, \
    ToGray, CLAHE, GaussNoise, GaussianBlur, RandomGamma, \
    RandomBrightnessContrast, RGBShift, HueSaturationValue  # noqa
# datetime.utcfromtimestamp(float('.'.join(a.iloc[0, 0].split('.')[-2:]))).
# strftime(('%Y-%m-%d %H:%M:%S'))


class KuzushijiDataset(data.Dataset):
  def __init__(self, image_fns, gt_boxes=None,
               label_to_int=None,
               augment=False,
               train_image_dir='train_images',
               test_image_dir='test_images',
               height=1536,
               width=1536,
               feature_scale=0.25):
    self.image_fns = image_fns
    self.gt_boxes = gt_boxes
    self.label_to_int = label_to_int
    self.augment = augment
    self.aug = Compose([
      ShiftScaleRotate(p=0.9, rotate_limit=10,
          scale_limit=0.2, border_mode=cv2.BORDER_CONSTANT),
      RandomCrop(512, 512, p=1.0),
      ToGray(),
      CLAHE(),
      GaussNoise(),
      GaussianBlur(),
      RandomBrightnessContrast(),
      RandomGamma(),
      RGBShift(),
      HueSaturationValue(),
    ], bbox_params=BboxParams(format='coco', min_visibility=0.75))

    self.encoded_cache = None
    self.height = height
    self.width = width
    self.feature_scale = feature_scale

  def cache(self):
    self.encoded_cache = {}
    print("Caching ... ")
    with ThreadPoolExecutor() as e:
      encoded_imgs = list(tqdm(e.map(self.read_encoded, self.image_fns),
          total=len(self.image_fns)))
    for fn, encoded in zip(self.image_fns, encoded_imgs):
      self.encoded_cache[fn] = encoded

  def read_encoded(self, fn):
    with open(fn, 'rb') as f:
      img_bytes = np.frombuffer(f.read(), dtype=np.uint8)
    return img_bytes

  @staticmethod
  def fn_to_id(fn):
    return os.path.splitext(os.path.basename(fn))[0]

  def boxes_to_mask_centers_classes(self, boxes, height=1024, width=1024,
        merge_masks=True, scale_x=1, scale_y=1, feature_scale=0.25,
        num_max_samples=620 * 9):
    mask = np.zeros((int(height * feature_scale), int(width * feature_scale)),
        dtype=np.float32)
    centers = -1 * np.ones((num_max_samples, 2), dtype=np.int64)
    classes = -1 * np.ones((num_max_samples, ), dtype=np.int64)
    # pad mask
    mask = np.pad(mask, ((1, 1), (1, 1)), mode='constant')
    pos_kernel = np.float32([[0.5, 0.75, 0.5],
                             [0.75, 1.0, 0.75],
                             [0.5, 0.75, 0.5]])
    center_ind = 0
    for box in boxes:
      x, y, w, h, ll = box
      cx, cy = feature_scale * scale_x * (x + w / 2), \
          feature_scale * scale_y * (y + h / 2)
      cx, cy = int(round(cx)), int(round(cy))
      # drop out of mask centers
      if cy >= (mask.shape[0] - 2) or cx >= (mask.shape[1] - 2):
        continue
      # mask[cy: cy + 1, cx: cx + 1] = 1
      mask[cy: cy + 3, cx: cx + 3] = pos_kernel
      label_int = self.label_to_int[ll]
      for x_offset in [-1, 0, 1]:
        cxx = cx + x_offset
        if cxx < 0 or cxx >= mask.shape[1] - 2:
          continue
        for y_offset in [-1, 0, 1]:
          cyy = cy + y_offset
          if cyy < 0 or cyy >= mask.shape[0] - 2:
            continue

          centers[center_ind] = cxx, cyy
          classes[center_ind] = label_int
          center_ind += 1

    # remove padding
    mask = mask[1: -1, 1: -1]
    return mask, centers, classes

  @staticmethod
  def mask_to_rle(img, mask_value=255, transpose=True):
    img = np.int32(img)
    if transpose:
      img = img.T
    img = img.flatten()
    img[img == mask_value] = 1
    pimg = np.pad(img, 1, mode='constant')
    diff = np.diff(pimg)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    rle = []
    previous_end = 0
    for start, end in zip(starts, ends):
      relative_start = start - previous_end
      length = end - start
      previous_end = end
      rle.append(str(relative_start))
      rle.append(str(length))
    if len(rle) == 0:
      return "-1"
    return " ".join(rle)

  @staticmethod
  def get_paddings(h, w, ratio):
    current_ratio = h / w
    # pad height
    if current_ratio < ratio:
      pad_h = int(w * ratio - h)
      pad_top = pad_h // 2
      pad_bottom = pad_h - pad_top
      pad_left, pad_right = 0, 0
    # pad width
    else:
      pad_w = int(h / ratio - w)
      pad_left = pad_w // 2
      pad_right = pad_w - pad_left
      pad_top, pad_bottom = 0, 0

    return pad_top, pad_bottom, pad_left, pad_right

  @staticmethod
  def pad_to_ratio(img, ratio):
    h, w = img.shape[:2]
    pad_top, pad_bottom, pad_left, pad_right = KuzushijiDataset.get_paddings(
        h, w, ratio)
    paddings = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    img = np.pad(img, paddings, mode='constant')
    return img, pad_top, pad_left

  def __getitem__(self, index, to_tensor=True):
    fn = self.image_fns[index]

    if self.encoded_cache is not None:
      encoded_img = self.encoded_cache[fn]
      img = cv2.imdecode(encoded_img, 1)
    else:
      img = cv2.cvtColor(cv2.imread(fn, 1), cv2.COLOR_BGR2RGB)

    img, pad_top, pad_left = self.pad_to_ratio(img, ratio=1.5)
    h, w = img.shape[:2]
    # print(h / w, pad_left, pad_top)
    assert img.ndim == 3
    img = cv2.resize(img, (self.width, self.height))
    scale_x, scale_y = self.width / w, self.height / h

    if self.gt_boxes is not None:
      gt_boxes = self.gt_boxes[self.fn_to_id(fn)][:]
      # rescale boxes
      for box_ind in range(len(gt_boxes)):
        x_min, y_min, box_w, box_h, ll = gt_boxes[box_ind]
        # correct padding
        x_min += pad_left
        y_min += pad_top
        # correct scale
        x_min, box_w = x_min * scale_x, box_w * scale_x
        y_min, box_h = y_min * scale_y, box_h * scale_y
        if box_w > self.width - x_min:
          print("W out")
          box_w = self.width - x_min
        if box_h > self.height - y_min:
          print("H out")
          box_h = self.height - y_min

        gt_boxes[box_ind] = (x_min, y_min, box_w, box_h, ll)

      if self.augment:
        augmented = self.aug(image=img, bboxes=gt_boxes)
        img, gt_boxes = augmented['image'], augmented['bboxes']

      curr_h, curr_w = img.shape[:2]
      mask, centers, classes = self.boxes_to_mask_centers_classes(
          gt_boxes, height=curr_h, width=curr_w,
          scale_x=1.0, scale_y=1.0, feature_scale=self.feature_scale)

      # if flip_lr:
      #   img = img[:, ::-1]
      #   mask = mask[:, ::-1]
      #   center_mask = centers[:, 0] >= 0
      #   centers[center_mask, 0] = (mask.shape[1] - 1) - centers[center_mask, 0]
      #   classes[center_mask] += 4212

      if to_tensor:
        img = img.transpose((2, 0, 1))
        img = th.from_numpy(img.copy())

        mask = np.expand_dims(mask, 0)
        mask = th.from_numpy(mask.copy())

        centers = th.from_numpy(centers)
        classes = th.from_numpy(classes)

      return img, fn, mask, centers, classes

    assert not self.augment, "Don't"
    if to_tensor:
      if img.ndim == 2:
        img = np.expand_dims(img, 0)
      elif img.ndim == 3:
        img = img.transpose((2, 0, 1))
      else:
        assert False, img.ndim
      img = th.from_numpy(img.copy())

    return img, fn

  def __len__(self):
    return len(self.image_fns)


class MultiScaleInferenceKuzushijiDataset(data.Dataset):
  def __init__(self, image_fns, height, width, scales):
    self.image_fns = image_fns.copy()
    self.height = height
    self.width = width
    self.scales = scales

  def __getitem__(self, index, to_tensor=True):
    fn = self.image_fns[index]
    img = cv2.cvtColor(cv2.imread(fn, 1), cv2.COLOR_BGR2RGB)

    img, pad_top, pad_left = KuzushijiDataset.pad_to_ratio(img, ratio=1.5)
    h, w = img.shape[:2]
    # print(h / w, pad_left, pad_top)
    assert img.ndim == 3
    scaled_imgs = []
    for scale in self.scales:
      h_scale = int(scale * self.height)
      w_scale = int(scale * self.width)
      simg = cv2.resize(img, (w_scale, h_scale))

      if to_tensor:
        assert simg.ndim == 3, simg.ndim
        simg = simg.transpose((2, 0, 1))
        simg = th.from_numpy(simg.copy())

      scaled_imgs.append(simg)

    return scaled_imgs + [fn]

  def __len__(self):
    return len(self.image_fns)


def load_gt(fn, label_key='labels', has_height_width=True):
  labels = pd.read_csv(fn, dtype={'image_id': str, label_key: str})
  labels = labels.fillna('')
  labels_ = defaultdict(list)
  all_labels = set()
  for img_id, label_str in zip(labels['image_id'], labels[label_key]):
    img_labels = label_str.split(' ')
    if has_height_width:
      l, x, y, h, w = img_labels[::5], img_labels[1::5], img_labels[2::5], \
          img_labels[3::5], img_labels[4::5]
      for ll, xx, yy, hh, ww in zip(l, x, y, h, w):
        labels_[img_id].append((int(xx), int(yy), int(hh), int(ww), ll))
        all_labels.add(ll)
    else:
      l, x, y = img_labels[::3], img_labels[1::3], img_labels[2::3]
      for ll, xx, yy in zip(l, x, y):
        labels_[img_id].append((int(xx), int(yy), ll))
        all_labels.add(ll)

  label_to_int = {v: k for k, v in enumerate(sorted(list(all_labels)))}
  labels = dict(labels_)
  return labels, label_to_int


if __name__ == '__main__':
  from matplotlib import pyplot as plt
  np.random.seed(321)
  th.manual_seed(321)
  gt, label_to_int = load_gt('train.csv')
  train_image_fns = sorted(glob(os.path.join('train_images', '*.jpg')))
  test_image_fns = sorted(glob(os.path.join('test_images', '*.jpg')))

  # remove empty masks from training data
  non_empty_gt = {k: v for k, v in gt.items() if '-1' not in v[0]}
  train_image_fns = [fn for fn in train_image_fns if
      KuzushijiDataset.fn_to_id(fn) in non_empty_gt]
  # subset
  train_image_fns = train_image_fns[50:60]
  print("[Non-EMPTY] TRAIN: ", len(train_image_fns), os.path.basename(
      train_image_fns[0]))
  train_ds = KuzushijiDataset(train_image_fns, gt_boxes=gt,
      label_to_int=label_to_int, augment=True)
  train_ds.cache()

  for k in range(500):
    index = np.random.randint(len(train_ds))
    ret = train_ds.__getitem__(index)
    img, fn, mask, centers, classes = ret
    img, mask = img.squeeze(), mask.squeeze()
    img = img.permute(1, 2, 0).numpy()
    print(img.shape, mask.shape)
    mask = mask.numpy()
    # h, w = mask.shape[:2]
    # mask = cv2.resize(mask, (w * 4, h * 4), interpolation=cv2.INTER_NEAREST)
    plt.imshow(mask)
    nm = 0
    classes_mask = np.zeros_like(mask, dtype=np.int32)
    for cind, (x, y) in enumerate(centers):
      if x == -1:
        break
      nm += 1
      classes_mask[y, x] = classes[cind]

    print(mask.shape, img.shape)
    mask = np.dstack([mask] * 3)
    print(np.unique(classes_mask[classes_mask > 0]))
    classes_mask = np.dstack([classes_mask] * 3)
    h, w = img.shape[:2]
    classes_mask = cv2.resize(classes_mask, (w, h),
        interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = 255 - (mask * 255).astype(np.uint8)
    mask = cv2.addWeighted(img, 0.7, mask, 0.3, 0.0)
    vis = np.hstack([img, mask])
    h, w = vis.shape[:2]
    # vis = cv2.resize(vis, (w // 2, h // 2))
    cv2.imshow("t", vis)
    # cv2.imshow("c", classes_mask.astype(np.uint8))
    q = cv2.waitKey()
    if q == ord('q'):
      break
