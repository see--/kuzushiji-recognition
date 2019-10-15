import torch as th
from collections import defaultdict  # noqa
from tqdm import tqdm
import torch.utils.data as td
import os
import cv2
# import apex
from PIL import Image
from matplotlib import pyplot as plt  # noqa
import pandas as pd
from zipfile import ZipFile
import numpy as np  # noqa
from glob import glob

from models import FPNSegmentation
from data import KuzushijiDataset, MultiScaleInferenceKuzushijiDataset, load_gt


def seed_all():
  SEED = 32
  np.random.seed(SEED)
  th.manual_seed(SEED)
  th.cuda.manual_seed(SEED)


def nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = th.nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def create_submission(model, loader, int_to_label, config, pred_zip=None,
      tta=False, base_h=1536, base_w=1536):
  if pred_zip is not None and os.path.exists(pred_zip):
    os.remove(pred_zip)

  sub = []
  model.eval()
  image_ids, labels = [], []
  with th.no_grad():
    for ret in tqdm(loader):
      Xs = ret[:-1]
      fns = ret[-1]
      hm_preds, classes_embeddings = 0., 0.
      # scale TTA
      for X in Xs:
        X = X.to(config.device).float()
        hm_pred, classes_embedding = model(X, return_embeddings=True)
        hm_pred = th.nn.functional.interpolate(hm_pred, size=(base_h // 4,
            base_w // 4), mode='bilinear', align_corners=True)
        classes_embedding = th.nn.functional.interpolate(classes_embedding,
            size=(base_h // 4, base_w // 4), mode='bilinear',
            align_corners=True)
        hm_pred = th.sigmoid(hm_pred)
        hm_preds += hm_pred
        classes_embeddings += classes_embedding

      hm_pred = hm_preds / len(Xs)
      classes_embeddings /= len(Xs)
      hm_pred = nms(hm_pred).cpu().numpy()
      for j in range(len(hm_pred)):
        img_id = KuzushijiDataset.fn_to_id(fns[j])
        w, h = Image.open(fns[j]).size
        pad_top, pad_bottom, pad_left, pad_right = \
            KuzushijiDataset.get_paddings(h, w, ratio=1.5)
        h = h + pad_top + pad_bottom
        w = w + pad_left + pad_right
        image_ids.append(img_id)
        hmp = hm_pred[j].squeeze(0)
        if pred_zip is not None:
          pred_fn = img_id + '.png'
          yp_img = np.uint8(hmp * 255)
          img_bytes = cv2.imencode('.png', yp_img)[1].tobytes()
          with ZipFile(pred_zip, 'a') as f:
            f.writestr(pred_fn, img_bytes)

        hmp_thresholded = hmp > config.p_letter
        ys, xs = np.where(hmp_thresholded)
        if len(ys) == 0:
          labels.append('')
          continue

        centers = th.stack([th.tensor(xs), th.tensor(ys)], -1).unsqueeze(0)
        gathered_embeddings = model.gather_embeddings(classes_embeddings[
            j: j + 1], centers)
        classes_preds = model.classes(gathered_embeddings.unsqueeze(
            -1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        # set prob for extra letters to zero
        classes_preds = classes_preds[:, :4212]
        classes_preds = th.nn.functional.softmax(classes_preds, 1).cpu().numpy()

        per_image_labels = []
        for center_ind in range(len(ys)):
          x, y = xs[center_ind], ys[center_ind]
          pred_probs = classes_preds[center_ind]
          # scale to (padded) image coords
          x, y = int(round(x * 4 * (w / base_w))), int(round(y * 4 * (
              h / base_h)))
          # undo padding
          x -= pad_left
          y -= pad_top
          # pred_classes = np.where(pred_probs > config.p_class)[0]
          pred_classes = [pred_probs.argmax()]
          # print(len(pred_classes))
          for pred_class in pred_classes:
            ll = int_to_label[pred_class]
            per_image_labels.extend([ll, str(x), str(y)])

        image_label_str = ' '.join(per_image_labels)
        labels.append(image_label_str)
        # img = cv2.imread('test_images/%s.jpg' % (img_id), 1)
        # plt.subplot(2, 1, 1)
        # plt.imshow(img)
        # plt.subplot(2, 1, 2)
        # plt.imshow(hmp.squeeze())
        # plt.show()

  sub = pd.DataFrame({'image_id': image_ids, 'labels': labels})
  return sub


def main(config):
  seed_all()
  train_image_fns = sorted(glob(os.path.join(config.train_dir, '*.jpg')))
  test_image_fns = sorted(glob(os.path.join(config.test_dir, '*.jpg')))

  gt, label_to_int = load_gt(config.train_csv)
  int_to_label = {v: k for k, v in label_to_int.items()}
  # create folds
  np.random.shuffle(train_image_fns)

  folds = np.arange(len(train_image_fns)) % config.num_folds
  val_image_fns = [fn for k, fn in enumerate(train_image_fns)
      if folds[k] == config.fold]
  train_image_fns = [fn for k, fn in enumerate(train_image_fns)
      if folds[k] != config.fold]

  # TODO: drop empty images <- is this helpful?
  train_image_fns = [fn for fn in train_image_fns
      if KuzushijiDataset.fn_to_id(fn) in gt]
  val_image_fns = [fn for fn in val_image_fns
      if KuzushijiDataset.fn_to_id(fn) in gt]

  print("VAL: ", len(val_image_fns), val_image_fns[123])
  print("TRAIN: ", len(train_image_fns), train_image_fns[456])
  if config.submit_val:
    test_image_fns = val_image_fns

  test_ds = MultiScaleInferenceKuzushijiDataset(test_image_fns, 1536, 1536,
      config.scales)

  test_loader = td.DataLoader(test_ds, batch_size=config.batch_size,
                              shuffle=False, num_workers=0,
                              pin_memory=False, drop_last=False)

  model = FPNSegmentation(config.slug, pretrained=False)
  print("Loading: %s" % config.weight)
  model.load_state_dict(th.load(config.weight))
  model = model.to(config.device)
  # model = apex.amp.initialize(model, opt_level="O1")
  model.eval()
  sub = create_submission(model, test_loader, int_to_label, config,
      pred_zip=config.pred_zip, tta=config.tta)
  sub.to_csv(config.submission_fn, index=False)
  print("Wrote to %s" % config.submission_fn)


class Config:
  def as_dict(self):
    return vars(self)
  def __str__(self):
    return str(self.as_dict())
  def __repr__(self):
    return str(self)


if __name__ == '__main__':
  config = Config()
  config.train_dir = 'train_images'
  config.test_dir = 'test_images'
  config.batch_size = 2
  config.fold = 0
  config.num_folds = 10
  config.device = 'cuda'
  config.p_letter = 0.4
  config.p_class = 0.3
  config.tta = True
  config.scales = [0.75, 1.0, 1.25]
  config.train_csv = 'train.csv'
  config.slug = 'r101d'
  config.weight = 'Logdir_038_f00/f00-ep-0125-val_hm_acc-0.9944-val_classes_acc-0.4986.pth'  # noqa
  for submit_val in [True, False]:
    config.submit_val = submit_val
    dn = os.path.dirname(config.weight)
    bn = os.path.basename(config.weight)
    config.submission_fn = os.path.join(dn, bn.split('-')[0] +
        '%s-PREDS-p%.2f-%s.csv' % ('-TTA-V7' if config.tta else '-V5',
        config.p_letter, '_VAL' if config.submit_val else ''))
    print("Saving to: %s" % config.submission_fn)
    config.pred_zip = None  # config.submission_fn.replace('.csv', '.zip')
    print(config)
    main(config)
