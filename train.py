import torch as th
import torch.utils.data as data
from tqdm import tqdm; tqdm.monitor_interval = 0  # noqa
import os
import time
import apex
import pandas as pd  # noqa
from shutil import copyfile
from matplotlib import pyplot as plt  # noqa
import numpy as np
from glob import glob

from models import FPNSegmentation
from data import KuzushijiDataset, load_gt
from schedules import WarmupLinearSchedule
# from server import score
from submit import create_submission
from adamw import AdamW

th.backends.cudnn.benchmark = False
th.backends.cudnn.deterministic = True


def seed_all():
  SEED = 32
  np.random.seed(SEED)
  th.manual_seed(SEED)
  th.cuda.manual_seed(SEED)


def kuzushiji_loss(hm, centers, classes, hm_pred, classes_pred, weights=None):
  assert hm.shape == hm_pred.shape
  hm = hm.to(hm_pred.dtype)
  hm_loss = th.nn.functional.binary_cross_entropy_with_logits(
      hm_pred, hm, reduction='mean')

  classes_ = []
  for sample_ind in range(len(hm)):
    center = centers[sample_ind]
    center_mask = center[:, 0] != -1
    per_image_letters = center_mask.sum().item()
    if per_image_letters == 0:
      continue
    classes_per_img = classes[sample_ind][center_mask]
    classes_.append(classes_per_img)

  classes = th.cat(classes_, 0)
  classes_loss = th.nn.functional.cross_entropy(classes_pred, classes,
      reduction='mean')
  # print("hm: ", hm_loss.item(), " classes: ", classes_loss)
  total_loss = hm_loss + 0.1 * classes_loss
  return total_loss


def main(config):
  seed_all()
  os.makedirs('cache', exist_ok=True)
  os.makedirs(config.logdir, exist_ok=True)
  print("Logging to: %s" % config.logdir)
  src_files = sorted(glob('*.py'))
  for src_fn in src_files:
    dst_fn = os.path.join(config.logdir, src_fn)
    copyfile(src_fn, dst_fn)

  train_image_fns = sorted(glob(os.path.join(config.train_dir, '*.jpg')))
  test_image_fns = sorted(glob(os.path.join(config.test_dir, '*.jpg')))

  assert len(train_image_fns) == 3881
  assert len(test_image_fns) == 4150

  gt, label_to_int = load_gt(config.train_rle)
  int_to_label = {v: k for k, v in label_to_int.items()}
  # create folds
  np.random.shuffle(train_image_fns)

  if config.subset > 0:
    train_image_fns = train_image_fns[:config.subset]

  folds = np.arange(len(train_image_fns)) % config.num_folds
  val_image_fns = [fn for k, fn in enumerate(train_image_fns)
      if folds[k] == config.fold]
  train_image_fns = [fn for k, fn in enumerate(train_image_fns)
      if folds[k] != config.fold]

  if config.add_val:
    print("Training on validation set")
    train_image_fns = train_image_fns + val_image_fns[:]

  print(len(val_image_fns), len(train_image_fns))

  # TODO: drop empty images <- is this helpful?
  train_image_fns = [fn for fn in train_image_fns
      if KuzushijiDataset.fn_to_id(fn) in gt]
  val_image_fns = [fn for fn in val_image_fns
      if KuzushijiDataset.fn_to_id(fn) in gt]

  print("VAL: ", len(val_image_fns), val_image_fns[123])
  print("TRAIN: ", len(train_image_fns), train_image_fns[456])

  train_ds = KuzushijiDataset(train_image_fns, gt_boxes=gt,
      label_to_int=label_to_int, augment=True)
  val_ds = KuzushijiDataset(val_image_fns, gt_boxes=gt,
      label_to_int=label_to_int)

  if config.cache:
    train_ds.cache()
    val_ds.cache()

  val_loader = data.DataLoader(val_ds, batch_size=config.batch_size // 8,
                               shuffle=False, num_workers=config.num_workers,
                               pin_memory=config.pin, drop_last=False)

  model = FPNSegmentation(config.slug)
  if config.weight is not None:
    print("Loading: %s" % config.weight)
    model.load_state_dict(th.load(config.weight))
  model = model.to(config.device)

  no_decay = ['mean', 'std', 'bias'] + ['.bn%d.' % i for i in range(100)]
  grouped_parameters = [{'params': [], 'weight_decay': config.weight_decay},
      {'params': [], 'weight_decay': 0.0}]
  for n, p in model.named_parameters():
    if not any(nd in n for nd in no_decay):
      # print("Decay: %s" % n)
      grouped_parameters[0]['params'].append(p)
    else:
      # print("No Decay: %s" % n)
      grouped_parameters[1]['params'].append(p)
  optimizer = AdamW(grouped_parameters, lr=config.lr)

  if config.apex:
    model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1",
                                           verbosity=0)

  updates_per_epoch = len(train_ds) // config.batch_size
  num_updates = int(config.epochs * updates_per_epoch)
  scheduler = WarmupLinearSchedule(warmup=config.warmup, t_total=num_updates)

  # training loop
  smooth = 0.1
  best_acc = 0.0
  best_fn = None
  global_step = 0
  for epoch in range(1, config.epochs + 1):
    smooth_loss = None
    smooth_accuracy = None
    model.train()
    train_loader = data.DataLoader(train_ds, batch_size=config.batch_size,
                                   shuffle=True, num_workers=config.num_workers,
                                   pin_memory=config.pin, drop_last=True)
    progress = tqdm(total=len(train_ds), smoothing=0.01)
    if True:
      for i, (X, fns, hm, centers, classes) in enumerate(train_loader):
        X = X.to(config.device).float()
        hm = hm.to(config.device)
        centers = centers.to(config.device)
        classes = classes.to(config.device)
        hm_pred, classes_pred = model(X, centers=centers)
        loss = kuzushiji_loss(hm, centers, classes, hm_pred, classes_pred)
        if config.apex:
          with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        else:
          loss.backward()

        lr_this_step = None
        if (i + 1) % config.accumulation_step == 0:
          optimizer.step()
          optimizer.zero_grad()
          lr_this_step = config.lr * scheduler.get_lr(global_step,
              config.warmup)
          for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_step
          global_step += 1

        smooth_loss = loss.item() if smooth_loss is None else \
            smooth * loss.item() + (1. - smooth) * smooth_loss
        # print((y_true >= 0.5).sum().item())
        accuracy = th.mean(((th.sigmoid(hm_pred) >= 0.5) == (hm == 1)).to(
            th.float)).item()
        smooth_accuracy = accuracy if smooth_accuracy is None else \
            smooth * accuracy + (1. - smooth) * smooth_accuracy
        progress.set_postfix(ep='%d/%d' % (epoch, config.epochs),
              loss='%.4f' % smooth_loss, accuracy='%.4f' %
              (smooth_accuracy), lr='%.6f' % (config.lr if lr_this_step is None
                else lr_this_step))
        progress.update(len(X))

    # skip validation
    if epoch not in [10, 20, 30, 40, 50]:
      if 1 < epoch <= 65:
        continue

    # validation loop
    model.eval()
    progress = tqdm(enumerate(val_loader), total=len(val_loader))
    hm_correct, classes_correct = 0, 0
    num_hm, num_classes = 0, 0
    with th.no_grad():
      for i, (X, fns, hm, centers, classes) in progress:
        X = X.to(config.device).float()
        hm = hm.cuda()
        centers = centers.cuda()
        classes = classes.cuda()
        hm_pred, classes_pred = model(X)
        hm_pred = th.sigmoid(hm_pred)
        classes_pred = th.nn.functional.softmax(classes_pred, 1)
        hm_cuda = hm.cuda()
        # PyTorch 1.2 has `bool`
        if hasattr(hm_cuda, 'bool'):
          hm_cuda = hm_cuda.bool()
        hm_correct += (hm_cuda == (hm_pred >= 0.5)).float().sum().item()
        num_hm += np.prod(hm.shape)
        num_samples = len(X)
        for sample_ind in range(num_samples):
          center_mask = centers[sample_ind, :, 0] != -1
          per_image_letters = center_mask.sum().item()
          if per_image_letters == 0:
            continue
          num_classes += per_image_letters
          centers_per_img = centers[sample_ind][center_mask]
          classes_per_img = classes[sample_ind][center_mask]
          classes_per_img_pred = classes_pred[sample_ind][
              :, centers_per_img[:, 1], centers_per_img[:, 0]].argmax(0)
          classes_correct += (classes_per_img_pred ==
              classes_per_img).sum().item()
          num_classes += per_image_letters

    val_hm_acc = hm_correct / num_hm
    val_classes_acc = classes_correct / num_classes
    summary_str = 'f%02d-ep-%04d-val_hm_acc-%.4f-val_classes_acc-%.4f' % (
        config.fold, epoch, val_hm_acc, val_classes_acc)

    progress.write(summary_str)
    if val_classes_acc >= best_acc:
      weight_fn = os.path.join(config.logdir, summary_str + '.pth')
      progress.write("New best: %s" % weight_fn)
      th.save(model.state_dict(), weight_fn)
      best_acc = val_classes_acc
      best_fn = weight_fn
      fns = sorted(glob(os.path.join(config.logdir, 'f%02d-*.pth' %
          config.fold)))
      for fn in fns[:-config.n_keep]:
        os.remove(fn)

  # create submission
  test_ds = KuzushijiDataset(test_image_fns)
  test_loader = data.DataLoader(test_ds, batch_size=config.batch_size // 8,
                               shuffle=False, num_workers=config.num_workers,
                               pin_memory=False, drop_last=False)
  if best_fn is not None:
    model.load_state_dict(th.load(best_fn))
  model.eval()
  sub = create_submission(model, test_loader, int_to_label, config,
      pred_zip=config.pred_zip)
  sub.to_csv(config.submission_fn, index=False)
  print("Wrote to: %s" % config.submission_fn)

  # create val submission
  val_fn = config.submission_fn.replace('.csv', '_VAL.csv')
  model.eval()
  sub = []
  sub = create_submission(model, val_loader, int_to_label, config,
      pred_zip=config.pred_zip.replace('.zip', '_VAL.zip'))
  sub.to_csv(val_fn, index=False)
  print("Wrote to: %s" % val_fn)


class Config:
  def as_dict(self):
    return vars(self)
  def __str__(self):
    return str(self.as_dict())
  def __repr__(self):
    return str(self)


if __name__ == '__main__':
  config = Config()
  config.id = 38
  config.train_dir = 'train_images'
  config.test_dir = 'test_images'
  config.sample_submission = 'sample_submission.csv'
  config.train_rle = 'train.csv'
  config.epochs = 125
  config.batch_size = 8  # TODO
  config.lr = 5e-5  # 1e-4
  config.weight_decay = 1e-4
  config.weight = None
  config.warmup = 0.03
  config.accumulation_step = 1
  config.num_folds = 10
  config.num_workers = 8
  config.p_clf = 0.6
  config.p_seg = 0.2
  config.pin = False
  config.slug = 'r101d'  # TODO
  config.device = 'cuda'
  config.apex = True  # TODO
  config.subset = -1  # TODO
  config.n_keep = 1
  config.is_kernel = False
  config.cache = True
  config.p_letter = 0.1
  config.p_class = 0.1
  config.add_val = True

  for fold in range(config.num_folds):
    tic = time.time()
    config.fold = fold
    config.logdir = 'Logdir_%03d_f%02d' % (config.id, config.fold)
    config.pred_zip = os.path.join(config.logdir, 'f%02d-PREDS.zip' % (
        config.fold))
    config.submission_fn = os.path.join(config.logdir,
        'Logdir_%d_f%02d.csv' % (config.id, config.fold))
    print(config)
    main(config)
    print("[Fold %d] Duration: %.3f mins" % (config.fold,
        (time.time() - tic) / 60))
