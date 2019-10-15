import argparse
import pandas as pd
import os
import numpy as np
import cv2
from collections import defaultdict  # noqa
from matplotlib import pyplot as plt  # noqa
from PIL import Image, ImageDraw, ImageFont


def visualize_predictions(img, labels, font, unicode_map, fontsize=50):
    """
    This function takes in a filename of an image, and the labels in the string format given in a submission csv, and returns an image with the characters and predictions annotated.
    Copied from:
    https://www.kaggle.com/anokas/kuzushiji-visualisation
    """
    # Convert annotation string to array
    labels_split = labels.split(' ')
    if len(labels_split) < 3:
      return img
    labels = np.array(labels_split).reshape(-1, 3)

    # Read image
    imsource = Image.fromarray(img).convert('RGBA')
    bbox_canvas = Image.new('RGBA', imsource.size)
    char_canvas = Image.new('RGBA', imsource.size)
    bbox_draw = ImageDraw.Draw(bbox_canvas) # Separate canvases for boxes and chars so a box doesn't cut off a character
    char_draw = ImageDraw.Draw(char_canvas)

    for codepoint, x, y in labels:
        x, y = int(x), int(y)
        char = unicode_map[codepoint] # Convert codepoint to actual unicode character

        # Draw bounding box around character, and unicode character next to it
        bbox_draw.rectangle((x-10, y-10, x+10, y+10), fill=(255, 0, 0, 255))
        char_draw.text((x+25, y-fontsize*(3/4)), char, fill=(255, 0, 0, 255), font=font)

    imsource = Image.alpha_composite(Image.alpha_composite(imsource, bbox_canvas), char_canvas)
    imsource = imsource.convert("RGB") # Remove alpha for saving in jpg format.
    return np.asarray(imsource)


def main():
  np.random.seed(123)
  parser = argparse.ArgumentParser()
  parser.add_argument('--fn', type=str)
  parser.add_argument('--show-empty', action='store_true')
  parser.add_argument('--seed', type=int, default=32)
  parser.add_argument('--height', type=int, default=1024)

  args = parser.parse_args()
  np.random.seed(args.seed)
  fontsize = 80
  unicode_map = {codepoint: char for codepoint, char in
      pd.read_csv('unicode_translation.csv').values}
  font = ImageFont.truetype('assets/NotoSansCJKjp-Regular.otf',
      fontsize, encoding='utf-8')

  sub = pd.read_csv(args.fn).fillna('')
  sub = {k: v for k, v in zip(sub['image_id'], sub['labels'])}
  img_ids = sorted(sub.keys())
  np.random.shuffle(img_ids)
  p = 0
  while True:
    img_id = img_ids[p]
    fn = 'test_images/%s.jpg' % img_id
    if not os.path.exists(fn):
      fn = 'train_images/%s.jpg' % img_id

    pred_str = sub[img_id]
    img = cv2.imread(fn, 1)
    img = visualize_predictions(img, pred_str, font, unicode_map)

    img = cv2.resize(img, (1000, 800))
    cv2.imshow("t", img)
    q = cv2.waitKey()
    if q == ord('q'):
      break
    elif q == ord('b'):
      p -= 1
    elif q == ord('s'):
      out_fn = os.path.basename(fn).replace('.jpg', '.png')
      cv2.imwrite(out_fn, img)
      print("Saved to: %s" % out_fn)
    else:
      p += 1


if __name__ == '__main__':
  main()
