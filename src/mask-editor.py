import os
import sys
import logging
import json

import numpy as np
import cv2 as cv

from time import perf_counter as clock
from os.path import basename
from datetime import datetime

from mask_stats import hash_np_array
from undo import History
from blends import alpha_blend, apply_colourmap


# setup logging
logger = logging.getLogger(__name__)

# disable loggers
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


# global vars
# FIXME: take a conf file
WINDOW_TITLE = "mask-edit"
CURRENT_LABEL = 1
LABEL_BACKGROUND = 0
SHAPE_COLOUR = (1.0, 1.0, 1.0)
SHAPE_SIZE = 6
SHAPE_SIZE_INC = 1
BLEND_ALPHA = 0.25
MASK_THRESHOLD = 0.5
DRAW_MODE = "point" # (point|line)

# initial values of global vars
mouse_pos = (0, 0)
line_start_pos = (0, 0)
source_img = None
source_msk = None
display_img = None
show_help = False
show_help_timestamp = 0

colourmap = {
    0: (0.0, 0.0, 0.0),
    1: (1.0, 1.0, 1.0),
    2: (1.0, 0.0, 0.0),
    3: (0.0, 1.0, 0.0),
    4: (0.0, 0.0, 1.0),
    5: (1.0, 0.0, 1.0),
    6: (0.0, 1.0, 1.0),
    7: (1.0, 1.0, 0.0)
}

# FIXME: this assumes the key->index mapping is identical
np_colourmap = np.array([x for x in colourmap.values()])


def init():
  """
  create a window
  """
  cv.namedWindow(WINDOW_TITLE)
  cv.setMouseCallback(WINDOW_TITLE, on_mousemove)


def on_mousemove(event, x, y, flags, userparam):
  """
  mousemove + ctrl paints the mask
  """
  global mouse_pos
  global source_img, source_msk, display_img
  global DRAW_MODE

  if event == cv.EVENT_MOUSEMOVE:
    mouse_pos = (x, y)

    if flags & cv.EVENT_FLAG_SHIFTKEY:
      current_label = LABEL_BACKGROUND
    else:
      current_label = CURRENT_LABEL

    if DRAW_MODE == "point":
      if flags & cv.EVENT_FLAG_CTRLKEY:
        cv.circle(source_msk, (x, y), SHAPE_SIZE, current_label, -1)
    elif DRAW_MODE == "line":
      # line drawing is done in the line-mode keypress handler (keydown())
      pass


def help():
  """
  string to display over the editor image when the user presses 'h'
  """
  return """
    MASK EDITOR
    [h] help
    [x] clear mask
    [+] increase shape size
    [-] decrease shape size
    [ ] next image (saves mask)
    [a] point mode (default)
    [s] line mode
    [f] flood fill
    CTRL paint mask
    SHFT unpaint mask
    ESC exit
    1-10 label value
"""


def on_keydown(key):
  """
  keydown functions and mappings
  """
  global source_img, source_msk

  def next_image():
    return False

  def increase_shape_size():
    global SHAPE_SIZE
    SHAPE_SIZE = min(64, SHAPE_SIZE+SHAPE_SIZE_INC)
    return True

  def decrease_shape_size():
    global SHAPE_SIZE
    SHAPE_SIZE = max(1, SHAPE_SIZE-SHAPE_SIZE_INC)
    return True

  def clear_mask():
    global source_msk
    source_msk *= 0
    return True

  def display_help():
    global show_help, show_help_timestamp
    show_help_timestamp = datetime.now()
    show_help = True
    return True

  def stop_editing():
    raise StopIteration

  def set_current_label(value):
    global CURRENT_LABEL
    CURRENT_LABEL = value

  def set_mode_point():
    """
    default point drawing mode
    press CTRL on mousemove to draw
    """
    global DRAW_MODE
    DRAW_MODE="point"

  def set_mode_line():
    """
    start drawing in line mode
    if already in line mode, commit a line to the mask and start anew
    """
    global DRAW_MODE, CURRENT_LABEL, SHAPE_SIZE
    global mouse_pos, line_start_pos

    if DRAW_MODE=="line":
      # draw the line on the mask
      cv.line(source_msk, line_start_pos, mouse_pos, CURRENT_LABEL, thickness=SHAPE_SIZE)

    line_start_pos = mouse_pos
    DRAW_MODE="line"

  def flood_fill():
    """
    flood fill a region in the mask
    FIXME: we really need undo for this!
    """
    global CURRENT_LABEL
    global mouse_pos

    im_mask = (source_msk==CURRENT_LABEL).astype(np.uint8)
    cv.floodFill(im_mask, None, mouse_pos, CURRENT_LABEL)
    source_msk[im_mask!=0] = CURRENT_LABEL

  # function map
  fns = {
    ord(' '): next_image,
    ord('+'): increase_shape_size,
    ord('-'): decrease_shape_size,
    ord('x'): clear_mask,
    ord('h'): display_help,
    27: stop_editing,
    ord('0'): lambda: set_current_label(0),
    ord('1'): lambda: set_current_label(1),
    ord('2'): lambda: set_current_label(2),
    ord('3'): lambda: set_current_label(3),
    ord('4'): lambda: set_current_label(4),
    ord('5'): lambda: set_current_label(5),
    ord('6'): lambda: set_current_label(6),
    ord('7'): lambda: set_current_label(7),
    ord('s'): set_mode_line,
    ord('a'): set_mode_point,
    ord('f'): flood_fill
  }

  try:
    return fns[key]()
  except KeyError:
    # FIXME: value 255 is not handled, what is 255? should we do a noop?
    #logger.warning("don't handle '%i'" % key)
    pass

def on_draw(output_text):
  """
  redraw the image
  """
  from random import shuffle

  global mouse_pos, display_img
  global SHAPE_SIZE #, SHAPE_COLOUR
  global show_help, show_help_timestamp
  global CURRENT_LABEL
  global DRAW_MODE, line_start_pos

  # draw the outline of the shape onto the display imge
  # display_img = blend_in_channel(source_img, source_msk)

  SHAPE_COLOUR = colourmap[CURRENT_LABEL]

  label_img = apply_colourmap(source_msk, np_colourmap)
  display_img = alpha_blend(source_img, label_img, alpha=BLEND_ALPHA)

  # draw things on the display image to indicate whats happening
  if DRAW_MODE == "point":
    cv.circle(display_img, mouse_pos, SHAPE_SIZE, SHAPE_COLOUR, 1)
  elif DRAW_MODE == "line":
    cv.line(display_img, line_start_pos, mouse_pos, SHAPE_COLOUR, thickness=SHAPE_SIZE)
  else:
    logger.error("unknown draw mode: '%s'" % DRAW_MODE)

  if show_help is True:
    if (datetime.now() - show_help_timestamp).seconds > 5:
      show_help = False

    output_lines = help().split("\n")
  elif callable(output_text):
    output_lines = output_text().split("\n")
  elif isinstance(output_text, str):
    # draw normal text
    output_lines = output_text.split("\n")

  # draw text
  font = cv.FONT_HERSHEY_SIMPLEX
  font_scale = 0.4
  font_thickness = 1
  font_colour = SHAPE_COLOUR
  bg_colour = colourmap[LABEL_BACKGROUND if CURRENT_LABEL != LABEL_BACKGROUND else (LABEL_BACKGROUND+1)%len(colourmap)]
  text_vscale = font_scale/0.25 # fudge factor for spacing lines vertically, scale.25 and spacing 1 worked ok)

  for i, line in enumerate(output_lines):
    x = 10
    y = int((i+1)*(10*text_vscale))

    cv.putText(display_img,
        line,
        (x, y),
        font,
        font_scale,
        (0,0,0),
        font_thickness+4
    )

    cv.putText(display_img,
        line,
        (x, y),
        font,
        font_scale,
        font_colour,
        font_thickness
    )


def read_image(filename):
  """
  read an image from disk
  """
  return cv.imread(filename).astype(float)/255.0


def read_mask(filename):
  """
  read a label mask from disk

  all the operations are performed on integer data
  but the mask is returned as a float array because
  various opencv operations require a float array
  input or fail with `Expected Ptr<cv::UMat> for argument 'img'`
  """
  # single label images
  # source_msk = (cv.imread(mask_filename) > 0).astype(np.float)
  # display_img = blend_in_channel(source_img, source_msk)

  # multi-label images
  try:
    source_msk = cv.imread(filename, cv.IMREAD_ANYCOLOR)
  except FileNotFoundError as e:
    logger.warning("'%s' not found, creating empty" % filename)
    source_msk = np.zeros(source_img.shape[:2], dtype=np.uint8)
    logger.debug("source_msk.shape: '%s'" % str(source_msk.shape))

  # if the image is multichannel, take only the first channel
  if len(source_msk.shape) > 2:
    logger.warning("'%s'.shape = %s, reducing to first channel" % (basename(filename), str(source_msk.shape)))
    source_msk = source_msk.mean(axis=-1).astype(int)

  source_msk = source_msk[..., np.newaxis]

  # mask label values
  labels = np.unique(source_msk)
  logger.info("'%s':%s:%s %i labels" % (basename(filename), str(source_msk.shape), str(source_msk.dtype), len(labels)))

  if any([label > max(colourmap.keys()) for label in labels]):
    logger.warning("label values > colourmap range [%i, %i] are mapped to %i" % (
        min(colourmap.keys()), max(colourmap.keys()), 1))

    for label in labels:
      if label > max(colourmap.keys()):
        source_msk[source_msk==label] = 1

  labels = np.unique(source_msk)
  logger.info("'%s':%s:%s labels: %s" % (basename(filename), str(source_msk.shape), str(source_msk.dtype), labels))

  return source_msk.astype(float)


if __name__ == "__main__":
  import sys
  import argparse
  from glob import glob
  from os.path import exists

  log_format = "[%(asctime)s] - %(name)s:%(lineno)d - %(levelname)s - %(message)s"

  root = logging.getLogger()
  root.setLevel(logging.DEBUG)
  root.setLevel(os.getenv("LOG_LEVEL", "DEBUG"))

  ch = logging.StreamHandler(sys.stdout)
  formatter = logging.Formatter(log_format)
  ch.setFormatter(formatter)
  root.addHandler(ch)

  parser = argparse.ArgumentParser(description="Edit image masks. Images and masks are assumed to be aligned lists.")
  parser.add_argument("-i", "--images", required=True, nargs="+", help="list or filename-pattern for images")
  parser.add_argument("-m", "--masks", required=True, nargs="+", help="list or filename-pattern for masks")
  parser.add_argument("-n", "--dry-run", action="store_true", dest="dryrun")
  parser.add_argument("-s", "--start", type=int, default=0, help="index of the first image to start labelling")
  args = parser.parse_args()

  images = sorted(glob(*args.images))
  masks = sorted(glob(*args.masks))

  logger.info("found %i/%i images/masks" % (len(images), len(masks)))
  assert len(images) > 0

  # get the intersection of the images and masks so we only get images with masks and vv
  images = [f for f in images if basename(f) in [basename(m) for m in masks]]
  masks  = [m for m in masks  if basename(m) in [basename(f) for f in images]]
  logger.info("found %i/%i overlapping images/masks" % (len(images), len(masks)))
  assert len(images) > 0
  assert len(images) == len(masks)

  if args.dryrun:
    for image_filename, mask_filename in zip(images[args.start:], masks[args.start:]):
      logger.info("'%s', '%s'" % (image_filename, mask_filename))
      if exists(image_filename) is False:
        logger.error("'%s' not found" % image_filename)
      if exists(mask_filename) is False:
        logger.error("'%s' not found" % mask_filename)
  else:
    init()

    processing_data = {}

    # edit the images
    try:
      for idx, (image_filename, mask_filename) in enumerate(zip(images[args.start:], masks[args.start:])):
        source_img = read_image(image_filename)
        source_msk = read_mask(mask_filename)
        source_msk_hash = hash_np_array(source_msk)
        logger.debug("'%s', mask_hash: '%s'" % (basename(image_filename), str(source_msk_hash)))

        # create the initial display image
        display_img = np.zeros_like(source_img).astype(float)

        T0 = clock()

        while(1):
          on_draw(output_text=lambda: "[%i/%i] [%i]" % (args.start+idx, len(images), CURRENT_LABEL))

          cv.imshow("mask-edit", display_img)
          k = cv.waitKey(1) & 0xFF

          if on_keydown(k) is False:
            break

        T1 = clock()

        # check for updates to the mask
        output_msk_hash = hash_np_array(source_msk)
        logger.debug("'%s', mask_hash: '%s' in %0.2fs" % (basename(image_filename), str(output_msk_hash), (T1-T0)))

        if output_msk_hash != source_msk_hash:
          logger.info("'%s': updating mask with %s" % (basename(image_filename), str(source_msk[:, :, 0].shape)))
          cv.imwrite(mask_filename, source_msk[:, :, 0].astype(int))
          processing_data.update({basename(image_filename): {"time": T1-T0}})

    except StopIteration as e:
      logger.info("stopping")

    # FIXME: write out the processing data for fun
    # logger.info(json.dumps(processing_data, indent=2))

    # do some stats
    times = [stats["time"] for stats in processing_data.values() if "time" in stats]

    total_time = sum(times)
    average_time = total_time / len(times) if len(times) > 0 else 0.0
    logger.info("session duration: %0.2fs, %i images, %0.2fs average per image" % (total_time, len(times), average_time))
