import sys
import logging

import numpy as np
import cv2 as cv

from os.path import basename
from datetime import datetime

logger = logging.getLogger(__name__)

# disable loggers
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


# global vars
# FIXME: take a conf file
WINDOW_TITLE = "mask-edit"
MASK_COLOUR = (1.0, 0.0, 0.0)
UNMASK_COLOUR = (0.0, 0.0, 0.0)
SHAPE_COLOUR = (1.0, 1.0, 1.0)
SHAPE_SIZE = 6
SHAPE_SIZE_INC = 1
BLEND_ALPHA = 0.75
MASK_THRESHOLD = 0.5

# initial values of global vars
mouse_pos = (0, 0)
source_img = None
source_msk = None
display_img = None
show_help = False
show_help_timestamp = 0

7
def init():
  """
  create a window
  """
  cv.namedWindow(WINDOW_TITLE)
  cv.setMouseCallback(WINDOW_TITLE, on_mousemove)


def blend_images(img_a, img_b, alpha):
  """
  blend two images using opacity
  """
  return img_a * (1.0-alpha) + img_b * (alpha)


def blend_in_channel(img_a, img_b):
  """
  creates a three channel image with:
    channel 1: image_b[0],
    channel 2: image_a[1]
    channel 3: image_a[2]
  opencv is bgr so this makes the blue channel of the colour image equal to the masks contents
  """
  return np.dstack([img_b[...,0], img_a[...,1], img_a[...,2]])


def on_mousemove(event, x, y, flags, userparam):
  """
  mousemove + ctrl paints the mask
  """
  global mouse_pos
  global source_img, source_msk, display_img

  if event == cv.EVENT_MOUSEMOVE:
    mouse_pos = (x, y)

    if flags & cv.EVENT_FLAG_SHIFTKEY:
      colour = UNMASK_COLOUR
    else:
      colour = MASK_COLOUR

    if flags & cv.EVENT_FLAG_CTRLKEY:
      cv.circle(source_msk, (x, y), SHAPE_SIZE, colour, -1)


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
    CTRL paint mask
    SHFT unpaint mask
    ESC exit
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

  # function map
  fns = {
    ord(' '): next_image,
    ord('+'): increase_shape_size,
    ord('-'): decrease_shape_size,
    ord('x'): clear_mask,
    ord('h'): display_help,
    27: stop_editing
  }

  try:
    return fns[key]()
  except KeyError:
    # FIXME: value 255 is not handled, what is 255? should we do a noop?
    #logger.warning("don't handle '%i'" % key)
    pass

def on_draw():
  """
  redraw the image
  """
  global mouse_pos, display_img
  global SHAPE_SIZE, SHAPE_COLOUR
  global show_help, show_help_timestamp

  # draw the outline of the shape onto the display imge
  display_img = blend_in_channel(source_img, source_msk)
  cv.circle(display_img, mouse_pos, SHAPE_SIZE, SHAPE_COLOUR, 1)

  if show_help is True:
    if (datetime.now() - show_help_timestamp).seconds > 5:
      show_help = False

    lines = help().split("\n")
    scale = 0.4
    text_vscale = scale/0.25 # fudge factor for spacing lines vertically, scale.25 and spacing 1 worked ok)
    for i, line in enumerate(lines):
      cv.putText(display_img,
                 line,
                 (10, int(i*(10*text_vscale))),
                 cv.FONT_HERSHEY_SIMPLEX, scale, (255,0,255), 1)


if __name__ == "__main__":
  import sys
  import argparse
  from glob import glob
  from os.path import exists

  # setup logging
  root = logging.getLogger()
  root.setLevel(logging.DEBUG)

  ch = logging.StreamHandler(sys.stdout)
  ch.setLevel(logging.DEBUG)
  formatter = logging.Formatter("%(name)s:%(levelname)s:%(message)s")
  ch.setFormatter(formatter)
  root.addHandler(ch)

  #global source_img, source_msk, display_img

  parser = argparse.ArgumentParser(description="Edit image masks. Images and masks are assumed to be aligned lists.")
  parser.add_argument("-i", "--images", nargs="+", help="list or filename-pattern for images")
  parser.add_argument("-m", "--masks", nargs="+", help="list or filename-pattern for masks")
  parser.add_argument("--dry-run", action="store_true", dest="dryrun")
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
    for image_filename, mask_filename in zip(images, masks):
      logger.info("'%s', '%s'" % (image_filename, mask_filename))
      if exists(image_filename) is False:
        logger.error("'%s' not found" % image_filename)
      if exists(mask_filename) is False:
        logger.error("'%s' not found" % mask_filename)
  else:
    init()

    # edit the images
    for image_filename, mask_filename in zip(images, masks):
      # FIXME: take an MD5 of the mask contents and only save if the hash differs so we don't wreck timestamps
      # FIXME: check for multi-class labels
      source_img = cv.imread(image_filename).astype(np.float)/255.0
#      source_msk = cv.imread(mask_filename).astype(np.float)/255.0
      source_msk = (cv.imread(mask_filename) > 0).astype(np.float)
      display_img = blend_in_channel(source_img, source_msk)

      while(1):
        on_draw()
        cv.imshow("mask-edit", display_img)
        k = cv.waitKey(1) & 0xFF

        if on_keydown(k) is False:
          logger.info("writing image to '%s' [%s]" % (mask_filename, str(np.ptp(source_msk))))
          cv.imwrite(mask_filename, (source_msk[...,0] * 255.0).astype(np.uint8))
          break
