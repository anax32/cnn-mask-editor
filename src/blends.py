"""
functions for blending two images
"""

def alpha_blend(img_a, img_b, alpha=0.5):
  """
  blend two images using opacity
  """
  return img_a * (1.0-alpha) + img_b * (alpha)


def channel_blend(img_a, img_b):
  """
  creates a three channel image with:
    channel 1: image_b[0],
    channel 2: image_a[1]
    channel 3: image_a[2]
  opencv is bgr so this makes the blue channel of the colour image equal to the masks contents
  """
  return np.dstack([img_b[...,0], img_a[...,1], img_a[...,2]])


def apply_colourmap(index_img, colourmap):
  """
  applys a colourmap to the index img

  index img is cast to int before apply
  """
  return colourmap[index_img[:, :, 0].astype(int)]
