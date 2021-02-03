import numpy as np

from hashlib import md5

def hash_np_array(a):
  return md5(a.tobytes()).hexdigest()
