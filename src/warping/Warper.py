
import os
import numpy as np
import cv2
from warp_support import locator, aligner, warper, blender, plotter, videoer


def verify_args(args):
  if args['--images'] is None:
    valid = os.path.isfile(args['--src']) & os.path.isfile(args['--dest'])
    if not valid:
      print('--src=%s or --dest=%s file does not exist. Double check the supplied paths' % (
        args['--src'], args['--dest']))
      exit(1)
  else:
    valid = os.path.isdir(args['--images'])
    if not valid:
      print('--images=%s is not a valid directory' % args['--images'])
      exit(1)

def load_image_points(path, size):
  img = cv2.imread(path)
  if "MyFace" in path:
      img = locator.align(img,size)
  points = locator.face_points(img,size)

  if len(points) == 0:
    print('No face in %s' % path)
    return None, None
  else:
    return aligner.resize_align(img, points, size)

def load_valid_image_points(imgpaths, size):
  for path in imgpaths:
    img, points = load_image_points(path, size)
    if img is not None:
      print(path)
      yield (img, points)