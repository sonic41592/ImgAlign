from __future__ import print_function
import os.path
import os
import sys
import cv2
import glob
import numpy as np
import argparse
from PIL import Image
import math

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--scale", help="Positive integer value.  How many times bigger you want the HR resolution to be from the LR resolution.", required=True)
parser.add_argument("-m", "--mode", required=True, help="Options: 0 or 1.  Mode 0 manipulates the HR images while remaining true to the LR images aside from cropping.  Mode 1 manipulates the LR images and remains true to the HR images aside from cropping.")
parser.add_argument("-c", "--autocrop", action='store_true', default=False, help="Disabled by default.  If enabled, this auto crops black boarders around HR and LR images.")
parser.add_argument("-t", "--threshold", default=15, help="Integer 0-255, default 15.  Luminance threshold for autocropping.  Higher values cause more agressive cropping.")
parser.add_argument("-r", "--rotate", action='store_true', default=False, help="Disabled by default.  If enabled, this allows rotations when aligning images.")
parser.add_argument("-g", "--hr", default='', help="HR File or folder directory.  No need to use if they are in HR folder in current working directory.")
parser.add_argument("-l", "--lr", default='', help="LR File or folder directory.  No need to use if they are in LR folder in current working directory.")
parser.add_argument("-o", "--overlay", action='store_true', default=False, help="Disabled by default.  After saving aligned images, this option will create a separate 50:50 merge of the aligned images in the Overlay folder. Useful for quickly checking through image sets for poorly aligned outputs")

args = vars(parser.parse_args())

scale = float(args["scale"])
mode = int(args["mode"])
autocrop = args["autocrop"]
lumthresh = int(args["threshold"])
rotate = args["rotate"]
HRfolder = args["hr"]
LRfolder = args["lr"]
Overlay = args["overlay"]

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15


def AutoCrop(image):
    """Crops any edges below or equal to threshold

    Crops blank image to 1x1.

    Returns cropped image.

    """
    threshold = lumthresh
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]

    return image

def Find_Affine(im1, im2):
  
  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)

  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)

  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]
  
  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)

  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

  # Find homography
  h, mask = cv2.estimateAffine2D(points1, points2)
  
  return h

def rotatedRectWithMaxArea(w, h, angle):
  """
  Given a rectangle of size wxh that has been rotated by 'angle' (in
  radians), computes the width and height of the largest possible
  axis-aligned rectangle (maximal area) within the rotated rectangle.
  """
  if w <= 0 or h <= 0:
    return 0,0

  width_is_longer = w >= h
  side_long, side_short = (w,h) if width_is_longer else (h,w)

  # since the solutions for angle, -angle and 180-angle are all the same,
  # if suffices to look at the first quadrant and the absolute values of sin,cos:
  sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
  if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
    # half constrained case: two crop corners touch the longer side,
    #   the other two corners are on the mid-line parallel to the longer line
    x = 0.5*side_short
    wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
  else:
    # fully constrained case: crop touches all 4 sides
    cos_2a = cos_a*cos_a - sin_a*sin_a
    wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

  return wr,hr


def Fix_Rotation(Affine, img):
  h, w, _ = img.shape
  sx = math.sqrt(Affine[0,0]**2+Affine[1,0]**2)
  sy = math.sqrt(Affine[0,1]**2+Affine[1,1]**2)
  theta1 = np.arcsin(-Affine[0,1]/sy)
  theta2 = np.arcsin(Affine[1,0]/sx)
  theta = (theta1+theta2)/2
  img = Image.fromarray(img)
  img = img.rotate(-theta*180/3.14159265, resample=Image.BICUBIC, expand=True)
  img = np.asarray(img)
  dw, dh = rotatedRectWithMaxArea(w, h, theta)
  h, w, _ = img.shape
  h = h/2
  w = w/2
  dh = dh/2
  dw = dw/2
  xl = int(math.ceil(w-dw))
  xh = int(math.floor(w+dw))
  yl = int(math.ceil(h-dh))
  yh = int(math.floor(h+dh))
  img = img[yl:yh,xl:xh]
  return img



def maint_lr_res(im1, im2):
  
  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)

  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)

  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]
  
  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)

  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

  # Find homography
  h, mask = cv2.estimateAffine2D(points1, points2)
  
  # Remove rotation 
  sx = math.sqrt(h[0,0]**2+h[1,0]**2)
  sy = math.sqrt(h[0,1]**2+h[1,1]**2)
  h[0,0] = sx
  h[1,1] = sy
  h[0,1] = 0
  h[1,0] = 0
  
  # Map 4 corners of HR to LR coordinates
  height1, width1, channels1 = im1.shape
  height2, width2, channels2 = im2.shape
  
  bhr = np.array([[width1],[height1],[1]])
  chr = np.array([[0],[0],[1]])
  Bclr = h @ bhr
  Cclr = h @ chr
  
  # Check and fix out of bounds
  
  if Bclr[0] > width2:
    Bclr[0] = width2
  if Bclr[1] > height2:
    Bclr[1] = height2
  if Cclr[0] < 0:
    Cclr[0] = 0
  if Cclr[1] < 0:
    Cclr[1] = 0
  
  # Fix imperfect downscaling (only applicable in mode 1)
  if scale < 1:
    yrem = np.mod(int(round(Bclr[1,0]))-int(round(Cclr[1,0])),1/scale)
    xrem = np.mod(int(round(Bclr[0,0]))-int(round(Cclr[0,0])),1/scale)
    Bclr[0,0] = Bclr[0,0] - xrem
    Bclr[1,0] = Bclr[1,0] - yrem
    
  # Crop LR 
  yl = int(round(Cclr[1,0]))
  yh = int(round(Bclr[1,0]))
  xl = int(round(Cclr[0,0]))
  xh = int(round(Bclr[0,0]))
  
  crop_lr = im2[yl:yh, xl:xh]
  
  # Resize and crop HR
  
  Hom = np.zeros((3,3))
  Hom[0,0] = h[0,0]
  Hom[1,1] = h[1,1]
  Hom[0,2] = h[0,2]
  Hom[1,2] = h[1,2]
  Hom[2,2] = 1
  iHom = np.linalg.inv(Hom)
  
  Bhr = np.array([[float(Bclr[0])],[float(Bclr[1])],[1]])
  Chr = np.array([[float(Cclr[0])],[float(Cclr[1])],[1]])
  iBclr = iHom @ Bhr
  iCclr = iHom @ Chr
  
  YL = int(round(float(iCclr[1,0])))
  YH = int(round(float(iBclr[1,0])))
  XL = int(round(float(iCclr[0,0])))
  XH = int(round(float(iBclr[0,0])))
  
  crop_hr = im1[YL:YH, XL:XH]
  
  hgt, wdt, chn = crop_lr.shape
  dim = (round(scale*wdt), round(scale*hgt))
  
  scalehr = cv2.resize(crop_hr, dim, interpolation=cv2.INTER_CUBIC)
  return crop_lr, scalehr



def Do_Work(hrimg, lrimg):
  highres = cv2.imread(hrimg, cv2.IMREAD_COLOR)
  lowres = cv2.imread(lrimg, cv2.IMREAD_COLOR)
  
  if autocrop == True:
    highres = AutoCrop(highres)
    lowres = AutoCrop(lowres)
  
  if mode == 0:
    
    if rotate == True:
      Aff = Find_Affine(highres,lowres)
      highres = Fix_Rotation(Aff, highres)
    
    crop_lr, scalehr = maint_lr_res(highres, lowres)
    cv2.imwrite('Output\\LR\\{:s}.png'.format(base), crop_lr)
    cv2.imwrite('Output\\HR\\{:s}.png'.format(base), scalehr)
  
  if mode == 1:
  
    if rotate == True:
      Aff = Find_Affine(lowres,highres)
      lowres = Fix_Rotation(Aff, lowres)
    
    crop_lr, scalehr = maint_lr_res(lowres,highres)
    cv2.imwrite('Output\\HR\\{:s}.png'.format(base), crop_lr)
    cv2.imwrite('Output\\LR\\{:s}.png'.format(base), scalehr)
  
  # Create overlays
  if Overlay == True:
    hgt, wdt, chn = crop_lr.shape
    h_hr, w_hr, c_hr = scalehr.shape
    dim_overlay = (max(wdt, w_hr), max(hgt, h_hr))
    scalelr = cv2.resize(crop_lr,dim_overlay, interpolation=cv2.INTER_CUBIC)
    scaleHR = cv2.resize(scalehr,dim_overlay, interpolation=cv2.INTER_CUBIC)
    overlay = cv2.addWeighted(scaleHR,0.5,scalelr,0.5,0)
    cv2.imwrite('Output\\Overlay\\{:s}.png'.format(base), overlay)
  
  
  
# Make Output folders
if not os.path.exists('Output'):
  os.mkdir('Output')
if not os.path.exists('Output\\LR'):
  os.mkdir('Output\\LR')
if not os.path.exists('Output\\HR'):
  os.mkdir('Output\\HR')
if Overlay == True:
  if not os.path.exists('Output\\Overlay'):
    os.mkdir('Output\\Overlay')

# Invert scale for mode 1
if mode == 1:
  scale = 1/scale

if os.path.isfile(HRfolder) == True:
  base = os.path.splitext(os.path.basename(HRfolder))[0]
  hrim = HRfolder
  lrim = LRfolder
  Do_Work(hrim, lrim)


else:
    # Batch processing for all images in HR folder
  if len(HRfolder) == 0:
    HRfolder = 'HR'
    LRfolder = 'LR\\'

  for path in glob.glob(HRfolder+'/*'):
    base = os.path.splitext(os.path.basename(path))[0]
    extention = os.path.splitext(os.path.basename(path))[1]
    hrim = path
    lrim = LRfolder+'/'+base+extention
    print('{:s}'.format(base)+extention)
    Do_Work(hrim, lrim)
