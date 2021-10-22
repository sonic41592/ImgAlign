from __future__ import print_function
import os.path
import sys
import cv2
import glob
import numpy as np

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

mode = int(sys.argv[1])
hrimg = sys.argv[2]
lrimg = sys.argv[3]
scale = float(sys.argv[4])

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
  
  scalehr = cv2.resize(crop_hr, dim)
  
  # Write output images
  cv2.imwrite('croppedlr.png', crop_lr)
  cv2.imwrite('scaledhr.png', scalehr)
  
  
  
def maint_hr_res(im1, im2):

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
  
  # Normalize scale
  h = h/scale
  
  # Remove rotations
  h[0,1] = 0
  h[1,0] = 0
  
  # Resize and apply transformation to LR
  height1, width1, channels1 = im1.shape
  height2, width2, channels2 = im2.shape
  
  width = round(width2/scale)
  height = round(height2/scale)
  
  im1Reg = cv2.warpAffine(im1, h, (width, height))
  
  # Write output image
  cv2.imwrite('scaledlr.png',im1Reg)

  # Read HR image
highres = cv2.imread(hrimg, cv2.IMREAD_COLOR)

  # Read LR image
lowres = cv2.imread(lrimg, cv2.IMREAD_COLOR)

if __name__ == '__main__':

  if mode == 0:
    maint_lr_res(highres, lowres)
  
  if mode == 1:
    maint_hr_res(lowres,highres)