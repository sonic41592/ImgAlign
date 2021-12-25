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
parser.add_argument("-f", "--full", action='store_true', default=False, help="Disabled by default.  If enabled, this allows full homography mapping of the image, correcting rotations, translations, and warping.")


args = vars(parser.parse_args())

scale = float(args["scale"])
mode = int(args["mode"])
autocrop = args["autocrop"]
lumthresh = int(args["threshold"])
rotate = args["rotate"]
HRfolder = args["hr"]
LRfolder = args["lr"]
Overlay = args["overlay"]
Homography = args["full"]

if Homography == True:
  rotate = False

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

def Find_Homography(im1, im2):
  
  im1y, im1x, _ = im1.shape
  im2y, im2x, _ = im2.shape
  im0 = im1[:]
  im00 = im2[:]
  
  # Double lower resolution dummy image if the scale is too big for cv2 to match reliably
  if scale == 4 and mode == 0:
    im00 = cv2.resize(im2,(round(2*im2x),round(2*im2y)),interpolation=cv2.INTER_CUBIC)
  if scale == 4 and mode == 1:
    im0 = cv2.resize(im1,(round(2*im1x),round(2*im1y)),interpolation=cv2.INTER_CUBIC)
  
  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im00, cv2.COLOR_BGR2GRAY)
  
  # Detect SIFT features and compute descriptors.
  orb = cv2.SIFT_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
  
  # SIFT feature matching
  bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
  matches = bf.knnMatch(descriptors1,descriptors2,k=2)
  
  good = []
  for m,n in matches:
    if m.distance < 0.7*n.distance:
      good.append(m)
  
  if len(good) > 5:
    points1 = np.float32([ keypoints1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    points2 = np.float32([ keypoints2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
  
  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
  # Fix upscaled homography
  if scale == 4 and mode == 0:
    h = np.array([[0.5,0,0],[0,0.5,0],[0,0,1]]) @ h
  if scale == 4 and mode == 1:
    h = np.array([[2,0,0],[0,2,0],[0,0,1]]) @ h
  
  # print(h)
  
  # Get original image dimensions 
  height, width, _ = im1.shape

  # Find corners of transformed image
  a = np.array([[0,0],[0,height],[width,height],[width,0]],dtype='float32')
  a = np.array([a])
  A = cv2.perspectiveTransform(a,h)

  # Get distances between points to scale homograpy matrix so that the most reduced edge matches resolution of original image
  sab = (np.sqrt((A[0,0,0]-A[0,1,0])**2+(A[0,0,1]-A[0,1,1])**2))/height
  sbc = (np.sqrt((A[0,1,0]-A[0,2,0])**2+(A[0,1,1]-A[0,2,1])**2))/width
  scd = (np.sqrt((A[0,2,0]-A[0,3,0])**2+(A[0,2,1]-A[0,3,1])**2))/height
  sda = (np.sqrt((A[0,3,0]-A[0,0,0])**2+(A[0,3,1]-A[0,0,1])**2))/width
  scaleR = 1/min(sab,sbc,scd,sda)
  smat = np.array([[scaleR,0,0],[0,scaleR,0],[0,0,1]])
  h = smat @ h

  # Find new corners and translation components of proper scale transformation 
  A = cv2.perspectiveTransform(a,h)
  [xmin, ymin] = np.array([min(A[0,:,0])-1, min(A[0,:,1])-1],dtype='int32')
  [xmax, ymax] = np.array([max(A[0,:,0])+1, max(A[0,:,1])+1],dtype='int32')

  # Translate homography to x and y axis
  T = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]])
  h = T @ h

  # Generate, warp, and find new dimensions of a white image with same dimensions as original image
  white = np.zeros([height,width,3],dtype=np.uint8)
  white.fill(255)
  warpwhite = cv2.warpPerspective(white,h,(xmax-xmin,ymax-ymin),flags=0)
  wh, ww, _ = warpwhite.shape
  a1 = [0,0] 
  b1 = [0,wh]
  c1 = [ww,wh]
  d1 = [ww,0]
  
  # Shrink corners until contained within white boundaries
  while max(warpwhite[a1[1],a1[0]]) == 0 or max(warpwhite[b1[1],b1[0]]) == 0 or max(warpwhite[c1[1],c1[0]]) == 0 or max(warpwhite[d1[1],d1[0]]) == 0:
    a1[0] = a1[0] + 1
    a1[1] = a1[1] + 1
    b1[0] = b1[0] + 1
    b1[1] = b1[1] - 1
    c1[0] = c1[0] - 1
    c1[1] = c1[1] - 1
    d1[0] = d1[0] - 1
    d1[1] = d1[1] + 1

  # Boundary testing variables
  sA = a1[:]
  sB = b1[:]
  sC = c1[:]
  sD = d1[:]
  tA = a1[:]
  tB = b1[:]
  tC = c1[:]
  tD = d1[:]

  # Individually increase edge boundaries to maximize area
  while sA[1] > 0 and (max(warpwhite[sA[1]-1,sA[0]]) > 0 and max(warpwhite[sD[1]-1,sD[0]]) > 0):
    sA[1] = sA[1] - 1
    sD[1] = sD[1] - 1
  while sB[1] < wh and (max(warpwhite[sB[1]+1,sB[0]]) > 0 and max(warpwhite[sC[1]+1,sC[0]]) > 0):
    sB[1] = sB[1] + 1
    sC[1] = sC[1] + 1
  while sA[0] > 0 and (max(warpwhite[sA[1],sA[0]-1]) > 0 and max(warpwhite[sB[1],sB[0]-1]) > 0):
    sA[0] = sA[0] - 1
    sB[0] = sB[0] - 1
  while sD[0] < ww and (max(warpwhite[sD[1],sD[0]+1]) > 0 and max(warpwhite[sC[1],sC[0]+1]) > 0):
    sD[0] = sD[0] + 1
    sC[0] = sC[0] + 1

  while tA[0] > 0 and (max(warpwhite[tA[1],tA[0]-1]) > 0 and max(warpwhite[tB[1],tB[0]-1]) > 0):
    tA[0] = tA[0] - 1
    tB[0] = tB[0] - 1
  while tD[0] < ww and (max(warpwhite[tD[1],tD[0]+1]) > 0 and max(warpwhite[tC[1],tC[0]+1]) > 0):
    tD[0] = tD[0] + 1
    tC[0] = tC[0] + 1
  while tA[1] > 0 and (max(warpwhite[tA[1]-1,tA[0]]) > 0 and max(warpwhite[tD[1]-1,tD[0]]) > 0):
    tA[1] = tA[1] - 1
    tD[1] = tD[1] - 1
  while tB[1] < wh and (max(warpwhite[tB[1]+1,tB[0]]) > 0 and max(warpwhite[tC[1]+1,tC[0]]) > 0):
    tB[1] = tB[1] + 1
    tC[1] = tC[1] + 1

  # Choose biggest bound image
  ssize = (sC[0]-sA[0])*(sC[1]-sA[1])
  tsize = (tC[0]-tA[0])*(tC[1]-tA[1])

  if tsize >= ssize:
    a1 = tA[:]
    b1 = tB[:]
    c1 = tC[:]
    d1 = tD[:]
  else:
    a1 = sA[:]
    b1 = sB[:]
    c1 = sC[:]
    d1 = sD[:]

  # Warp original image
  output = cv2.warpPerspective(im1,h,(ww,wh),flags=2)
  Im1 = output[a1[1]:c1[1],a1[0]:c1[0]]
  
  return Im1

def Find_Affine(im1, im2):
  
  im1y, im1x, _ = im1.shape
  im2y, im2x, _ = im2.shape
  im0 = im1[:]
  im00 = im2[:]
  
  # Double lower resolution dummy image if the scale is too big for cv2 to match reliably
  if scale == 4 and mode == 0:
    im00 = cv2.resize(im2,(round(2*im2x),round(2*im2y)),interpolation=cv2.INTER_CUBIC)
  if scale == 4 and mode == 1:
    im0 = cv2.resize(im1,(round(2*im1x),round(2*im1y)),interpolation=cv2.INTER_CUBIC)
  
  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im00, cv2.COLOR_BGR2GRAY)
  
  # Detect SIFT features and compute descriptors.
  orb = cv2.SIFT_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
  
  # SIFT feature matching
  bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
  matches = bf.knnMatch(descriptors1,descriptors2,k=2)
  
  good = []
  for m,n in matches:
    if m.distance < 0.7*n.distance:
      good.append(m)
  
  if len(good) > 5:
    points1 = np.float32([ keypoints1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    points2 = np.float32([ keypoints2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

  # Find homography
  h, mask = cv2.estimateAffine2D(points1, points2, cv2.RANSAC)
  # Fix upscaled homography
  if scale == 4 and mode == 0:
    h = np.array([[0.5,0],[0,0.5]]) @ h
  if scale == 4 and mode == 1:
    h = np.array([[2,0],[0,2]]) @ h
  
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
  
  im1y, im1x, _ = im1.shape
  im2y, im2x, _ = im2.shape
  im0 = im1[:]
  im00 = im2[:]
  
  # Double lower resolution dummy image if the scale is too big for cv2 to match reliably
  if scale == 4 and mode == 0:
    im00 = cv2.resize(im2,(round(2*im2x),round(2*im2y)),interpolation=cv2.INTER_CUBIC)
  if scale == 4 and mode == 1:
    im0 = cv2.resize(im1,(round(2*im1x),round(2*im1y)),interpolation=cv2.INTER_CUBIC)
  
  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im00, cv2.COLOR_BGR2GRAY)
  
  # Detect SIFT features and compute descriptors.
  orb = cv2.SIFT_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
  
  # SIFT feature matching
  bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
  matches = bf.knnMatch(descriptors1,descriptors2,k=2)
  
  good = []
  for m,n in matches:
    if m.distance < 0.7*n.distance:
      good.append(m)
  
  if len(good) > 5:
    points1 = np.float32([ keypoints1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    points2 = np.float32([ keypoints2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
  
  # Find homography
  h, mask = cv2.estimateAffine2D(points1, points2, cv2.RANSAC)
  # Fix upscaled homography
  if scale == 4 and mode == 0:
    h = np.array([[0.5,0],[0,0.5]]) @ h
  if scale == 4 and mode == 1:
    h = np.array([[2,0],[0,2]]) @ h  
  
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
  
  
  if mode == 0:
  
    if autocrop == True:
      highres = AutoCrop(highres)
      lowres = AutoCrop(lowres)
    
    if rotate == True:
      Aff = Find_Affine(highres,lowres)
      highres = Fix_Rotation(Aff, highres)
    
    if Homography == True:
      highres = Find_Homography(highres,lowres)
    
    crop_lr, scalehr = maint_lr_res(highres, lowres)
    cv2.imwrite('Output\\LR\\{:s}.png'.format(base), crop_lr)
    cv2.imwrite('Output\\HR\\{:s}.png'.format(base), scalehr)
  
  if mode == 1:
  
    if autocrop == True:
      highres = AutoCrop(highres)
      lowres = AutoCrop(lowres)
  
    if rotate == True:
      Aff = Find_Affine(lowres,highres)
      lowres = Fix_Rotation(Aff, lowres)
    
    if Homography == True:
      lowres = Find_Homography(lowres,highres)
    
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
    try:
      Do_Work(hrim, lrim)
    except KeyboardInterrupt:
      break
    except:
      with open('Output\Failed.txt', 'a+') as f:
        f.write('{:s}'.format(base)+extention+'\n')
        f.close()
      print('Match failed for ','{:s}'.format(base)+extention)
