from mpl_interactions import zoom_factory, panhandler
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import cv2
import glob
import argparse
import math
from scipy import ndimage
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.linear_model import RANSACRegressor
from python_color_transfer.color_transfer import ColorTransfer
PT = ColorTransfer()


parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-s", "--scale", help="Positive integer value.  How many times bigger you want the HR resolution to be from the LR\nresolution.", required=True)
parser.add_argument("-m", "--mode", required=True, help="Options: 0 or 1.  Mode 0 manipulates the HR images while remaining true to the LR images aside\nfrom cropping.  Mode 1 manipulates the LR images and remains true to the HR images aside from\ncropping.")
parser.add_argument("-c", "--autocrop", action='store_true', default=False, help="Disabled by default.  If enabled, this auto crops black boarders around HR and LR images.")
parser.add_argument("-t", "--threshold", default=50, help="Integer 0-255, default 50.  Luminance threshold for autocropping.  Higher values cause more\nagressive cropping.")
parser.add_argument("-n", "--threads", default=1, help="Default 1.  Number of threads to use for automatic matching.  Large images require a lot of RAM,\nso start small to test first.")
parser.add_argument("-r", "--rotate", action='store_true', default=False, help="Disabled by default.  If enabled, this allows rotations when aligning images.")
parser.add_argument("-g", "--hr", default='', help="HR File or folder directory.  No need to use if they are in HR folder in current working\ndirectory.")
parser.add_argument("-l", "--lr", default='', help="LR File or folder directory.  No need to use if they are in LR folder in current working\ndirectory.")
parser.add_argument("-o", "--overlay", action='store_false', default=True, help="Enabled by default.  After saving aligned images, this option will create a separate 50:50\nmerge of the aligned images in the Overlay folder. Useful for quickly checking through image\nsets for poorly aligned outputs")
parser.add_argument("-i", "--color", default=0, help="Default 0.  Choose which color to use for color correction.  -1 uses LR color and 1 uses HR color")
parser.add_argument("-f", "--full", action='store_true', default=False, help="Disabled by default.  If enabled, this allows full homography mapping of the image, correcting\nrotations, translations, and warping.")
parser.add_argument("-e", "--score", action='store_true', default=False, help="Disabled by default.  Calculate an alignment score for each processed pair of images")
parser.add_argument("-w", "--warp", action='store_true', default=False, help="Disabled by default.  Match images using Thin Plate Splines, allowing full image warping")
parser.add_argument("-a", "--semiauto", action='store_true', default=False, help="Disabled by default.  Semiautomatic mode.  Automatically find matching points, but load into a\nviewer window to manually delete or add more.")
parser.add_argument("-u", "--manual", action='store_true', default=False, help="Disabled by default.  Manual mode.  If enabled, this opens windows for working pairs of images\nto be aligned.  Double click pairs of matching points on each image in sequence, and close the\nwindows when finished.\n\nManual Keys: \nDouble click left: Select point.\nClick and Drag left: Pan image.\nClick Scroll Wheel: Delete matching pairs of points.\nScroll Wheel: Zoom in and out.\nDouble Click right: Reset image view.\nu: Undo last point selection.\nw: Close all windows to progress.\np: Preview alignment.  Overlays images using current alignment points.")

args = vars(parser.parse_args())

scale = float(args["scale"])
mode = int(args["mode"])
autocrop = args["autocrop"]
lumthresh = int(args["threshold"])
threads = int(args["threads"])
rotate = args["rotate"]
HRfolder = args["hr"]
LRfolder = args["lr"]
Overlay = args["overlay"]
Homography = args["full"]
Manual = args["manual"]
score = args["score"]
semiauto = args["semiauto"]
warp = args["warp"]
color_correction = int(args["color"])

# Changing conflicting or priority setting
if warp:
  Homography = False
  if not Manual:
    semiauto = True
  
if Manual or semiauto:
  threads = 1

MAX_FEATURES = 500

def AutoCrop(image):

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

# Remove outliers from points
def ransac(pnt1, pnt2):
  pnt1x, pnt1y = pnt1.reshape(-1,2)[:,0].reshape(-1,1), pnt1.reshape(-1,2)[:,1].reshape(-1,1)
  pnt2x, pnt2y = pnt2.reshape(-1,2)[:,0].reshape(-1,1), pnt2.reshape(-1,2)[:,1].reshape(-1,1)
  ransacx = RANSACRegressor().fit(pnt1x, pnt2x)
  ransacy = RANSACRegressor().fit(pnt1y, pnt2y)
  inlier_maskx = ransacx.inlier_mask_
  inlier_masky = ransacy.inlier_mask_
  inliers = inlier_maskx*inlier_masky
  pnt1, pnt2 = pnt1[inliers], pnt2[inliers]
  return pnt1, pnt2

# Create and apply Thin Plate Spline transform to an image
def WarpImage_TPS(source, target, img, interp):
  tps = cv2.createThinPlateSplineShapeTransformer()

  source=source.reshape(-1,max(source.shape[0],source.shape[1]),2)
  target=target.reshape(-1,max(target.shape[0],target.shape[1]),2)  

  matches=list()
  for i in range(0,len(source[0])):

    matches.append(cv2.DMatch(i,i,0))

  tps.estimateTransformation(target, source, matches)
  if interp == 0:
    new_img = tps.warpImage(img, flags = cv2.INTER_NEAREST)
  else:
    new_img = tps.warpImage(img, flags = cv2.INTER_CUBIC)

  return new_img

# Make and manipulate plots for manual point selection 
def manual_points(img1, img2, pointsA = None, pointsB = None):
  global pnts1, pnts2, markers1, markers2, active
  
  pnts1 = np.array([])
  pnts2 = np.array([])
  pnts1.shape = (0,2)
  pnts2.shape = (0,2)
  markers1 = []
  markers2 = []
  active = []
  characters=['o', 'v','^','<','>','1','2','3','4','s','p','P','*','+','x','X','D','d']
  if pointsA is not None:
    pointsA, pointsB = pointsA.reshape(-1,2), pointsB.reshape(-1,2)
    for row in pointsA:
      pnts1 = np.concatenate((pnts1,row.reshape(1,2)))
    for row in pointsB:
      pnts2 = np.concatenate((pnts2,row.reshape(1,2)))
  
  # Matplotlib UI functions
  def tnuoc(mnum1, acnum):
    global active
    numA = 0
    count = 0
    for idx in range(len(active)-1,-1,-1):
      if active[idx] == acnum:
        count += 1
      if count == mnum1:
        ele = idx
        break
    return ele
    
  def onclick(event, graph):
    
    if event.dblclick and str(event.button) == 'MouseButton.LEFT':
      global pnts1, pnts2, markers1, markers2, active
      
      ix, iy = event.xdata, event.ydata
      if ix != None and iy != None:
        if graph == 1:
          active.append(1)
          pnts1 = np.concatenate((pnts1,np.array([[ix,iy]])))
          marker = plt.plot(event.xdata, event.ydata, characters[len(markers1)%18], color=mpl.colormaps.get_cmap('hsv')((len(markers1)*25)%256), picker = 5)
          markers1.append(marker)
          plt.draw()
          print(f'x1 = {ix}, y1 = {iy}')
        if graph == 2:
          active.append(2)
          pnts2 = np.concatenate((pnts2,np.array([[ix,iy]])))
          marker = plt.plot(event.xdata, event.ydata, characters[len(markers2)%18], color=mpl.colormaps.get_cmap('hsv')((len(markers2)*25)%256), picker = 5)
          markers2.append(marker)
          plt.draw()
          print(f'x2 = {ix}, y2 = {iy}')
    if event.dblclick and str(event.button) == 'MouseButton.RIGHT':
      plt.autoscale(enable=True, axis='both', tight=None)
      plt.draw()

        
  def onpick(event, graph):
    global pnts1, pnts2, markers1, markers2, active
    if str(event.mouseevent.button) == 'MouseButton.MIDDLE':
      pairpoints = event.artist
      rowel = (pairpoints.get_xdata()[0], pairpoints.get_ydata()[0])
      if graph == 1:
        rownum = np.where(np.all(pnts1 == rowel, axis = 1))[0][0]
      else:
        rownum = np.where(np.all(pnts2 == rowel, axis = 1))[0][0]
      if len(pnts1) >= rownum + 1 and len(pnts2) >= rownum + 1:
        pnts1 = np.delete(pnts1, rownum, axis = 0)
        pnts2 = np.delete(pnts2, rownum, axis = 0)      
        if len(markers1) - rownum <= active.count(1):
          m1row = len(markers1) - rownum
          m2row = len(markers2) - rownum
          A1 = tnuoc(m1row, 1)
          active.pop(A1)
          A2 = tnuoc(m2row, 2)
          active.pop(A2)
        markers1.pop(rownum)[0].remove()
        markers2.pop(rownum)[0].remove()        
        fig1.canvas.draw()
        fig2.canvas.draw()
    
  def on_key_press(event):
    global pnts1, pnts2, markers1, markers2, active
    
    if event.key == 'u':
      print('Undo')
      if active:
        if active[-1] == 1:
          lastact = active.pop()
          last_marker = markers1.pop()
          last_marker[0].remove()
          pnts1 = pnts1[:-1]
        else:
          lastact = active.pop()
          last_marker = markers2.pop()
          last_marker[0].remove()
          pnts2 = pnts2[:-1]
        fig1.canvas.draw()
        fig2.canvas.draw()
        
    if event.key == 'w':
      plt.close('all')
      
    if event.key == 'p':
      if len(pnts1) >= 4 and (len(pnts1) == len(pnts2)):
        def on_press(event):
          if event.button == 1:
            fig3.canvas.toolbar.press_pan(event)
        def on_release(event):
          if event.button == 1:
            fig3.canvas.toolbar.release_pan(event)
        
        for fig in plt.get_fignums():
          if plt.figure(fig).canvas.manager.get_window_title() == 'Figure 3':
            plt.close(fig)
            break
        if Homography:
          hom, _ = cv2.findHomography(pnts1, pnts2, cv2.RANSAC)
          temp1 = cv2.warpPerspective(img1, hom, (img2.shape[1],img2.shape[0]), flags = cv2.INTER_CUBIC)
        elif warp:
          temp1 = np.pad(img1,[(0,max(0,img2.shape[0]-img1.shape[0])),(0,max(0,img2.shape[1]-img1.shape[1])),(0,0)])
          temp1 = WarpImage_TPS(pnts1, pnts2, temp1, 1)
          temp1 = temp1[0:img2.shape[0],0:img2.shape[1],:]
        else:
          hom, _ = cv2.estimateAffine2D(pnts1, pnts2, cv2.RANSAC)
          if not rotate:
            sx = math.sqrt(h[0,0]**2+h[1,0]**2)
            sy = math.sqrt(h[0,1]**2+h[1,1]**2)
            h[:,:2] = np.array([[sx,0],[0,sy]])
          temp1 = cv2.warpAffine(img1, hom, (img2.shape[1],img2.shape[0]), flags = cv2.INTER_CUBIC)
        preview = cv2.addWeighted(temp1,0.5,img2,0.5,0)
        with plt.ioff():
          fig3, ax3 = plt.subplots()
        ax3.imshow(cv2.cvtColor(preview,cv2.COLOR_BGR2RGB))
        plt.get_current_fig_manager().toolbar.pack_forget()
        disconnect_zoom3 = zoom_factory(ax3)
        cid = fig3.canvas.mpl_connect('button_press_event', lambda event: onclick(event, 3) if event.dblclick else None)
        cid_key = fig3.canvas.mpl_connect('key_press_event', on_key_press)
        cidpress = fig3.canvas.mpl_connect('button_press_event', on_press)
        cidrelease = fig3.canvas.mpl_connect('button_release_event', on_release)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
      else:
        print('At least 4 points must be selected and the same number of points must be on each image.')
  
  # Generate the plots and link functions and controls
  preview = img2[:]
  repeat = 0
  while len(pnts1) < 4 or (len(pnts1) != len(pnts2)) or repeat == 0:
    with plt.ioff():
      fig1, ax1 = plt.subplots()
    fig1.subplots_adjust(left=0,bottom=0,right=1,top=1)
    ax1.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
    plt.get_current_fig_manager().toolbar.pack_forget()
    disconnect_zoom1 = zoom_factory(ax1)
    pan_handler1 = panhandler(fig1,button=1)

    if len(pnts1) > 0 or len(pnts2) > 0:
      markers1 = []
      i = 0
      for redo in pnts1:
        markers1.append(plt.plot(redo[0], redo[1], characters[i%18], color=mpl.colormaps.get_cmap('hsv')((i*25)%256), picker = 5))
        i += 1
    fig1.canvas.mpl_disconnect(fig1.canvas.manager.key_press_handler_id)
    cid = fig1.canvas.mpl_connect('button_press_event', lambda event: onclick(event, 1) if event.dblclick else None)
    cid_pick = fig1.canvas.mpl_connect('pick_event', lambda event: onpick(event,1))
    cid_key = fig1.canvas.mpl_connect('key_press_event', on_key_press)

    plt.axis('off')
    plt.tight_layout()
    
    with plt.ioff():
      fig2, ax2 = plt.subplots()
    fig2.subplots_adjust(left=0,bottom=0,right=1,top=1)
    ax2.imshow(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))
    plt.get_current_fig_manager().toolbar.pack_forget()
    disconnect_zoom2 = zoom_factory(ax2)
    pan_handler2 = panhandler(fig2,button=1)

    if len(pnts1) > 0 or len(pnts2) > 0:
      markers2 = []
      i = 0
      for redo in pnts2:
        markers2.append(plt.plot(redo[0], redo[1], characters[i%18], color=mpl.colormaps.get_cmap('hsv')((i*25)%256), picker = 5))
        i += 1
    fig2.canvas.mpl_disconnect(fig2.canvas.manager.key_press_handler_id)
    cid = fig2.canvas.mpl_connect('button_press_event', lambda event: onclick(event, 2) if event.dblclick else None)
    cid_pick = fig2.canvas.mpl_connect('pick_event', lambda event: onpick(event,2))
    cid_key = fig2.canvas.mpl_connect('key_press_event', on_key_press)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    if len(pnts1) < 4 or (len(pnts1) != len(pnts2)):
      print('At least 4 points must be selected and the same number of points must be on each image.')
    repeat = 1
  return pnts1, pnts2

# Automatic point finding with SIFT
def auto_points(im1, im2):

  im1y, im1x, _ = im1.shape
  im2y, im2x, _ = im2.shape

  im1 = cv2.resize(im1,(max(im1x,im2x),max(im1y,im2y)),interpolation=cv2.INTER_CUBIC)
  im2 = cv2.resize(im2,(max(im1x,im2x),max(im1y,im2y)),interpolation=cv2.INTER_CUBIC)

  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
  
  sift = cv2.SIFT_create(MAX_FEATURES)
  keypoints1, descriptors1 = sift.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = sift.detectAndCompute(im2Gray, None)
  
  bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
  matches = bf.knnMatch(descriptors1,descriptors2,k=2)
  
  good = []
  for m,n in matches:
    if m.distance < 0.7*n.distance:
      good.append(m)
  
  if len(good) > 5:
    points1 = np.float32([ keypoints1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    points2 = np.float32([ keypoints2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
  
  points1[:,0,0], points1[:,0,1] = points1[:,0,0]*im1x/max(im1x,im2x), points1[:,0,1]*im1y/max(im1y,im2y)
  points2[:,0,0], points2[:,0,1] = points2[:,0,0]*im2x/max(im1x,im2x), points2[:,0,1]*im2y/max(im1y,im2y)

  return points1, points2

# Find a large usable rectangle from a transformed dummy array
def find_rectangle(arr):
  
  # Find the center of mass of the transform
  arrrow, arrcol = arr.shape
  rowCoM, colCoM = np.around(ndimage.center_of_mass(arr)).astype('int')
  CoM = np.array([rowCoM,colCoM])
  pntA, pntB, pntC, pntD = CoM[:], CoM[:], CoM[:], CoM[:]
  Top, Right, Bottom, Left = True, True, True, True
  
  # Increase the borders clockwise until the perimeter of the transform runs out of bounds
  while Top or Right or Bottom or Left:
    if Top:
      if pntA[0] >= 2 and np.min(arr[pntA[0]-2,pntA[1]:pntB[1]+1]) == 1:
        pntA, pntB = pntA + [-2,0], pntB + [-2,0]
      else: 
        if pntA[0] >= 1 and np.min(arr[pntA[0]-1,pntA[1]:pntB[1]+1]) == 1:
          pntA, pntB = pntA + [-1,0], pntB + [-1,0]
          Top = False
        else:
          Top = False
    
    if Right:
      if pntD[1] <= arrcol - 3 and np.min(arr[pntB[0]:pntD[0]+1,pntB[1]+2]) == 1:
        pntB, pntD = pntB + [0,2], pntD + [0,2]
      else: 
        if pntB[1] <= arrcol - 2 and np.min(arr[pntB[0]:pntD[0]+1,pntB[1]+1]) == 1:
          pntB, pntD = pntB + [0,1], pntD + [0,1]
          Right = False
        else:
          Right = False
    
    if Bottom:
      if pntD[0] <= arrrow - 3 and np.min(arr[pntC[0]+2,pntC[1]:pntD[1]+1]) == 1:
        pntC, pntD = pntC + [2,0], pntD + [2,0]
      else: 
        if pntD[0] <= arrrow - 2 and np.min(arr[pntC[0]+1,pntC[1]:pntD[1]+1]) == 1:
          pntC, pntD = pntC + [1,0], pntD + [1,0]
          Bottom = False
        else:
          Bottom = False
  
    if Left:
      if pntA[1] >= 2 and np.min(arr[pntA[0]:pntC[0]+1,pntA[1]-2]) == 1:
        pntA, pntC = pntA + [0,-2], pntC + [0,-2]
      else: 
        if pntA[1] >= 1 and np.min(arr[pntA[0]:pntC[0]+1,pntA[1]-1]) == 1:
          pntA, pntC = pntA + [0,-1], pntC + [0,-1]
          Left = False
        else:
          Left = False
       
  return pntA, pntD

def Align_Process(im1, im2):
  
  # Make dummy array the dimensions of image 1
  im1y, im1x, _ = im1.shape
  im2y, im2x, _ = im2.shape
  white1 = np.zeros((im1y,im1x))
  white1.fill(1)
  
  if Manual:
    points1, points2 = manual_points(im1, im2)
  else:
    points1, points2 = auto_points(im1, im2)
    if semiauto:
      points1, points2 = ransac(points1, points2)
      _, ind1 = np.unique(points1, axis=0, return_index=True)
      _, ind2 = np.unique(points2, axis=0, return_index=True)
      remrows = np.intersect1d(ind1, ind2)
      points1, points2 = points1[remrows], points2[remrows]
      points1, points2 = manual_points(im1, im2, points1, points2)
  
  # Find transform based on points
  if Homography:
    smat = np.array([[scale,0,0],[0,scale,0],[0,0,1]])
    h, _ = cv2.findHomography(points1, points2, cv2.RANSAC)
    warp1 = cv2.warpPerspective(white1,h,(im2x,im2y),flags=0)

  elif warp:
    white1 = np.pad(white1,[(0,max(0,im2y-im1y)),(0,max(0,im2x-im1x))])
    warp1 = WarpImage_TPS(points1, points2, white1, 0)
    warp1 = warp1[0:im2y,0:im2x]
    
  else:
    smat = np.array([[scale,0],[0,scale]])
    h, _ = cv2.estimateAffine2D(points1, points2, cv2.RANSAC)
    if not rotate:
      sx = math.sqrt(h[0,0]**2+h[1,0]**2)
      sy = math.sqrt(h[0,1]**2+h[1,1]**2)
      h[:,:2] = np.array([[sx,0],[0,sy]])
      
    warp1 = cv2.warpAffine(white1,h,(im2x,im2y),flags=0)
    
  # Get usable overlapping region
  top_left, bottom_right = find_rectangle(warp1)
  
  if not warp:
    newh = smat @ h
  
  # Ensure integer multiple scale down for mode 1
  if mode == 1:
    bottom_right[0] = bottom_right[0] - (bottom_right[0] - top_left[0] + 1) % (1/scale)
    bottom_right[1] = bottom_right[1] - (bottom_right[1] - top_left[1] + 1) % (1/scale)
  
  # Transform image 1
  if Homography:
    im1 = cv2.warpPerspective(im1,newh,(int(scale*(bottom_right[1]+1)),int(scale*(bottom_right[0]+1))),flags=cv2.INTER_CUBIC)
  elif warp:
    im1 = np.pad(im1,[(0,int(np.around(max(0,scale*im2y-im1y)))),(0,int(np.around(max(0,scale*im2x-im1x)))),(0,0)])
    im1 = WarpImage_TPS(points1, scale*points2, im1, 1)
    im1 = im1[:int(scale*(bottom_right[0]+1)),:int(scale*(bottom_right[1]+1))]
  else:
    im1 = cv2.warpAffine(im1,newh,(int(scale*(bottom_right[1]+1)),int(scale*(bottom_right[0]+1))),flags=cv2.INTER_CUBIC)
  
  # Crop images
  im1 = im1[int(scale*top_left[0]):,int(scale*top_left[1]):]
  im2 = im2[top_left[0]:(bottom_right[0]+1),top_left[1]:(bottom_right[1]+1)]
  
  return im1, im2

def align_score(img1, img2):

  img1 = cv2.resize(img1,(256,256),interpolation=cv2.INTER_CUBIC)
  img2 = cv2.resize(img2,(256,256),interpolation=cv2.INTER_CUBIC)
  points1, points2 = auto_points(img1,img2)
  points1, points2 = ransac(points1,points2)
  points = points2-points1
  score = max(1-3*(np.sum(abs(points))/len(points))/100,0)

  return score

def sort(file):
  with open(file, "r") as f:
    lines = f.readlines()
    sorted_lines = sorted(lines)

  with open(file, "w") as f:
   f.writelines(sorted_lines)



def Do_Work(hrimg, lrimg, base = None):

  highres = cv2.imread(hrimg, cv2.IMREAD_COLOR)
  lowres = cv2.imread(lrimg, cv2.IMREAD_COLOR)
  
  if autocrop:
    highres = AutoCrop(highres)
    lowres = AutoCrop(lowres)
  
  if mode == 0:
    highres, lowres = Align_Process(highres, lowres)
  
  if mode == 1:
    lowres, highres = Align_Process(lowres, highres)
  
  if color_correction == -1:
    highres = PT.pdf_transfer(img_arr_in = highres, img_arr_ref = lowres, regrain = True)
  elif color_correction == 1:
    lowres = PT.pdf_transfer(img_arr_in = lowres, img_arr_ref = highres, regrain = True)
  
  cv2.imwrite('Output/HR/{:s}.png'.format(base), highres)
  cv2.imwrite('Output/LR/{:s}.png'.format(base), lowres)
  

  if Overlay:
    
    hhr, whr, _ = highres.shape
    dim_overlay = (whr, hhr)
    scalelr = cv2.resize(lowres,dim_overlay, interpolation=cv2.INTER_CUBIC)
    overlay = cv2.addWeighted(highres,0.5,scalelr,0.5,0)
    cv2.imwrite('Output/Overlay/{:s}.png'.format(base), overlay)
  
  if score:
    try:
      ascore = align_score(lowres, highres)
    except:
      ascore = 0
    print('{:s}'.format(base)+' score: '+str(ascore))
    with open('Output/AlignmentScore.txt', 'a+') as f:
      f.write('{:s}'.format(base)+'   '+ str(ascore) +'\n')
      f.close()
 
 
if not os.path.exists('Output'):
  os.mkdir('Output')
if not os.path.exists('Output/LR'):
  os.mkdir('Output/LR')
if not os.path.exists('Output/HR'):
  os.mkdir('Output/HR')
if Overlay:
  if not os.path.exists('Output/Overlay'):
    os.mkdir('Output/Overlay')

if mode == 1:
  scale = 1/scale

# Single image pair execution
if os.path.isfile(HRfolder):
  base = os.path.splitext(os.path.basename(HRfolder))[0]
  hrim = HRfolder
  lrim = LRfolder
  Do_Work(hrim, lrim, base)

elif threads > 1:

  if len(HRfolder) == 0:
    HRfolder = 'HR'
    LRfolder = 'LR/'
  
  # Create multithreading function
  def multi(path):
      base = os.path.splitext(os.path.basename(path))[0]
      extention = os.path.splitext(os.path.basename(path))[1]
      hrim = path
      lrim = LRfolder+'/'+base+extention
      print('{:s}'.format(base)+extention)
      try:
        Do_Work(hrim,lrim,base)
      except:
        with open('Output/Failed.txt', 'a+') as f:
          f.write('{:s}'.format(base)+extention+'\n')
          f.close()
        print('Match failed for ','{:s}'.format(base)+extention)
  with ThreadPoolExecutor(max_workers=threads) as executor:
    futures = [executor.submit(multi,path)for path in glob.glob(HRfolder+'/*')]
    try:
      for future in as_completed(futures):
        future.result()
    except KeyboardInterrupt:
      for future in futures:
        future.cancel()
      if os.path.exists('Output/Failed.txt'):
        sort('Output/Failed.txt')
      if score:
        sort('Output/AlignmentScore.txt')

# Single threaded execution
else:
  if len(HRfolder) == 0:
    HRfolder = 'HR'
    LRfolder = 'LR/'
  for path in glob.glob(HRfolder+'/*'):
    base = os.path.splitext(os.path.basename(path))[0]
    extention = os.path.splitext(os.path.basename(path))[1]
    hrim = path
    lrim = LRfolder+'/'+base+extention
    print('{:s}'.format(base)+extention)
    try:
      Do_Work(hrim, lrim, base)
    except KeyboardInterrupt:
      break
    except:
      with open('Output/Failed.txt', 'a+') as f:
        f.write('{:s}'.format(base)+extention+'\n')
        f.close()
      print('Match failed for ','{:s}'.format(base)+extention)
      
if os.path.exists('Output/Failed.txt'):
  sort('Output/Failed.txt')
if score:
  sort('Output/AlignmentScore.txt')
