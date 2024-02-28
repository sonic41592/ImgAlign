from mpl_interactions import zoom_factory, panhandler
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os
import sys
import cv2
import glob
import textwrap
import torch
import argparse
from argparse import Namespace
import math
from itertools import groupby
from scipy.ndimage import map_coordinates
from .raft.raft import RAFT
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.linear_model import RANSACRegressor
from python_color_transfer.color_transfer import ColorTransfer
PT = ColorTransfer()


parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,epilog = textwrap.dedent('''\

Manual Keys: 
Double click left: Select point.
Click and Drag left: Pan image.
Scroll Wheel: Zoom in and out.
Click Scroll Wheel: Delete matching pairs of points.
Double Click right: Reset image view.
Spacebar: Toggle edge detection view
u: Undo last point selection.
w: Close all windows to progress.
p: Preview alignment. Overlays images using current alignment points.'''))
parser.add_argument("-s", "--scale", help="Positive integer value. How many times bigger you want the HR resolution to be from the LR resolution.", required=True)
parser.add_argument("-m", "--mode", default=0, help="Options: 0 or 1. Mode 0 manipulates the HR images while remaining true to the LR images aside from cropping. Mode 1 manipulates the LR images and remains true to the HR images aside from cropping. In almost every case, you will want to use mode 0 so as not to alter the degradations on the LR images.")
parser.add_argument("-l", "--lr", default='', help="LR File or folder directory. Use this to specify your low resolution image file or folder of images. By default, ImgAlign will use images in the LR folder in the current working directory.")
parser.add_argument("-g", "--hr", default='', help="HR File or folder directory. Use this to specify your high resolution image file or folder of images. By default, ImgAlign will use images in the HR folder in the current working directory.")
parser.add_argument("-o", "--output", default='', help="Output folder directory. Defaults to current terminal directory. Use this to specify where your Output folder will be saved.")
parser.add_argument("-c", "--autocrop", action='store_true', default=False, help="Disabled by default. If enabled, this auto crops black boarders around HR and LR images. Manually cropping images before running through ImgAlign will usually yield more consistent results so that dark frames aren't overcropped")
parser.add_argument("-t", "--threshold", default=50, help="Integer 0-255, default 50. Luminance threshold for autocropping. Higher values cause more agressive cropping.")
parser.add_argument("-j", "--affine", action='store_true', default=False, help="Basic affine alignment. Used as default if no other option is specified")
parser.add_argument("-r", "--rotate", action='store_true', default=False, help="Disabled by default. If enabled, this allows rotations when aligning images.")
parser.add_argument("-f", "--full", action='store_true', default=False, help="Disabled by default. If enabled, this allows full homography mapping of the image, correcting rotations, translations, and perspecive warping.")
parser.add_argument("-w", "--warp", action='store_true', default=False, help="Disabled by default. Match images using Thin Plate Splines, allowing full image warping. Because of the nature of TPS warping, this option requires that manual or semiautomatic points are used.")
parser.add_argument("-ai", "--ai", action='store_true', default=False, help="Disabled by default. This option allows use of RAFT optical flow to align images. This can be used in conjunction with any of the aligning methods, affine, rotation, homography, or warping to improve alignment, or by itself. This method can occasionally cause artifacts in the output depending on the type of low resolution images being used, this can usually be fixed by lowering the quality parameter to 2 or 1.")
parser.add_argument("-q", "--quality", default=3, help="Integer 1-3, Default 3. Quality of the AI alignment. This also functions as a maximum quality used when auto quality is enabled. Higher numbers are more aggressive and ususally improves alignment, but can cause AI artifacts on some sources. Lower numbers might impact alignment, but causes fewer AI artifacts, uses less VRAM, runs a little faster, and is more suitable for multithreading.")
parser.add_argument("-aq", "--autoquality", action='store_false', default=True, help="Enabled by default. Using this option disables the auto quality step down to try to fix AI artifacts.")
parser.add_argument("-u", "--manual", action='store_true', default=False, help="Disabled by default. Manual mode. If enabled, this opens windows for working pairs of images to be aligned. Double click pairs of matching points on each image in sequence, and close the windows when finished.")
parser.add_argument("-a", "--semiauto", action='store_true', default=False, help="Disabled by default. Semiautomatic mode. Automatically finds matching points, but loads them into a viewer window to manually delete or add more.")
parser.add_argument("-O", "--overlay", action='store_false', default=True, help="Enabled by default. After saving aligned images, this option will create a separate 50:50 merge of the aligned images in the Overlay folder. Useful for quickly checking through image sets for poorly aligned outputs.")
parser.add_argument("-i", "--color", default=0, help="Default disabled. After alignment, option -1 changes the colors of the HR image to match those of the LR image. Option 1 changes the color of the LR images to match the HR images. This can occasionally cause miscolored regions in the altered images, so examine the results carefully.")
parser.add_argument("-n", "--threads", default=1, help="Default 1. Number of threads to use for automatic matching. Large images require a lot of RAM, so start small to test first.")
parser.add_argument("-e", "--score", action='store_true', default=False, help="Disabled by default. Calculate an alignment score for each processed pair of images. These scores should be taken with a grain of salt, they are mainly to give a general idea of how well aligned things are.")

args = vars(parser.parse_args())

scale = float(args["scale"])
mode = int(args["mode"])
autocrop = args["autocrop"]
lumthresh = int(args["threshold"])
threads = int(args["threads"])
rotate = args["rotate"]
affi = args["affine"]
HRfolder = args["hr"]
LRfolder = args["lr"]
outfolder = args["output"]
Overlay = args["overlay"]
Homography = args["full"]
Manual = args["manual"]
score = args["score"]
semiauto = args["semiauto"]
warp = args["warp"]
color_correction = int(args["color"])
optical_flow = args["ai"]
quality = int(args["quality"])
auto_quality = args["autoquality"]

# Changing conflicting or priority setting
if optical_flow:
    Qh = (544, 720, 1080)
    Qw = (720, 960, 1440)
    gauss = ((1,1), (3,3), (5,5))
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'
    def load_image(imfile):
        img = np.array(imfile).astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(DEVICE)
    RAFT_MODULE_DIR = os.path.dirname(__file__)
    RAFT_THINGS_FILE = os.path.join(RAFT_MODULE_DIR, 'raft', 'raft-things.pth')
    args = Namespace(small=False, alternate_corr=False, mixed_precision=False, model=RAFT_THINGS_FILE)
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model,map_location=torch.device(DEVICE)))
    model = model.module
    model.to(DEVICE)
    model.eval()

if warp:
    Homography = False
    rotate = False
    if not Manual:
        semiauto = True
    
if Manual or semiauto:
    affi = True

if Manual or semiauto:
    threads = 1

if outfolder:
    outfolder = outfolder + '/'

MAX_FEATURES = 500

# Cropping dark boarders around images
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
    global pnts1, pnts2, markers1, markers2, active, dis1, dis2, axis1, axis2
    
    dis1 = img1
    dis2 = img2
    img1edge = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY), (5,5), 0), 50,150)
    img2edge = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY), (5,5), 0), 50,150)
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
    
    # Matplotlib UI function
    # Count backwards, helps redraw after scroll wheel click paired point deletion
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
    
    # Click event functions
    def onclick(event, graph):
        # Select a point
        if event.dblclick and event.button == 1:
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
        # Reset view
        if event.dblclick and event.button == 3:
            plt.autoscale(enable=True, axis='both', tight=None)
            plt.draw()

                
    def onpick(event, graph):
        global pnts1, pnts2, markers1, markers2, active
        # Scroll wheel click for paired point deletion
        if event.mouseevent.button == 2:
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
        global pnts1, pnts2, markers1, markers2, active, dis1, dis2, axis1, axis2
        
        # Undo, preview, edge detection, and close all key strokes
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
            
        if event.key == ' ':
            if dis1 is img1:
                dis1 = img1edge
                dis2 = img2edge
            else:
                dis1 = img1
                dis2 = img2
            axis1.set_data(cv2.cvtColor(dis1,cv2.COLOR_BGR2RGB))
            axis2.set_data(cv2.cvtColor(dis2,cv2.COLOR_BGR2RGB))
            fig1.canvas.draw()
            fig2.canvas.draw()
            
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
                temp2 = dis2
                if Homography:
                    hom, _ = cv2.findHomography(pnts1, pnts2, cv2.RANSAC)
                    temp1 = cv2.warpPerspective(dis1, hom, (dis2.shape[1],dis2.shape[0]), flags = cv2.INTER_CUBIC)
                elif warp:
                    if len(dis1.shape) == 2:
                        temp1 = np.pad(dis1,[(0,max(0,dis2.shape[0]-dis1.shape[0])),(0,max(0,dis2.shape[1]-dis1.shape[1]))])
                        temp1 = WarpImage_TPS(pnts1, pnts2, temp1, 1)
                        temp1 = temp1[0:dis2.shape[0],0:dis2.shape[1]]
                    else:
                        temp1 = np.pad(dis1,[(0,max(0,dis2.shape[0]-dis1.shape[0])),(0,max(0,dis2.shape[1]-dis1.shape[1])),(0,0)])
                        temp1 = WarpImage_TPS(pnts1, pnts2, temp1, 1)
                        temp1 = temp1[0:dis2.shape[0],0:dis2.shape[1],:]
                else:
                    hom, _ = cv2.estimateAffine2D(pnts1, pnts2, cv2.RANSAC)
                    if not rotate:
                        sx = math.sqrt(hom[0,0]**2+hom[1,0]**2)
                        sy = math.sqrt(hom[0,1]**2+hom[1,1]**2)
                        hom[:,:2] = np.array([[sx,0],[0,sy]])
                    temp1 = cv2.warpAffine(dis1, hom, (dis2.shape[1],dis2.shape[0]), flags = cv2.INTER_CUBIC)
                with plt.ioff():
                    fig3, ax3 = plt.subplots()
                ani = ax3.imshow(cv2.cvtColor(temp1,cv2.COLOR_BGR2RGB))
                def update(frame):
                    if frame % 2 == 0:
                        ani.set_array(cv2.cvtColor(temp1,cv2.COLOR_BGR2RGB))
                    else:
                        ani.set_array(cv2.cvtColor(temp2,cv2.COLOR_BGR2RGB))
                animation = FuncAnimation(fig3, update, frames=np.arange(0,10), interval=500)
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
    preview = dis2[:]
    repeat = 0
    while len(pnts1) < 4 or (len(pnts1) != len(pnts2)) or repeat == 0:
        with plt.ioff():
            fig1, ax1 = plt.subplots()
        fig1.subplots_adjust(left=0,bottom=0,right=1,top=1)
        axis1 = ax1.imshow(cv2.cvtColor(dis1,cv2.COLOR_BGR2RGB))
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
        axis2 = ax2.imshow(cv2.cvtColor(dis2,cv2.COLOR_BGR2RGB))
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

    points1, points2 = ransac(points1, points2)
    _, ind1 = np.unique(points1, axis=0, return_index=True)
    _, ind2 = np.unique(points2, axis=0, return_index=True)
    remrows = np.intersect1d(ind1, ind2)
    points1, points2 = points1[remrows], points2[remrows]
    
    return points1, points2

# Creates a histogram of the longest stretch of consecutive ones for every row in an array
def longest_ones(matrix):
    result = []
    for row in matrix:
        max_stretch = 0
        for _, group in groupby(row):
            if _ == 1:
                max_stretch = max(max_stretch, len(list(group)))
        result.append(max_stretch)
    return result

# Finds the bounds of the largest rectangle in a histogram
def get_largest_rectangle_indices(heights):
    stack = [-1]
    max_area = 0
    max_indices = (0, 0)
    for i in range(len(heights)):
        while stack[-1] != -1 and heights[stack[-1]] >= heights[i]:
            current_height = heights[stack.pop()]
            current_width = i - stack[-1] - 1
            current_area = current_height * current_width
            if current_area > max_area:
                max_area = current_area
                max_indices = (stack[-1] + 1, i - 1)
        stack.append(i)
    while stack[-1] != -1:
        current_height = heights[stack.pop()]
        current_width = len(heights) - stack[-1] - 1
        current_area = current_height * current_width
        if current_area > max_area:
            max_area = current_area
            max_indices = (stack[-1] + 1, len(heights) - 1)
    return max_indices

# Find a large usable rectangle from a transformed dummy array
def find_rectangle(arr):
    
    rowhist = longest_ones(arr)
    colhist = longest_ones(arr.T)
    rows = get_largest_rectangle_indices(rowhist)
    cols = get_largest_rectangle_indices(colhist)
    
    if 0 in arr[rows[0]:rows[1]+1,cols[0]:cols[1]+1]:
        while 0 in arr[rows[0]:rows[1]+1,cols[0]:cols[1]+1]:
            rows += np.array([1,-1])
            cols += np.array([1,-1])
        while cols[0] > 0 and 0 not in arr[rows[0]:rows[1]+1,cols[0]-1]:
            cols[0] -= 1
        while cols[1] < arr.shape[1]-1 and 0 not in arr[rows[0]:rows[1]+1,cols[1]+1]:
            cols[1] += 1
        while rows[0] > 0 and 0 not in arr[rows[0]-1,cols[0]:cols[1]+1]:
            rows[0] -= 1
        while rows[1] < arr.shape[0]-1 and 0 not in arr[rows[1]+1,cols[0]:cols[1]+1]:
            rows[1] += 1
    
    return np.array([rows[0], cols[0]]), np.array([rows[1], cols[1]])

def find_map(aim1, aim2, qual):
    aim1r = cv2.resize(aim1,(Qw[qual-1],Qh[qual-1]),cv2.INTER_CUBIC)
    aim2r = cv2.resize(aim2,(Qw[qual-1],Qh[qual-1]),cv2.INTER_CUBIC)
    aim1g=cv2.cvtColor(cv2.cvtColor(aim1r,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
    aim2g=cv2.cvtColor(cv2.cvtColor(aim2r,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
    aim1g = cv2.GaussianBlur(aim1g, gauss[qual-1], 0)
    aim2g = cv2.GaussianBlur(aim2g, gauss[qual-1], 0)
    with torch.no_grad():
        image1 = load_image(aim1g)
        image2 = load_image(aim2g)
        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
        displacement = flow_up[0].permute(1,2,0).detach().cpu().numpy()
        return displacement

# Improves alignment using RAFT
def AI_Align_Process(aim1, aim2, pre = 0):
    # Precrops the images to overlapping regions if they aren't prealigned
    if pre == 0:
        aim1y, aim1x, _ = aim1.shape
        aim2y, aim2x, _ = aim2.shape
        prewhite = np.ones_like(aim1[:,:,0])
        taim1 = cv2.resize(aim1,(min(aim1x,aim2x),min(aim1y,aim2y)),interpolation=cv2.INTER_CUBIC)
        taim2 = cv2.resize(aim2,(min(aim1x,aim2x),min(aim1y,aim2y)),interpolation=cv2.INTER_CUBIC)
        points1, points2 = auto_points(taim1, taim2)
        points1[:,0,0], points1[:,0,1] = points1[:,0,0]*aim1x/min(aim1x,aim2x), points1[:,0,1]*aim1y/min(aim1y,aim2y)
        points2[:,0,0], points2[:,0,1] = points2[:,0,0]*aim2x/min(aim1x,aim2x), points2[:,0,1]*aim2y/min(aim1y,aim2y)
        h, _ = cv2.findHomography(points1, points2, cv2.RANSAC)
        prewarp = cv2.warpPerspective(prewhite,h,(aim2x,aim2y),flags=0)
        pntsA, pntsD = find_rectangle(prewarp)
        ih = np.linalg.inv(h)
        A = np.matmul(ih,np.array([pntsA[1], pntsA[0], 1]).T)
        B = np.matmul(ih,np.array([pntsA[1], pntsD[0], 1]).T)
        C = np.matmul(ih,np.array([pntsD[1], pntsA[0], 1]).T)
        D = np.matmul(ih,np.array([pntsD[1], pntsD[0], 1]).T)
        top = int(np.clip(min(A[1], B[1], C[1], D[1]),0,aim1y))
        bottom = int(np.clip(max(A[1], B[1], C[1], D[1]),0,aim1y))
        left = int(np.clip(min(A[0], B[0], C[0], D[0]),0,aim1x))
        right = int(np.clip(max(A[0], B[0], C[0], D[0]),0,aim1x))
        aim1 = aim1[top:bottom+1,left:right+1]
        aim2 = aim2[pntsA[0]:pntsD[0]+1,pntsA[1]:pntsD[1]+1]
    # RAFT mapping
    aim1y, aim1x, _ = aim1.shape
    aim2y, aim2x, _ = aim2.shape
    q = quality
    if mode == 1:
        aim2 = aim2[:aim2y-int((aim2y%ogscale)),:aim2x-int((aim2x%ogscale)),:]
    displacement = find_map(aim1, aim2, q)
    magnitude = np.sqrt(displacement[:,:,0]**2+displacement[:,:,1]**2)
    gradienty = np.gradient(magnitude,axis=0)
    gradientx = np.gradient(magnitude,axis=1)
    grangex = gradientx.max() - gradientx.min()
    grangey = gradienty.max() - gradienty.min()
    
    if auto_quality:
        while (max(grangex, grangey) > 3.25 or max(gradientx.std(), gradienty.std()) > 0.1) and q > 1:
            q -= 1
            print("Artifacts detected, lowering to quality "+ str(q) +" and trying again.")
            displacement = find_map(aim1, aim2, q)
            magnitude = np.sqrt(displacement[:,:,0]**2+displacement[:,:,1]**2)
            gradienty = np.gradient(magnitude,axis=0)
            gradientx = np.gradient(magnitude,axis=1)
            grangex = gradientx.max() - gradientx.min()
            grangey = gradienty.max() - gradienty.min()
    if max(grangex, grangey) > 3.25 or max(gradientx.std(), gradienty.std()) > 0.1:
        print("Artifacts are likely present on "+'{:s}'.format(base)+". Name saved to Artifacts.txt for later inspection.")
        with open(outfolder+'Output/Artifacts.txt', 'a+') as f:
            f.write('{:s}'.format(base)+'\n')
            f.close()
    grid_array = np.indices((Qh[q-1], Qw[q-1]),dtype='float').transpose(1,2,0)
    grid_array[:,:,[0,1]] = grid_array[:,:,[1,0]]
    dis = grid_array-displacement
    map = cv2.resize(dis, (int(scale*aim2x),int(scale*aim2y)),cv2.INTER_CUBIC)
    map[:,:,0] = map[:,:,0]*aim1x/Qw[q-1]
    map[:,:,1] = map[:,:,1]*aim1y/Qh[q-1]
    warpr = map_coordinates(aim1[:,:,0],(map[:,:,1],map[:,:,0]), order=3, mode='nearest')
    warpb = map_coordinates(aim1[:,:,1],(map[:,:,1],map[:,:,0]), order=3, mode='nearest')
    warpg = map_coordinates(aim1[:,:,2],(map[:,:,1],map[:,:,0]), order=3, mode='nearest')
    warp = cv2.merge((warpr,warpb,warpg))
    white = np.ones_like(aim1[:,:,0])
    mapw = cv2.resize(dis, (aim2x,aim2y),cv2.INTER_CUBIC)
    mapw[:,:,0] = mapw[:,:,0]*aim1x/Qw[q-1]
    mapw[:,:,1] = mapw[:,:,1]*aim1y/Qh[q-1]
    warpw = map_coordinates(white,(mapw[:,:,1],mapw[:,:,0]), order=3, mode='constant')
    top_left, bottom_right = find_rectangle(warpw)
    if mode == 1:
        top_left[0] = top_left[0] + top_left[0] % ogscale
        top_left[1] = top_left[1] + top_left[1] % ogscale
        bottom_right[0] = bottom_right[0] - (bottom_right[0]+1) % ogscale
        bottom_right[1] = bottom_right[1] - (bottom_right[1]+1) % ogscale
    warp = warp[int(scale*top_left[0]):int(scale*(bottom_right[0]+1)),int(scale*top_left[1]):int(scale*(bottom_right[1]+1))]
    aim2 = aim2[top_left[0]:(bottom_right[0]+1),top_left[1]:(bottom_right[1]+1)]
    return warp, aim2

def Align_Process(im1, im2):
    
    # Make dummy array the dimensions of image 1
    im1y, im1x, _ = im1.shape
    im2y, im2x, _ = im2.shape
    white1 = np.ones_like(im1[:,:,0])
    
    if Manual:
        points1, points2 = manual_points(im1, im2)
    else:
        points1, points2 = auto_points(im1, im2)
        if semiauto:
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
    
    if optical_flow:
        im1, im2 = AI_Align_Process(im1, im2, pre = 1)
    
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
    
    if optical_flow and not (affi or rotate or Homography or warp):
        if mode == 0:
            highres, lowres = AI_Align_Process(highres, lowres)
    
        if mode == 1:
            lowres, highres = AI_Align_Process(lowres, highres)
            
    else:
        if mode == 0:
            highres, lowres = Align_Process(highres, lowres)
    
        if mode == 1:
            lowres, highres = Align_Process(lowres, highres)
    
    lowres = lowres[:lowres.shape[0]-lowres.shape[0]%2,:lowres.shape[1]-lowres.shape[1]%2]
    highres = highres[:int(ogscale*lowres.shape[0]),:int(ogscale*lowres.shape[1])]
    
    if color_correction == -1:
        highres = PT.pdf_transfer(img_arr_in = highres, img_arr_ref = lowres, regrain = True)
    elif color_correction == 1:
        lowres = PT.pdf_transfer(img_arr_in = lowres, img_arr_ref = highres, regrain = True)
    
    cv2.imwrite(outfolder+'Output/HR/{:s}.png'.format(base), highres)
    cv2.imwrite(outfolder+'Output/LR/{:s}.png'.format(base), lowres)
    

    if Overlay:
        
        hhr, whr, _ = highres.shape
        dim_overlay = (whr, hhr)
        scalelr = cv2.resize(lowres,dim_overlay, interpolation=cv2.INTER_CUBIC)
        overlay = cv2.addWeighted(highres,0.5,scalelr,0.5,0)
        cv2.imwrite(outfolder+'Output/Overlay/{:s}.png'.format(base), overlay)
    
    if score:
        try:
            ascore = align_score(lowres, highres)
        except:
            ascore = 0
        print('{:s}'.format(base)+' score: '+str(ascore))
        with open(outfolder+'Output/AlignmentScore.txt', 'a+') as f:
            f.write('{:s}'.format(base)+'     '+ str(ascore) +'\n')
            f.close()
 
 
if not os.path.exists(outfolder+'Output'):
    os.mkdir(outfolder+'Output')
if not os.path.exists(outfolder+'Output/LR'):
    os.mkdir(outfolder+'Output/LR')
if not os.path.exists(outfolder+'Output/HR'):
    os.mkdir(outfolder+'Output/HR')
if Overlay:
    if not os.path.exists(outfolder+'Output/Overlay'):
        os.mkdir(outfolder+'Output/Overlay')

ogscale = scale
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
                with open(outfolder+'Output/Failed.txt', 'a+') as f:
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
            if os.path.exists(outfolder+'Output/Failed.txt'):
                sort(outfolder+'Output/Failed.txt')
            if score:
                sort(outfolder+'Output/AlignmentScore.txt')

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
            with open(outfolder+'Output/Failed.txt', 'a+') as f:
                f.write('{:s}'.format(base)+extention+'\n')
                f.close()
            print('Match failed for ','{:s}'.format(base)+extention)
            
if os.path.exists(outfolder+'Output/Failed.txt'):
    sort(outfolder+'Output/Failed.txt')
if os.path.exists(outfolder+'Output/Artifacts.txt'):
    sort(outfolder+'Output/Artifacts.txt')
if score:
    sort(outfolder+'Output/AlignmentScore.txt')

def __main__():
    pass
