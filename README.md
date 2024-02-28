<img src="https://imgur.com/Ru7XVb6.png" align="center" width="400" height="335">

# ImgAlign
This tool is useful for auto aligning, cropping, and scaling HR and LR images for training image based neural networks.  It is a CLI that takes pairs of high and low resolution images that are misaligned, misscaled, cropped out, and rotated, and outputs new, properly aligned images for use in training neural networks.  


# Quick Start
For best performance, it is highly recommended that you install torch with cuda from https://pytorch.org/get-started/locally/
ImgAlign is installed through python pip using the command: 
pip install ImgAlign

Accepts file inputs or directories for the HR and LR inputs. Open a command prompt window in a folder containing folders named HR and LR which contain the HR and LR images with matching names, or use -l and -g to specify folders and images.  Use the options -s (or --scale) to set the scaling multiple.  Output images are saved in Output folder and are scaled properly. I recommend starting with a command like "ImgAlign -s 2 -ai -j" to use optical flow to improve an affine alignment.

Example:

ImgAlign -s 2 -j -ai

Example 2 with some settings enabled with default vaules:

ImgAlign -s 2 -m 0 -g HR\ -l LR\ -c -i -1 -j -ai


# Options:
<pre>
usage: ImgAlign.exe [-h] -s SCALE [-m MODE] [-l LR] [-g HR] [-o OUTPUT] [-c] [-t THRESHOLD] [-j] [-r] [-f] [-w] [-ai]
                    [-q QUALITY] [-aq] [-u] [-a] [-O] [-i COLOR] [-n THREADS] [-e]

optional arguments:
  -h, --help            show this help message and exit
  -s SCALE, --scale SCALE
                        Positive integer value. How many times bigger you want the HR resolution to be from the LR
                        resolution.
  -m MODE, --mode MODE  Options: 0 or 1. Mode 0 manipulates the HR images while remaining true to the LR images aside
                        from cropping. Mode 1 manipulates the LR images and remains true to the HR images aside from
                        cropping. In almost every case, you will want to use mode 0 so as not to alter the
                        degradations on the LR images.
  -l LR, --lr LR        LR File or folder directory. Use this to specify your low resolution image file or folder of
                        images. By default, ImgAlign will use images in the LR folder in the current working
                        directory.
  -g HR, --hr HR        HR File or folder directory. Use this to specify your high resolution image file or folder of
                        images. By default, ImgAlign will use images in the HR folder in the current working
                        directory.
  -o OUTPUT, --output OUTPUT
                        Output folder directory. Defaults to current terminal directory. Use this to specify where
                        your Output folder will be saved.
  -c, --autocrop        Disabled by default. If enabled, this auto crops black boarders around HR and LR images.
                        Manually cropping images before running through ImgAlign will usually yield more consistent
                        results so that dark frames aren't overcropped
  -t THRESHOLD, --threshold THRESHOLD
                        Integer 0-255, default 50. Luminance threshold for autocropping. Higher values cause more
                        agressive cropping.
  -j, --affine          Basic affine alignment. Used as default if no other option is specified
  -r, --rotate          Disabled by default. If enabled, this allows rotations when aligning images.
  -f, --full            Disabled by default. If enabled, this allows full homography mapping of the image, correcting
                        rotations, translations, and perspecive warping.
  -w, --warp            Disabled by default. Match images using Thin Plate Splines, allowing full image warping.
                        Because of the nature of TPS warping, this option requires that manual or semiautomatic points
                        are used.
  -ai, --ai             Disabled by default. This option allows use of RAFT optical flow to align images. This can be
                        used in conjunction with any of the aligning methods, affine, rotation, homography, or warping
                        to improve alignment, or by itself. This method can occasionally cause artifacts in the output
                        depending on the type of low resolution images being used, this can usually be fixed by
                        lowering the quality parameter to 2 or 1.
  -q QUALITY, --quality QUALITY
                        Integer 1-3, Default 3. Quality of the AI alignment. This also functions as a maximum quality
                        used when auto quality is enabled. Higher numbers are more aggressive and ususally improves
                        alignment, but can cause AI artifacts on some sources. Lower numbers might impact alignment,
                        but causes fewer AI artifacts, uses less VRAM, runs a little faster, and is more suitable for
                        multithreading.
  -aq, --autoquality    Enabled by default. Using this option disables the auto quality step down to try to fix AI
                        artifacts.
  -u, --manual          Disabled by default. Manual mode. If enabled, this opens windows for working pairs of images
                        to be aligned. Double click pairs of matching points on each image in sequence, and close the
                        windows when finished.
  -a, --semiauto        Disabled by default. Semiautomatic mode. Automatically finds matching points, but loads them
                        into a viewer window to manually delete or add more.
  -O, --overlay         Enabled by default. After saving aligned images, this option will create a separate 50:50
                        merge of the aligned images in the Overlay folder. Useful for quickly checking through image
                        sets for poorly aligned outputs.
  -i COLOR, --color COLOR
                        Default disabled. After alignment, option -1 changes the colors of the HR image to match those
                        of the LR image. Option 1 changes the color of the LR images to match the HR images. This can
                        occasionally cause miscolored regions in the altered images, so examine the results carefully.
  -n THREADS, --threads THREADS
                        Default 1. Number of threads to use for automatic matching. Large images require a lot of RAM,
                        so start small to test first.
  -e, --score           Disabled by default. Calculate an alignment score for each processed pair of images. These
                        scores should be taken with a grain of salt, they are mainly to give a general idea of how
                        well aligned things are.

Manual Keys:
Double click left: Select point.
Click and Drag left: Pan image.
Scroll Wheel: Zoom in and out.
Click Scroll Wheel: Delete matching pairs of points.
Double Click right: Reset image view.
Spacebar: Toggle edge detection view
u: Undo last point selection.
w: Close all windows to progress.
p: Preview alignment. Overlays images using current alignment points.</pre>

# Example Images

Examples

Misaligned image pairs:
[Image pairs](https://imgur.com/a/6u60o2x)

ImgAlign -s 2 -ai -j -c

[2x with Affine and Raft](https://imgsli.com/MjQwNzM1)

ImgAlign -s 1 -ai -j -c

[1x with Affine and Raft](https://imgsli.com/MjQwNzM2)

ImgAlign -s 4 -ai -f -c

[4x With full homography and Raft](https://imgsli.com/MjQwNzM3)

ImgAlign -s 2 -ai -r -q 1

[2x with rotations and Raft quality 2](https://imgsli.com/MjQwNzM4)


# Starting Point/Credit

I used lines of code from this site to get started with basic alignment:
https://learnopencv.com/feature-based-image-alignment-using-opencv-c-python/

Autocrop code:
https://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv

Algorithm to find the largest rectangle contained in a rotated rectangle:
https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders

RAFT optical flow architecture and model:
https://github.com/princeton-vl/RAFT

Largest rectangle in histogram:
https://www.interviewbit.com/blog/largest-rectangle-in-histogram/

