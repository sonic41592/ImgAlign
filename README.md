![ImgAlign]<img src="https://imgur.com/Ru7XVb6.png" width="600" height="500">)

# ImgAlign
If you have training images that aren't aligned properly, you've come to the right place.  This tool is useful for auto aligning, cropping, and scaling HR and LR images for training image based neural networks.  It is a CLI that takes pairs of high and low resolution images that are misaligned, misscaled, cropped out, and rotated, and outputs new, usable images for use in training neural networks.  


# Quick Start
Accepts file inputs or directories.  Add the ImgAlign.exe file to path or open a CMD prompt in the folder it is in.  Have a folder named HR and another named LR which contain the HR and LR images with matching names.  Use the options -s (or --scale) to set the scaling multiple, and -m (or --mode) to set retention mode.  Output images are saved in Output folder and are scaled properly. 

Example:

ImgAlign -s 2 -m 0

Example 2 with all options enabled with default vaules:

ImgAlign -s 2 -m 0 -g HR\ -l LR\ -c -t 15 -r -o


# Options:
The python script and exe file both work the same way.  If using the python scipt, make sure OpenCV and Pillow are installed installed using the line 'pip install opencv-python Pillow' (OpenCV not yet working on python 3.10).  It is suggested to add the exe file to path in Windows and used as a typical CLI. 


***All options are now fully functional:***

-s SCALE, --scale SCALE:                   Positive integer value. How many times bigger you want the HR resolution to be from the LR
                                          resolution.

-m MODE, --mode MODE:                      Options: 0 or 1. Mode 0 manipulates the HR images while remaining true to the LR images aside
                                          from cropping. Mode 1 manipulates the LR images and remains true to the HR images aside from
                                          cropping.

-c, --autocrop:                            Disabled by default. If enabled, this auto crops black boarders around HR and LR images.

-t THRESHOLD, --threshold THRESHOLD:       Integer 0-255, default 15. Luminance threshold for autocropping. Higher values cause more
                                          agressive cropping. Only works when autocrop is enabled.

-r, --rotate:                              Disabled by default. If enabled, this allows rotations when aligning images.

-g HR, --hr HR:                            HR File or folder directory. No need to use if they are in HR folder in current working
                                          directory.
                                          
-l LR, --lr LR:                            LR File or folder directory. No need to use if they are in LR folder in current working
                                          directory.
                                          
-o, --overlay:                             Disabled by default. After saving aligned images, this option will create a separate 50:50
                                          merge of the aligned images in the Overlay folder. Useful for quickly checking through image
                                          sets for poorly aligned outputs


# Example Images

***Github messes with the image scaling, output will be correct to scale***

Example 1: DVD and BluRay alignment

Low res DVD image:
![LR input](https://imgur.com/Ba6PSTH.png)

High res BluRay:
![HR input](https://imgur.com/KaGJigN.png)

DVD output:
![LR output](https://imgur.com/0leDQ8B.png)

BluRay output:
![HR output](https://imgur.com/c0ljhQD.png)


Rotated LR image:
![LR input](https://imgur.com/b3OnyKN.png)

Proper HR image:
![HR input](https://imgur.com/4N6Bk8q.png)

Usable area is cropped to make LR output:
![LR output](https://imgur.com/h1dr5lr.png)

HR output:
![HR output](https://imgur.com/NMc3Rai.png)



# Starting Point/Credit

I used lines of code from this site to get started with basic alignment:
https://learnopencv.com/feature-based-image-alignment-using-opencv-c-python/

Autocrop code:
https://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv

Algorithm to find the largest rectangle contained in a rotated rectangle:
https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders

