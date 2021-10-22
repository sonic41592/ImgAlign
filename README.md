# ImgAlign
For auto aligning, cropping, and scaling HR and LR images for training image based neural networks

# Usage
Make sure OpenCV is installed, 'pip install opencv-python' (OpenCV not yet working on python 3.10).

For now, the options are: mode (0 or 1), HR file name, LR file name, and scale (integer) in that other: ImgAlign.py mode HR LR scale

Example: 

ImgAlign.py 0 HR.png LR.png 2

This is still very much a work in progress. I have fairly limited coding knowledge, but am always trying to pick up new things.

I'd like to add batch functionality so that it will automatically process each picture with matching names in HR and LR directories. I also need to make the argument input nicer.

This cannot handle rotations at the moment, but I am going to try to add that feature soon.

ImgAlign can scale height and width independently, but being more similar tends to give better results. For instance, DVD images are stored at 720x480 resolution, but are almost always displayed at 720x540 or 640x480 (Also known as anamorphic, where SAR≠PAR). To match that with a 1920x1080 image (SAR=PAR), you'd get better results prescaling the the LR image (or HR image) to the intended 720x540 or 640x480 (1920x1280, 1620x1080, 1440x960, etc. for HR) than leaving it at 720x480, although either way works. 

Mode 0 is true to the LR file, meaning it maintains the resolution, aspect ratio, and orientation of the LR image, cropping where needed. The HR image is cropped, scaled, and translated accordingly.

Mode 1 is true to the HR image, maintaining its resolution, orientaion, and aspect ratio.  The LR image is cropped, scaled, translated to match.  **I have not added a boundary check for this mode yet, so the HR image should be fully contained within the LR image, or else black bars will likely be added.  I also haven't yet added a check to make sure the HR resolution is evenly divisible by scale, so be sure it is before using** This mode only outputs a new LR image because, as stated, the HR should be contained in the other image, so no cropping is needed.

# Starting Point/Credit

I used lines of code from this site to get started with basic alignment:
https://learnopencv.com/feature-based-image-alignment-using-opencv-c-python/
