# ImgAlign
For auto aligning, cropping, and scaling HR and LR images for training image based neural networks

# Usage
Make sure OpenCV is installed, 'pip install opencv-python' (not yet working on python 3.10).

For now, the options are: mode (0 or 1), HR file name, LR file name, and scale (integer) in that other: ImgAlign.py mode HR LR scale

Example: ImgAlign.py 0 HR.png LR.png 2

This is still very much a work in progress. 

I'd like to add batch functionality so that it will automatically work each picture in HR and LR directories. I also need to make the argument input nicer.

This cannot handle rotations at the moment, but I am going to try to add that feature soon.

ImgAlign can scale height and width independently, but being more similar tends to give better results. For instance, DVD images are stored at 720x480 resolution, but are almost always displayed at 720x540 or 640x480 (Also known as anamorphic, where SAR≠PAR). To match that with a 1920x1080 image (SAR=PAR), you'd get better results prescaling the the LR image (or HR image) to the intended 720x540 or 640x480 (1920x1280 or 1620x1080 for HR) than leaving it at 720x480, although either way works. 

Mode 0 is true to the LR file, meaning it maintains the resolution, aspect ratio, and orientation of the LR image, cropping where needed. The HR image is cropped, scaled, and translated accordingly.

Mode 1 is true to the HR image, maintaining its 
