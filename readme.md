# Virtual Neural Style Transfer Webcam for Linux
Ever wanted to have cool and unique filters for your video call? You found it!  
This repository provides you a virtual linux webcam* which applies a [artistic neural style transfer](https://github.com/pytorch/examples/tree/master/fast_neural_style) or a [cartoon style transfer](https://github.com/SystemErrorWang/White-box-Cartoonization) to your webcam video.
Own styles trained with the code provided by [artistic neural style transfer](https://github.com/pytorch/examples/tree/master/fast_neural_style) or  [cartoon style transfer](https://github.com/SystemErrorWang/White-box-Cartoonization/tree/master/train_code)  can also be used.

*Only tested with ubuntu 18.04 so far

## Source and Acknowledgement   
This work builds upon:  
hipersayanX [akvcam](https://github.com/webcamoid/akvcam)    
fangfufu  [Linux-Fake-Background-Webcam](https://github.com/fangfufu/Linux-Fake-Background-Webcam)  
Leon Gatys et. al. and the Pytorch team [artistic neural style transfer](https://github.com/pytorch/examples/tree/master/fast_neural_style)  
Xinrui Wang et.al. ant the Tensorflow team   [cartoon style transfer](https://github.com/SystemErrorWang/White-box-Cartoonization)   
Many thanks for their contributions.


## Requirements
1.  The [akvcam](https://github.com/webcamoid/akvcam) has to be installed. Please, follow [their wiki](https://github.com/webcamoid/akvcam/wiki) to install it.
    In our case (Ubuntu 18.04) in contrast to their documentation the driver is located at:  
    `lib/modules/$(uname -r)/updates/dkms/akvcam.ko`
2.  Copy the akvcam configuration files:   
    `sudo mkdir -p /etc/akvcam`
    `cp akvcam_config/* /etc/akvcam/`
     The akvcam output device is now located at `/dev/video3`  (this is the one you have to provide to the fakecam scipt)  
     The akvcam capture device is now located at `/dev/video2` (This is the one you have to choose in the software that displays your webcam video )
3.  Have a good graphics card. With a Geforce 2080TI we could achieve 12 fps for the artistic style tansfer and 16 fps for the cartoon style transfer with a resolution of 1280x720
4.  Install the cuda libraries. I have [cuda 10.0](https://developer.nvidia.com/cuda-10.0-download-archive) installed.
5.  Install torch 1.6.0 (only needed for the artistic style transfer). Newer versions will probably work as well.     
    `pip install torch=1.6.0`
6.  Install tensorflow 1.15 (only needed for the cartoon style transfer). Tensorflow 2.* is not supported!  
    `pip install tensorflow-gpu==1.15`
    
7. Further python packages required are: pyfakewebcam=0.1.0, opencv-python=4.2.0.32, torchvision=0.7.0, numpy=1.18.2,  

8. Download [style models](https://u-173-c142.cs.uni-tuebingen.de/index.php/s/ierXwx3DS8X48ss).   
   Extract the file and copy the folders to `./data` .

## How to start the webcam:
1. make sure the gpu driver is loaded:  
    `sudo modprobe videodev`
2. load the akvcam driver:  
   `sudo insmod lib/modules/$(uname -r)/updates/dkms/akvcam.ko`
    
3.  run the facecam program:  
`python3 main.py --cartoonize -w /dev/video1 -v /dev/video3`  
   Remove `-cartoonize` to apply artistic style transfer.  
   -w is the path to the real webcam device.  
   -v is the path to the virtual akvcam output device.  
   use --help to see further options.

   
## How to stop the webcam:
3. unload the akkvcam driver  
    `sudo rmmod lib/modules/$(uname -r)/updates/dkms/akvcam.ko`

## How to change and adapt styles:  
Press CTRL-1 to deactivate and activate styling  
The program can iterate over all styles provided in the artistic style tansfer model dir (-s) or the cartoon style transfer model dir (-c) and in corresponding subdirs.    
Press CTRL-2 to load the previous style  
Press CTRL-3 to load the next style  
Some style models achieve better results if the styled image is smaller or larger. This does not change the video output size.    
Press CTRL-4 to decrease the scale factor of the model input  
Press CTRL-5 to increase the scale factor of the model input   
Please CTRL-c to exit  

## How to add new styles
Put additional artistic style tansfer models in the directory provided with the -s flag (default ./data/style_transfer_models)  
Put additional cartoon style tansfer models in the directory provided with the -c flag (default ./data/cartoonize_models)  
You can train own styles with the code provided by [artistic neural style transfer](https://github.com/pytorch/examples/tree/master/fast_neural_style) or  [cartoon style transfer](https://github.com/SystemErrorWang/White-box-Cartoonization/tree/master/train_code).


## License

Copyright (C) Maximus Mutschler All rights reserved. Licensed under the CC BY-NC-SA 4.0  
license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).  
Commercial application is prohibited, please remain this license if you clone this repo . 

    