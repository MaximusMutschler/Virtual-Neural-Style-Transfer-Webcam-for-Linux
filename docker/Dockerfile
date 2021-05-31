#FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtimetime
FROM nvcr.io/nvidia/tensorrt:21.05-py3
# TODO wait until container with tensorrt 8 is published!
#FROM nvcr.io/nvidia/tensorrt:20.11-py3
RUN apt-get update
WORKDIR /stylecam/

COPY ./src/ ./src/
COPY ./requirements.txt .
COPY ./akvcam_config/ ./akvcam_config/
RUN mkdir data
COPY ./docker/entryscript.sh .
RUN ["chmod", "+x", "./entryscript.sh"]
RUN touch /dev/video13

##RUN ls /stylecam/data/cartoonize_models
## install akvcam
#RUN apt-get update
RUN apt update
RUN apt -y install dkms
RUN apt -y install git
RUN DEBIAN_FRONTEND=noninteractive apt install -y tzdata
RUN apt -y install linux-headers-generic
#
#
##
###RUN ls /dev/* | grep video
##
##
##RUN ln -s /usr/local/cuda-11.3 /usr/local/cuda
RUN apt update && \
    apt install -y --no-install-recommends \
      python3-numpy \
      python3-opencv \
      build-essential \
      && rm -rf /var/cache/apt/* /var/lib/apt/lists/*
RUN apt-get update &&  DEBIAN_FRONTEND=noninteractive apt-get install -y libudev-dev kbd console-data kmod

##RUN apt-get install -y python3-pip libopenblas-base libopenmpi-dev
##RUN pip3 install Cython
#
#
##RUN pip3 uninstall --yes pycuda
##RUN pip3 install --no-use-pep517  pycuda
##RUN pip install torch
##RUN pip install opencv-python
##RUN pip install torchvision
##RUN pip install numpy
##RUN pip install pynput
##RUN pip install onnx
####RUN pip install pycuda==2021.1
####RUN pip install tensorrt==8.0.0.3
##RUN pip install python-uinput
##
##
RUN pip install --upgrade pip
RUN pip install torch==1.8.1
RUN pip install opencv-python==4.5.1.48
RUN pip install torchvision==0.9.1
RUN pip install numpy==1.19.5
RUN pip install pynput==1.7.3
RUN pip install onnx==1.9.0
RUN pip install python-uinput


#### install host headers to get akvcam running in container
#: adding ubuntu 18 ppa
RUN  echo "deb http://security.ubuntu.com/ubuntu bionic-security main" >> /etc/apt/sources.list
RUN apt update
RUN apt -y install linux-headers-$(uname -r)
#sudo apt -y install software-properties-common dirmngr apt-transport-https lsb-release ca-certificates
#You should be able to use any of the listed mirrors by adding a line to your /etc/apt/sources.list like this:
 #
 #deb http://security.ubuntu.com/ubuntu bionic-security main
 # apt update

##
###RUN pip install --no-cache-dir -r requirements.txt
#
#RUN apt -y install linux-headers-4.18.0-13-generic
RUN git clone https://github.com/webcamoid/akvcam.git
#RUN mkdir -p /usr/lib/modules
#RUN  ln -s  /lib/modules/5.4.0-73-generic /usr/lib/modules/$(uname -r)
RUN cd akvcam/src && make
RUN cd akvcam/src && make dkms_install
RUN cd /stylecam/ && mkdir -p /etc/akvcam
RUN cp akvcam_config/* /etc/akvcam/

RUN pip uninstall pycuda --yes
RUN pip install pycuda==2021.1



#CMD insmod /lib/modules/$(uname -r)/updates/dkms/akvcam.ko

#ENTRYPOINT ["python3", "-u", "./src/main.py"]

