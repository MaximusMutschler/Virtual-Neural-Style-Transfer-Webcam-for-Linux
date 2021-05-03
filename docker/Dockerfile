#FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtimetime
FROM tensorflow/tensorflow:1.15.5-gpu

WORKDIR /stylecam/

COPY ./src/ ./src/
COPY ./requirements.txt .
COPY ./akvcam_config/ ./akvcam_config/
RUN mkdir data
COPY ./docker/entryscript.sh .
RUN ["chmod", "+x", "./entryscript.sh"]
RUN touch /dev/video13

#RUN ls /stylecam/data/cartoonize_models
# install akvcam
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt install -y tzdata
RUN apt -y install linux-headers-generic
RUN apt -y install linux-headers-5.4.0-72-generic
RUN apt -y install dkms
RUN apt -y install git
RUN git clone https://github.com/webcamoid/akvcam.git
RUN cd akvcam/src && make
RUN cd akvcam/src && make dkms_install
RUN cd /stylecam/ && mkdir -p /etc/akvcam
RUN cp akvcam_config/* /etc/akvcam/

#RUN ls /dev/* | grep video


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3-numpy \
      python3-opencv \
      build-essential \
      && rm -rf /var/cache/apt/* /var/lib/apt/lists/*
RUN apt-get update &&  DEBIAN_FRONTEND=noninteractive apt-get install -y libudev-dev kbd console-data kmod

RUN pip install torch==1.6.0
#RUN pip install tensorflow-gpu==1.15
RUN pip install opencv-python==4.2.0.32
RUN pip install torchvision==0.7.0
RUN pip install numpy==1.18.5
RUN pip install pynput==1.7.3
#RUN pip install --no-cache-dir -r requirements.txt
RUN pip install python-uinput


#CMD insmod /lib/modules/$(uname -r)/updates/dkms/akvcam.ko


#ENTRYPOINT ["python3", "-u", "./src/main.py"]

