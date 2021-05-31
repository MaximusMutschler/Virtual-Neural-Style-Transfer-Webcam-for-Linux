FROM nvcr.io/nvidia/tensorrt:21.05-py3

RUN apt-get update
WORKDIR /stylecam/

COPY ./src/ ./src/
COPY ./requirements.txt .
COPY ./akvcam_config/ ./akvcam_config/
RUN mkdir data
COPY ./docker/entryscript.sh .
RUN ["chmod", "+x", "./entryscript.sh"]
RUN touch /dev/video13


RUN apt update
RUN apt -y install dkms
RUN apt -y install git
RUN DEBIAN_FRONTEND=noninteractive apt install -y tzdata
RUN apt -y install linux-headers-generic


RUN apt update && \
    apt install -y --no-install-recommends \
      python3-numpy \
      python3-opencv \
      build-essential \
      && rm -rf /var/cache/apt/* /var/lib/apt/lists/*
RUN apt-get update &&  DEBIAN_FRONTEND=noninteractive apt-get install -y libudev-dev kbd console-data kmod


RUN pip install --upgrade pip
RUN pip install torch==1.8.1
RUN pip install opencv-python==4.5.1.48
RUN pip install torchvision==0.9.1
RUN pip install numpy==1.19.5
RUN pip install onnx==1.9.0


# adding ubuntu 18 ppa
RUN  echo "deb http://security.ubuntu.com/ubuntu bionic-security main" >> /etc/apt/sources.list
RUN apt update
RUN apt -y install linux-headers-$(uname -r)


RUN git clone https://github.com/webcamoid/akvcam.git
RUN cd akvcam/src && make
RUN cd akvcam/src && make dkms_install
RUN cd /stylecam/ && mkdir -p /etc/akvcam
RUN cp akvcam_config/* /etc/akvcam/

RUN pip uninstall pycuda --yes
RUN pip install pycuda==2021.1

#CMD insmod /lib/modules/$(uname -r)/updates/dkms/akvcam.ko

#ENTRYPOINT ["python3", "-u", "./src/main.py"]

