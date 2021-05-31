#!/bin/bash
echo Loaded Entry Script
rmmod /lib/modules/$(uname -r)/updates/dkms/akvcam.ko
insmod /lib/modules/$(uname -r)/updates/dkms/akvcam.ko
chmod 777 /dev/video0
chmod 777 /dev/video12
chmod 777 /dev/video13

python3 -u ./src/main.py

