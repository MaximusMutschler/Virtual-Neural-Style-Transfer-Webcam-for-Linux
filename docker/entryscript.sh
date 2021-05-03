#!/bin/bash
echo Loaded Entry Script
rmmod /lib/modules/$(uname -r)/updates/dkms/akvcam.ko
insmod /lib/modules/$(uname -r)/updates/dkms/akvcam.ko
if [ "$1" == "cartoonize" ]
then
echo cartoonize flag recognized
python3 -u ./src/main.py --cartoonize
else
python3 -u ./src/main.py
fi
