#!/bin/bash

# Startup script to enable CUDA GPUs
# even without monitor or X server.
#
# Add this file to /etc/init.d and run:
#	sudo update-rc.d nvidia-modprobe.sh defaults
#
# This script is tweaked for Ubuntu.

# Try to load ubuntu's nvidia-current or nvidia's native nvidia module
/sbin/modprobe nvidia-current || /sbin/modprobe nvidia

if [ "$?" -eq 0 ]; then

 # Count the number of NVIDIA controllers found.
 N3D=`/usr/bin/lspci | grep -i NVIDIA | grep "3D controller" | wc -l`
 NVGA=`/usr/bin/lspci | grep -i NVIDIA | grep "VGA compatible controller" | wc -l`

 N=`expr $N3D + $NVGA - 1`
 for i in `seq 0 $N`; do
  mknod -m 666 /dev/nvidia$i c 195 $i;
 done

 mknod -m 666 /dev/nvidiactl c 195 255

else
 exit 1
fi

