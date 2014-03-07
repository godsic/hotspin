#! /bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../libhotspin:/usr/local/cuda/lib64:/usr/lib64/nvidia:/usr/lib64/nvidia-bumblebee
../../bin/apigen $@
