#! /bin/bash


ARGV="$@"

# 1) set up environment
INITIALPATH=$PWD
cd ../../bin/
MUMAX2BIN=$PWD
cd $INITIALPATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MUMAX2BIN/../src/libmumax2
export PYTHONPATH=$PYTHONPATH:$MUMAX2BIN/../src/python

exec $MUMAX2BIN/texgen $ARGV
