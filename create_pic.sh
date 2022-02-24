#!/bin/bash
epoch=300
for((i=0;i<epoch;i++))
do
    blenderproc run random_pic.py
done
python hdf5andpaste.py
cd output
mkdir custom
mv rgb custom
mv mask custom
mv pose custom
mv camera.txt custom