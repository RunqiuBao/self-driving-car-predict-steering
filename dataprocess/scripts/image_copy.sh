#!/bin/bash
## Usage: bash image_copy.sh
## Author: sumanth
## Purpose: moves the images from .ros


if [ "$1" == 'left' ]; then
  image_folder="left_camera"
elif [ "$1" == 'right' ]; then
  image_folder="right_camera"
elif [ "$1" == 'center' ]; then
  image_folder="center_camera"
else
  echo ERROR: no image folder
  exit
fi

source ~/.bashrc
path="$(rospack find images)"/$image_folder

# move the images
mv ~/.ros/*.jpg $path 

