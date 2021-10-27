#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import time
import torch

from input.video_loader import VideoLoader
from imutils.video import FileVideoStream

from models.fast_mtcnn import FastMTCNN

# Read the data
video_loader = VideoLoader('../data/')


# Load the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
fast_mtcnn = FastMTCNN(
    stride=4,
    resize=1, # Do not resize
    margin=14,
    factor=0.6,
    keep_all=True,
    device=device
)

# For each video, determine whether there two different people or not
for video in video_loader.videos:
    # TODO Predict
    