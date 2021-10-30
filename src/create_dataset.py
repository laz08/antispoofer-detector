#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Example of usage...
# python create_dataset.py -i ../data/videos_training/real_1.mp4 -o ../data/dataset_training/train/real
# python create_dataset.py -i ../data/videos_training/real_2.mp4 -o ../data/dataset_training/val/real
# python create_dataset.py -i ../data/videos_training/fake_1.webm -o ../data/dataset_training/train/fake
# python create_dataset.py -i ../data/videos_training/fake_2.webm -o ../data/dataset_training/val/fake

# Misc
import numpy as np
import argparse
import cv2
import os

from imutils.video import FileVideoStream
from models.face_extractor import FaceExtractor

# Parse arguments from CLI
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
	help="path to input video")
ap.add_argument("-o", "--output", type=str, required=True,
	help="path to output directory of extracted faces")
ap.add_argument("-p", "--probability", type=str, default = 0.99,
	help="Minimum probability to accept the detected face")

args = vars(ap.parse_args())

# Load FaceExtractor based on MTCNN
print("[INFO] Load face extractor")
faceExtractor = FaceExtractor(stride = 4,
                              outdir = args['output'],
                              min_prob = args['probability'])

video = FileVideoStream(args['input']).start()
video_length = int(video.stream.get(cv2.CAP_PROP_FRAME_COUNT))

frames = []
for x in range(video_length):
    

    frame = video.read()
    try:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except:
        continue
    
    frames.append(frame)

    # If batch size is complete or the last frame reached, detect
    if(len(frames) >= 60 or (x == video_length - 1)):

        faceExtractor.detectSave(frames)

        frames = []
        
video.stop()
print("[OK] Done.")
