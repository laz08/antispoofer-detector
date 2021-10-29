#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import time
import torch

from input.video_loader import VideoLoader
from imutils.video import FileVideoStream

from models.face_tracker import FaceTracker

# Read the data
video_loader = VideoLoader('../data/')


# Load the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

fast_mtcnn = FaceTracker(
    stride=4,
    resize=1, # Do not resize
    margin=14,
    factor=0.6,
    keep_all=True,
    device=device
)


def detectFaces(video_filename, frame_batch_size = 60, verbose = False):
    '''
    Given a video filename (path included), detect and extract its faces.
    '''
    frames_processed = 0
    n_faces_det = 0
    
    # Start video
    video = FileVideoStream(video_filename).start()
    video_length = int(video.stream.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    for x in range(video_length):
        start = time.time()

        frame = video.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

        # If batch size is complete or the last frame reached, detect
        if(len(frames) >= frame_batch_size or (x == video_length - 1)):

            model.detectAndTrackFaces(frames)

            frames_processed += len(frames)
            frames = []
            
    video.stop()
    
    faces_detected = model.getDetectedFaces()
    
    return(faces_detected)

# For each video, determine whether there two different people or not
for video in video_loader.videos:
    faces_detected = detectFaces(video_filename)
    print("[*] Video '{}' has {} people.".format(video, len(faces_detected)))
    break