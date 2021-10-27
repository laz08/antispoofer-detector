#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File to dirty and explore different options with the IDE.
 To be structured later in different files.
"""

import cv2
import time
import torch

from input.video_loader import VideoLoader
from imutils.video import FileVideoStream

from models.fast_mtcnn import FastMTCNN

# Read the data
video_loader = VideoLoader('../data/')
filenames = [str(video) for video in video_loader.videos]


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


def detectFaces(video_filename, frame_batch_size = 60, verbose = False):
    '''
    Given a video filename (path included), detect and extract its faces.
    '''
    frames_processed = 0
    n_faces_det = 0
    faces_detected = []
    
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
        if(len(frames) >= frame_batch_size or x == video_length - 1):

            faces = fast_mtcnn(frames)

            frames_processed += len(frames)
            n_faces_det += len(faces)
            faces_detected += faces
            frames = []
            
            if(verbose):
                print(
                "[INFO]",
                "  [*] FPS: {}".format(frames_processed / (time.time() - start)),
                "  [*] Nr. Faces detected: {}".format(n_faces_det)
                )
    
    video.stop()
    return(faces_detected)
                
faces = detectFaces(filenames[0])
faces

# Save data to check manually
def saveFace(face, l):
    cv2.imwrite('face_{}.jpg'.format(l), face)
    
for n in range(0, len(faces)):    
    saveFace(faces[n], n)

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Maybe use their avg..?
RESIZE_X = 780
RESIZE_Y = 550

def compareFaces(faces, x, y):
    
    face_0_gr = cv2.cvtColor(faces[x], cv2.COLOR_BGR2GRAY)
    face_1_gr = cv2.cvtColor(faces[y], cv2.COLOR_BGR2GRAY)
    
    face_0_gr = cv2.resize(face_0_gr, dsize=(RESIZE_X, RESIZE_Y)) 
    face_1_gr = cv2.resize(face_1_gr, dsize=(RESIZE_X, RESIZE_Y)) 

    dist = np.mean(cosine_similarity(face_0_gr, face_1_gr))
    return(dist)

for x in range(0, len(faces) - 1):
    dist = compareFaces(faces, x, x+1)
    print(dist)
    if(x == 10):
        break
    
# Self face
compareFaces(faces, 153, 154)

# ID Card faces
compareFaces(faces, 153, 155)
compareFaces(faces, 153, 157)
compareFaces(faces, 153, 159)
compareFaces(faces, 153, 161)
compareFaces(faces, 153, 163)
compareFaces(faces, 127, 163)
compareFaces(faces, 161, 163)

# Todo check liveness