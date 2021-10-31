#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A modified version of Fast MTCNN.

Strided version of MTCNN to allow for a faster computation.
Also keeps track of seen faces to track them frame to frame.

sources:
    https://github.com/timesler/facenet-pytorch#guide-to-mtcnn-in-facenet-pytorch
    https://www.kaggle.com/timesler/fast-mtcnn-detector-55-fps-at-full-resolution
"""
from facenet_pytorch import MTCNN
import cv2
import time
import glob
import numpy as np
import pickle

class FaceDetector(object):
    """Face Tracker using a modified version of Fast MTCNN implementation.
    Detects and keeps track on detected faces in a video."""
    
    def __init__(self, stride, resize=1, *args, **kwargs):
        """Constructor for FastMTCNN class.
        
        Arguments:
            stride (int): The detection stride. Faces will be detected every `stride` frames
                and remembered for `stride-1` frames.
        
        Keyword arguments:
            resize (float): Fractional frame scaling. [default: {1}]
            *args: Arguments to pass to the MTCNN constructor. See help(MTCNN).
            **kwargs: Keyword arguments to pass to the MTCNN constructor. See help(MTCNN).
        """
        self.stride = stride
        self.resize = resize
        self.min_confidence = 0.97 # Arbitrary
        self.mtcnn = MTCNN(*args, **kwargs)
      
    def detect(self, frames):
        '''
        Detects faces.
        '''
        
        if self.resize != 1:
            frames = [
                cv2.resize(f, (int(f.shape[1] * self.resize), int(f.shape[0] * self.resize)))
                    for f in frames
            ]
                      
        boxes, probs = self.mtcnn.detect(frames[::self.stride])
        
        faces = []
        for i, frame in enumerate(frames):
            box_ind = int(i / self.stride)
            if boxes[box_ind] is None:
                continue
            
            for j, box in enumerate(boxes[box_ind]):
                box = [int(b) for b in box]
                face = frame[box[1]:box[3], box[0]:box[2]]

                face_prob = probs[box_ind][j]

                if(face_prob < self.min_confidence):
                    continue
                
                # If face detected is somehow empty, skip
                if(len(face) == 0):
                    continue
                
                faces.append(face)
                
        return(faces)
                