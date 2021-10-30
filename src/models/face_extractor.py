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
from pathlib import Path
import numpy as np
import pickle

class FaceExtractor(object):
    """Face Tracker using a modified version of Fast MTCNN implementation.
    Detects and keeps track on detected faces in a video."""
    
    def __init__(self, stride, outdir, min_prob, resize=1, *args, **kwargs):
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
        self.mtcnn = MTCNN(*args, **kwargs)
        self.save_name = "face_{}.jpg"
        self.min_prob = min_prob
        self.outdir = outdir
            
    def detectSave(self, frames):
        '''
        Detects and tracks faces (if they have already been seen by the model).
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
                try:
                    face_prob = probs[box_ind][j]
                except:
                    print(j)
                    print(box_ind)
                    print(probs)
                    
                if(face_prob < self.min_prob):
                    continue
                
                # If face detected is somehow empty, skip
                if(len(face) == 0):
                    continue
                
                self.saveFace(face)
                
    def saveFace(self, face):
        # Count how many faces there are already
        file_paths = Path(self.outdir).glob("*.jpg")
        files = [x for x in file_paths if(x.is_file())] 
        idx = len(files)
        
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        # Save
        outfile = "{}/{}".format(self.outdir, self.save_name.format(idx))
        print("[Info] Saving {} ".format(outfile))
        cv2.imwrite(outfile, face_rgb)
                