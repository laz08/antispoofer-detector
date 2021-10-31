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
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

class MultiplePeopleDetector(object):
    """Face Tracker using a modified version of Fast MTCNN implementation.
    Detects and keeps track on detected faces in a video."""
    
    def __init__(self, antispoofer, stride, resize=1, *args, **kwargs):
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
        self.max_faces_detected = 0
        self.antispoofer = antispoofer
        self.min_similarity = 0.90 # Arbitrary
        self.resize_x = 780
        self.resize_y = 550
            
        
    def faceToGrayscaleResize(self, face):
        # Grayscale first. We are removing the RBG channels to one so we can compare the faces.
        face_grayscale = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # Resize.
        face_grayscale_resized = cv2.resize(face_grayscale, dsize=(self.resize_x, 
                                                                   self.resize_y))
        return(face_grayscale_resized)

    def computeSimilarity(self, face_1, face_2):
        return np.mean(cosine_similarity(face_1, face_2))
        
    # Save data to check manually
    def saveFace(self, face, l):
        cv2.imwrite('face_{}.jpg'.format(l), face)
        
    def detect(self, frames):
        '''
        Detects and tracks faces (if they have already been seen by the model).
        '''
        
        if self.resize != 1:
            frames = [
                cv2.resize(f, (int(f.shape[1] * self.resize), int(f.shape[0] * self.resize)))
                    for f in frames
            ]
                      
        boxes, probs = self.mtcnn.detect(frames[::self.stride])
                
        for i, frame in enumerate(frames):
            box_ind = int(i / self.stride)
            if boxes[box_ind] is None:
                continue
            
            # Only check when there are at least 2 faces detected at once
            if(len(boxes[box_ind]) < 2):
                continue
            #print("At least 2 boxes at once {}".format(len(boxes[box_ind])))

            faces_frame = []
            for j, box in enumerate(boxes[box_ind]):
                box = [int(b) for b in box]
                face = frame[box[1]:box[3], box[0]:box[2]]
                
                
                # If face detected is somehow empty, skip
                if(len(face) == 0):
                    continue
                
                try:
                    self.saveFace(face, "{}_{}".format(i,j))
                    face_pil = Image.fromarray(np.uint8(face)).convert('RGB')
                    is_real = self.antispoofer.predict(face_pil, as_label= False)
               
                    if(is_real): 
                        face_grayscale = self.faceToGrayscaleResize(face)
                        # If someone has already been detected in this frame, 
                        # check that maybe this second faces isn't an ID 
                        # wrongly detected as someone real.
                        similarity_reached = False
                        for stored_face in faces_frame:
                            tmp_sim = self.computeSimilarity(face_grayscale,
                                                             stored_face)
                            # Check similarity
                            if(tmp_sim >= self.min_similarity):    
                                similarity_reached = True
                                break
                                
                        # No one detected yet. Let's save the vector just in case.
                        if(not similarity_reached):
                            faces_frame.append(face_grayscale)
                except:
                    continue
                    
            if(len(faces_frame) > self.max_faces_detected):
                self.max_faces_detected = len(faces_frame)

                
    def getDetectedFaces(self):
        ''' Return the faces detected so far'''
        return self.max_faces_detected
    