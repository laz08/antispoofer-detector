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
from sklearn.metrics.pairwise import cosine_similarity


class FaceTracker(object):
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
        self.mtcnn = MTCNN(*args, **kwargs)
        self.faces_detected = {}
        self.threshold_similarity = 0.85 # At least 0.85
        self.resize_x = 780
        self.resize_y = 550
        self.distance_bboxes_threshold = 1000
            
    def cacheFace(self, face_grayscale_resized, bbox, index):
        self.faces_detected[index] = {
            'vector': face_grayscale_resized,
            'bbox': bbox
            }
            
    def faceToGrayscaleResize(self, face):
        # Grayscale first. We are removing the RBG channels to one so we can compare the faces.
        face_grayscale = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # Resize.
        face_grayscale_resized = cv2.resize(face_grayscale, dsize=(self.resize_x, 
                                                                   self.resize_y))
        return(face_grayscale_resized)
        
    def distanceBboxes(self, bbox_1, bbox_2):
        return np.linalg.norm(np.array(bbox_1) - np.array(bbox_2))

    def computeSimilarity(self, face_1, face_2):
        return np.mean(cosine_similarity(face_1, face_2))
        
    def updateCachedFace(self, face_grayscale_resized, bbox):
        # If there are no cached faces..
        if(not len(self.faces_detected.keys())):
            self.cacheFace(face_grayscale_resized, bbox, 0)
            return
        
        # Search if it's one of the cached ones.
        for i in self.faces_detected.keys():
             faceObj = self.faces_detected[i]
             cached_vector = faceObj['vector']
             cached_bbox = faceObj['bbox']
             
             # Check bboxes to avoid comparing vector that are in regions vastly differents
             dist = self.distanceBboxes(bbox, cached_bbox)
             if(dist > self.distance_bboxes_threshold):
                 continue
             
             vec_sim = self.computeSimilarity(face_grayscale_resized, cached_vector)
             if(vec_sim >= self.threshold_similarity):
                 
                 self.cacheFace(face_grayscale_resized, bbox, i)
                 return
        
        # Not cached. Someone new. Let's save it.
        self.cacheFace(face_grayscale_resized, bbox, len(self.faces_detected.keys()))
        
    def detectAndTrackFaces(self, frames):
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
            
            for box in boxes[box_ind]:
                box = [int(b) for b in box]
                face = frame[box[1]:box[3], box[0]:box[2]]
                
                # If face detected is somehow empty, skip
                if(len(face) == 0):
                    continue
                
                # Convert to grayscale and resize
                try:
                    face_grayscale_resize = self.faceToGrayscaleResize(face)
                except:
                    # TODO Check what is happening here.
                    #print("FACE FAIL {} len: {} box {} frame {}".format(face[0], len(face[0]), box, frame))
                    return
                
                # Check if it was found in the previous frame and update
                self.updateCachedFace(face_grayscale_resize, box)
                
    def getDetectedFaces(self):
        ''' Return the faces detected so far'''
        return self.faces_detected
    