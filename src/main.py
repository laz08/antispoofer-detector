#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import cv2
import time
import torch

from input.video_loader import VideoLoader
from imutils.video import FileVideoStream

from models.face_detection import FaceDetector
from models.antispoofer import AntiSpoofer
from models.face_multiple_people import MultiplePeopleDetector


from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
# /home/laura/.local/lib/python3.8/site-packages/sklearn/metrics/pairwise.py:56: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
# Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations


# Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
	help="Input directory where videos are located")
ap.add_argument("-o", "--output", type=str, default='results.txt',
	help="Name of the file where to output the results")

args = vars(ap.parse_args())

# Read the data
base_path = args['input'] # '../data/videos_spoofing'
video_loader = VideoLoader(base_path)


# Load the models
device = 'cuda' if torch.cuda.is_available() else 'cpu'

antispoofer = AntiSpoofer()
peopleDet = MultiplePeopleDetector(
    antispoofer,
    stride=4,
    resize=1, # Do not resize
    margin=14,
    factor=0.6,
    keep_all=False,
    device='cpu' # MTCNN is giving CUDA OOM.. Let's keep it on CPU
)

def detectFaces(video_filename, 
                skip_frames = 15, 
                frame_batch_size = 60, 
                verbose = False):
    '''
    Given a video filename (path included), detect and extract its faces.
    '''
        
    frames_processed = 0
    n_faces_det = 0
    
    # Start video
    video = FileVideoStream(video_filename).start()
    video_length = int(video.stream.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Video length: {}".format(video_length))
    frames = []
    for x in tqdm(range(video_length)):
        start = time.time()
        
        if(not video.more()):
            break
        
        frame = video.read()
        
        if(x % skip_frames == 0):    
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
        # If batch size is complete or the last frame reached, detect
        if(len(frames) >= frame_batch_size 
           or (x == (video_length - 1))):

            peopleDet.detect(frames)
            #faceTracker.detectAndTrackFaces(frames)

            frames_processed += len(frames)
            frames = []
            
    video.stop()
    
    faces_detected = peopleDet.getDetectedFaces()
    
    return(faces_detected)

# For each video, determine whether there two different people or not
from shutil import copyfile
copyfile('../base_labels.txt', args['output'])
for video_filename in sorted(video_loader.videos):
    
    print("[*] Processing {}".format(video_filename))
    
    faces_detected = detectFaces(video_filename)
    print("[*] Video '{}' has {} real people at once.".format(video_filename, faces_detected))
    
    # Check whether there have been more than 1 person at once detected.
    label = 1 if faces_detected > 1 else 0
    data = "\t".join([video_filename.replace(base_path + '/', '').replace('.mp4', ''),
                      str(label)])
    data += '\n'
    with open(args['output'], "a") as text_file:
        text_file.write(data)

    