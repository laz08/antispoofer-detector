#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Loader.

Allows loading a video into memory and accessing it frame by frame.
"""
import imutils 

from imutils.video import VideoStream
import cv2
# Library to load a video.
from pathlib import Path


class VideoLoader():
    '''
    Video Loader.
    '''
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.videos = self.getPossibleVideos()
        
    def getPossibleVideos(self):
        vid_paths = Path(self.data_path).glob('*.mp4')
        vid_files = [str(x) for x in vid_paths if x.is_file()] # Let's avoid any directories, just in case

        return(vid_files)

    def loadVideo(self, video_path):
        vs = VideoStream(video_path)
        return(vs)
        
    def loadFrame(self, videostream): 
        video = videostream.start()
        frame = video.read()
           	   
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #print(frame_rgb.shape)
        # Resize
        frame_rgb = imutils.resize(frame, height=700)
        #frame_rgb = frame.shape[1] / float(frame_rgb.shape[1])
        return frame_rgb