#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anti-Spoofing predictor.

This model detects faces, as well as determining whether 
the faces are real or spoofed.

An spoofed face is that one that belongs to an ID, a mask or a photograph.
In other words, a face lacking 'liveness'.
"""
import torch
from torchvision import models
import torch.nn  as nn
from torchvision import transforms as T
from torch import optim


class AntiSpoofer(object):
    '''
    Anti Spoofer model.
    '''
    def __init__(self, model_name = 'mobilenet_sgd_00001_id.pt'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.loadModel(model_name)
        
        # Correspondence is:
        #   - 0: Fake
        #   - 1: Real
        self.classes = ['fake', 'real']
        mean_vec = [0.485, 0.456, 0.406]
        std_vec = [0.229, 0.224, 0.225]
        
        # Use same transformations as during validation
        self.transforms = T.Compose([
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean_vec, std_vec)
                    ])
        return
    
    def loadModel(self, model_name):
        # Load model
        model = models.mobilenet_v2(pretrained = True)
           
        #Modify last layer
        n_in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(n_in_features, 2)
        
        # Load saved state dict
        model.load_state_dict(torch.load(model_name))
        
        # Set to eval mode
        model.eval()

        # Load to corresponding device
        model = model.to(self.device)
        return(model)
        
    def predict(self, face, as_label = False):
        face_t = self.transforms(face)
        face_t = face_t.to(self.device)
        face_t = face_t.unsqueeze(0)
        with torch.no_grad():
            output = self.model(face_t)
        
        _, preds = torch.max(output, 1)
        
        
        
        
        if(as_label):
            result = self.classes[preds[0]]
        else:
            result = (preds[0] == 1) # True if real, False if not
            
        # Delete face tensor from device
        del face_t
        del output
        del preds
        torch.cuda.empty_cache()
        return result