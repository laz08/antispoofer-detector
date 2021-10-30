#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the AntiSpoofer model.
"""
# Misc
import os
import time
import copy
import argparse

# Import torch packages
import torch
from torchvision import models, datasets
import torch.nn  as nn
from torchvision import transforms as T
from torch import optim


# Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", type=str, required=True,
	help="Data directory for training")
ap.add_argument("-o", "--output", type=str, default='model_mobilenet_v2.pt',
	help="Name of the model once trained")
ap.add_argument("-e", "--epochs", type=str, default=10,
	help="Epochs for which to train the model")

args = vars(ap.parse_args())

# Load where the dataset can be found
data_dir = args['directory']
print("[INFO] Training data from {}".format(data_dir))

# Load pretrained model.
def getModel(n_classes = 2):
    # Load model
    model = models.mobilenet_v2(pretrained = True)
       
    #Modify last layer
    n_in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(n_in_features, n_classes)
    
    # model = models.resnet50(pretrained = True)
    # # change the number of classes in the last layer
    # n_inputs = model.fc.in_features 
    # model.fc = nn.Linear(n_inputs, 2)
    return model
 
    
# Transformations
mean_vec = [0.485, 0.456, 0.406]
std_vec = [0.229, 0.224, 0.225]

transforms_dict = {
    'train': T.Compose([
                    T.RandomResizedCrop(size=256),
                    T.RandomRotation(degrees=15),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean_vec, std_vec)
                    ]), 
    'val': T.Compose([
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean_vec, std_vec)
                    ])
}
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] Device to be used during training: {}".format(device))

print("[INFO] Loading model")
model = getModel()
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]


# Load datasets
print("[INFO] Loading datasets")
datasets_dict = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                         transforms_dict[x])
                  for x in ['train', 'val']}

print("[INFO] Creating DataLoaders")
dataloaders = {x: torch.utils.data.DataLoader(datasets_dict[x], 
                                              batch_size=4,
                                              shuffle=True, 
                                              num_workers=4)
              for x in ['train', 'val']}

dataset_sizes = {x: len(datasets_dict[x]) for x in ['train', 'val']}
classes = datasets_dict['train'].classes


def train_model(model, criterion, optimizer, scheduler, epochs=10):
    
    start = time.time()

    # To save best model achieved
    best_model = copy.deepcopy(model.state_dict())
    
    best_acc = 0.0
    best_loss = None
    
    for e in range(epochs):
        print('Epoch: {}/{}'.format(e+1, epochs))
        print('-' * 10)

        # First train, then validate.
        for action in ['train', 'val']:
            if action == 'train':
                scheduler.step() # Step the scheduler
                model.train()  # Put the model in trianing mode
            else:
                model.eval()   # Set model to evaluate mode

            curr_loss = 0.0
            tp_tn = 0 # True labels accomplished

            for inputs, labels in dataloaders[action]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(action == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if action == 'train':
                        loss.backward()
                        optimizer.step()

                # Save Loss and TP + TN
                curr_loss += loss.item() * inputs.size(0)
                tp_tn += torch.sum(preds == labels.data)

            epoch_loss = curr_loss / dataset_sizes[action]
            epoch_acc = tp_tn.double() / dataset_sizes[action]

            print('During -{}-'.format(action),
                  '\n    [-] Loss: {:.4f}'.format(epoch_loss),
                  '\n    [-] Acc: {:.4f}'.format(epoch_acc))

            
            if(action == 'val' and (best_loss is None 
                                    or (best_loss is not None and epoch_loss < best_loss))):
                best_loss = epoch_loss
                best_model = copy.deepcopy(model.state_dict())


    elapsed_time = time.time() - start
    print('Elapsed training time: {:.0f}m {:.0f}s'.format(elapsed_time // 60, 
                                                        elapsed_time % 60))
    print('Lowest Loss accomplished on val: {:4f}'.format(best_loss))

    # Now we'll load in the best model weights and return it
    model.load_state_dict(best_model)
    return model



#optimizer = torch.optim.SGD(params, lr=0.0005, momentum=0.9, weight_decay=0.0005) # 0.00005
#optimizer = torch.optim.AdamW(params, lr=0.000005, betas=(0.9, 0.999), weight_decay=0.01)
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

# Decay Learning Rate 0.1 every 7 steps
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
criterion = nn.CrossEntropyLoss()

epochs = int(args['epochs'])

print("[INFO] Starting to train model...")
print("    [-] Epochs: {}".format(epochs))
print("    [-] Optimizer: {}".format(optimizer))
print("    [-] Scheduler: {}".format(scheduler))
model = train_model(model, criterion, optimizer, scheduler, epochs=epochs)

# Save model
name_model = args['output']#'model_mobilenet_v2.pt'

print("[INFO] Saving model as {}".format(name_model))
torch.save(model.state_dict(), name_model)
print("[OK] Done.")
 
