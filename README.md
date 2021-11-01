# Antispoofing Detector

## About :clipboard:

This project centres around the problem of verifying whether there are two different people on a given video at once. To do so, the system relies on a video in `.mp4` format as the input, although videos in other extensions such as `.webm` can also be accepted. Nevertheless, being able to use another format that is not `.mp4` is not guaranteed. 

There may appear 0 to N people at the same time, where N is a natural number.




## Input and Output :pencil:
The input is a video where there may or not appear more than one different person at the same time. The output is a file that, for each videos, establishes a label of 0 or 1. 

- 1: There are at least 2 different real people at once.
- 0: Otherwise. (Only spoofing, or someone real alone, etc.)



## Installation :bulb:

Remember to create a virtual environment for this project in order to avoid dirtying other work spaces.

e.g.

```bash
$ python -m venv detection_venv
$ source detection_venv/bin/activate
$ python -m pip install -r requirements.txt
```



## Project Structure



## Run :rocket:

Place the video in some location where you are aware of its path (for example,  in `<project_root>/data/test_videos/`) and pass it as an argument to `main.py`.

```bash
$ python src/main.py -i data/test_videos -o results.txt
```

where 

- -i specifies the directory where the test videos are located.
- -d specifies the input directory where the data is located. You should use the same structure as the one provided in `data/dataset_training`.

You should see an output similar to this one:

```bash
[*] Processing data/test_videos/test_video_1.mp4
Video length: 253
100%|██████████████████████████████████| 253/253 [00:03<00:00, 63.27it/s]
[*] Video 'data/test_videos/test_video_1.mp4' has 1 real people at once.
```



## Training your own model :hammer:

If you want to train the Antispoofer model using your own data, you first have to create or gather a dataset of real and fake faces and put it in a folder that follows the same structure as `data/dataset_training`.

#### Creating your own dataset for the Antispoofer :performing_arts:

If you do not have such dataset, you can use videos of people you are sure are real (for example, take a video in selfie mode) and videos where the faces shown to the camera are fake (for example, videos where you are showing photographs or people in magazines.. **Without you appearing on camera!**). Once you have the videos, you can extract the faces.

You can use a script prepared in this project to do this exact thing in the `src` folder called  `create_dataset.py`, which reads a video and extracts the faces detected in the specified folder.

To use it you should run it once per each video, specifying the output directory. That is, for a video for which you know exactly whether the faces appearing are real or fake, place them on the training/validation directory labeled as such, in the subdirectory of real/fake.

The structure of `data/dataset_training` is, and it's recommended that you use it:

```bash
data/dataset_training
                    ├── train
                    │   ├── fake
                    │   └── real
                    └── val
                        ├── fake
                        └── real
```



For example, assume you have 4 videos, placed in `data/videos_training`:

```
data/videos_training
                    ├── fake_1.mp4
                    ├── fake_2.mp4
                    ├── real_1.mp4
                    └── real_2.mp4
```

Where those whose name starts with *fake* ensure that each face on it is fake and those starting with *real* ensure faces appearing are genuine.

Then, you can create your dataset by running:

```bash
$ python src/create_dataset.py -i data/videos_training/fake_1.mp4 -o data/dataset_training/train/fake
$ python src/create_dataset.py -i data/videos_training/fake_2.mp4 -o data/dataset_training/val/fake
$ python src/create_dataset.py -i data/videos_training/real_1.mp4 -o data/dataset_training/train/real
$ python src/create_dataset.py -i data/videos_training/real_2.mp4 -o data/dataset_training/val/real
```



:warning: **Warning:** The faces in the training videos, be them real or fake, should be of people of different skin tones and ethnicities, so that the model is not biased towards a specific face type.

Also, as an advice, you should aim for **equality** in the **number of faces for each class**, as well as having more data in the training folder than in the validation one.



#### Training the Antispoofer :wrench: :nut_and_bolt:

Then, you can train your model by running:

```bash
$ python src/train_antispoofer.py -e 10 -d data/dataset_training -o my_model.pt
```

where 

- -e specifies the number of epochs to train the model for
- -d specifies the input directory where the data is located. You should use the same structure as the one provided in `data/dataset_training`.





# TODO :construction:

Add webcam input to check whether what your webcam captures is real or fake in real time.