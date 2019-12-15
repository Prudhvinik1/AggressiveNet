import scipy
import os
import cv2
import pickle
import glob
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
import random
import re


corner_keys = ["Center","Left_up","Left_down","Right_up","Right_down"]


#extract frames from videos of selected datasets
def save_figures_from_video(dataset_video_path, video_filename, suffix,figures_path,skip_frames = 25,apply_norm = True, apply_diff = True,fix_len = None):
    seq_len = 0

    video_figures_path = os.path.join(figures_path ,video_filename)
    if not os.path.exists(video_figures_path):
        os.makedirs(video_figures_path)

    video_file = os.path.join(dataset_video_path, video_filename + suffix)
    label = 0
    print('Extracting frames from video: ', video_file)
    #reading from video
    videoCapture = cv2.VideoCapture(video_file)
    if fix_len is not None:
        vid_len = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_frames = int(float(vid_len)/float(fix_len))
    videoCapture.set(cv2.CAP_PROP_POS_MSEC, (seq_len * skip_frames))
    success, figure_ = videoCapture.read()
    success = True
    files = []
    while success:
        if seq_len % skip_frames == 0:
            if success:
                figure_curr = figure
                image_file = os.path.join(video_figures_path , "frame_%d.jpg" % seq_len)
                files.append(image_file)
                cv2.imwrite(image_file, figure_curr)
        success, figure = videoCapture.read()
        seq_len += 1
    video_images = dict(images_path = video_figures_path, name = video_filename,
                        images_files = files, sequence_length = seq_len, label = label)

    return video_images

#buil dataset for training with selected dataset and labels
def createDataset(datasets_video_path, figure_output_path,fix_len,classes, force = False):
    videos_seq_length = []
    datasets_images = {}
    videos_frames_paths = []
    videos_labels = []
    #Extract images for each video for each dataset
    for dataset_name, dataset_video_path in datasets_video_path.items():
        dataset_figures_path = os.path.join(figure_output_path,dataset_name)
        if not os.path.exists(dataset_figures_path):
            os.makedirs(dataset_figures_path)
        dataset_images = []
        for filename in os.listdir(dataset_video_path):
            if filename.endswith(".avi") or filename.endswith(".mpg") or filename.endswith(".mp4"):
                video_images_file = os.path.join(dataset_figures_path,filename[:-4], 'video_summary.pkl')
                if os.path.isfile(video_images_file) and not force:
                    with open(video_images_file, 'rb') as f:
                        video_images = pickle.load(f)
                else:
                    video_images = save_figures_from_video(dataset_video_path, filename[:-4],filename[-4:], dataset_figures_path, fix_len =fix_len)
                    #label dataset folders as classes
                    if dataset_name == "hocky":
                        if filename.startswith("fi"):
                            video_images['label'] = 1
                    elif dataset_name == "violentflow":
                        if "violence" in filename:
                            video_images['label'] = 1
                    elif dataset_name == "movies":
                        if "fi" in filename:
                            video_images['label'] = 1
                    elif dataset_name == "UniCrimes":
                        if "Rob" in filename:
                            video_images['label'] = 1

                    elif dataset_name == "crimes":
                        if classes == 1:
                            if "Normal" not in filename:
                                video_images['label'] = 1
                        elif classes == 14:
                            if "Abuse" in filename:
                                video_images['label'] = 0
                            elif "Arrest" in filename:
                                video_images['label'] = 1
                            elif "Arson" in filename:
                                video_images['label'] = 2
                            elif "Assault" in filename:
                                video_images['label'] = 3
                            elif "Burglary" in filename:
                                video_images['label'] = 4
                            elif "Explosion" in filename:
                                video_images['label'] = 5
                            elif "Fighting" in filename:
                                video_images['label'] = 6
                            elif "Normal" in filename:
                                video_images['label'] = 7
                            elif "RoadAccidents" in filename:
                                video_images['label'] = 8
                            elif "Robbery" in filename:
                                video_images['label'] = 9
                            elif "Shooting" in filename:
                                video_images['label'] = 10
                            elif "Shoplifting" in filename:
                                video_images['label'] = 11
                            elif "Stealing" in filename:
                                video_images['label'] = 12
                            elif "Vandalism" in filename:
                                video_images['label'] = 13
                            
                            
                            
                    with open(video_images_file, 'wb') as f:
                        pickle.dump(video_images, f, pickle.HIGHEST_PROTOCOL)
                dataset_images.append(video_images)
                videos_seq_length.append(video_images['sequence_length'])
                videos_frames_paths.append(video_images['images_path'])
                videos_labels.append(video_images['label'])
        datasets_images[dataset_name] = dataset_images
    avg_length = int(float(sum(videos_seq_length)) / max(len(videos_seq_length), 1))
    print("Printing Parameters")
    print(videos_frames_paths)
    print(videos_labels)
    
    train_path, test_path, train_y, test_y =  train_test_split(videos_frames_paths,videos_labels, test_size=0.20, random_state=42)

    if apply_aug:
        aug_paths = []
        aug_y = []
        for train_path_, train_y_ in zip(train_path,train_y):
    
            aug_path = generate_augmentations(train_path_,force = False)
            aug_paths.append(aug_path)
            aug_y.append(train_y_)
    
        train_path = train_path + aug_paths
        train_y = train_y + aug_y

    train_path, valid_path, train_y, valid_y = train_test_split(train_path, train_y, test_size=0.20, random_state=42)
    return train_path,valid_path, test_path,\
           train_y, valid_y, test_y,\
           avg_length

#use inbuilt keras preprocessing tools to load images,scale,normalize the input images
def frame_loader(frames,figure_shape,to_norm = True):
    output_frames = []
    for frame in frames:
        image = load_img(frame, target_size=(figure_shape, figure_shape),interpolation='bilinear')
        img_arr = img_to_array(image)
        # Scale
        figure = (img_arr / 255.).astype(np.float32)
        # Normalize
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        figure = (figure - mean) / std
        output_frames.append(figure)
    return output_frames

#get shuffled data for train and test 
def data_generator(data_paths,labels,batch_size,figure_shape,seq_length,use_aug,use_crop,classes = 1):
    while True:
        indexes = np.arange(len(data_paths))
        np.random.shuffle(indexes)
        select_indexes = indexes[:batch_size]
        data_paths_batch = [data_paths[i] for i in select_indexes]
        labels_batch = [labels[i] for i in select_indexes]

        X, y = get_sequences(data_paths_batch,labels_batch,figure_shape,seq_length, classes, use_augmentation = use_aug,use_crop=use_crop)

        yield X, y

#using popular image cropping technique to crop input images
def crop_img(img,figure_shape,percentage=0.8,corner="Left_up"):
    if(corner == None):
        corner = random.choice(corner_keys)

    if corner not in corner_keys:
        raise ValueError(
            'Invalid corner method {} specified. Supported '
            'corners are {}'.format(
                corner,
                ", ".join(corner_keys)))

    resize = int(figure_shape*percentage)

    if(corner =="Left_up"):
        x_start = 0
        x_end = resize
        y_start = 0
        y_end = resize
    if (corner == "Right_down"):
        x_start = figure_shape-resize
        x_end = figure_shape
        y_start = figure_shape-resize
        y_end = figure_shape
    if(corner =="Right_up"):
        x_start = 0
        x_end = resize
        y_start = figure_shape-resize
        y_end = figure_shape
    if (corner == "Left_down"):
        x_start = figure_shape-resize
        x_end = figure_shape
        y_start = 0
        y_end = resize
    if (corner == "Center"):
        half = int(figure_shape*(1-percentage))
        x_start = half
        x_end = figure_shape-half
        y_start = half
        y_end = figure_shape-half

    img = cv2.resize(img[y_start:y_end,x_start:x_end, :], (figure_shape, figure_shape)).astype(np.float32)
    return img

#get train and test datasets with labels after cropping and augmenting them
def get_sequences(data_paths,labels,figure_shape,seq_length,classes=1, use_augmentation = False,use_crop=True):
    X, y = [], []
    seq_len = 0
    for data_path, label in zip(data_paths,labels):
        frames = sorted(glob.glob(os.path.join(data_path, '*jpg')))
        x = frame_loader(frames, figure_shape)
        if use_augmentation:
            rand = scipy.random.random()
            corner=""
            if rand > 0.5:
                if(use_crop):
                    corner=random.choice(corner_keys)
                    x = [crop_img(x_,figure_shape,0.7,corner) for x_ in x]
                x = [frame.transpose(1, 0, 2) for frame in x]
               
        x = [x[i] - x[i+1] for i in range(len(x)-1)]
        X.append(x)
        y.append(label)
    X = pad_sequences(X, maxlen=seq_length, padding='pre', truncating='pre')
    if classes > 1:
        y = to_categorical(y,classes) #one hot encoding
    return np.array(X), np.array(y)

#sort files based on alphanumerical key
def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


#dataset augmentation function, does geometric data augmentation by transposing image frame
def generate_augmentations(data_path,figure_shape = 244, force = False):
    seq_len = 0
    crop_path = data_path + "_crop"
    if not os.path.exists(crop_path) or force:
        frames = natural_sort(glob.glob(os.path.join(data_path, '*jpg')))
        frames_arr = frame_loader(frames, figure_shape,to_norm = False)
        print("augmenting " + data_path)
        os.makedirs(crop_path)
        for frame in frames_arr:
            #transpose
            img_transpose = frame.transpose(1,0,2)
            data_path_aug = os.path.join(crop_path,"frame_%d.jpg" % seq_len)
            cv2.imwrite(data_path_aug, img_transpose)
            seq_len += 1
    return crop_path

