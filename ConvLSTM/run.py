import os
from itertools import chain

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from keras.optimizers import RMSprop, Adam

import pandas as pd
from keras.applications import Xception, ResNet50, InceptionV3, MobileNet, VGG19, DenseNet121, InceptionResNetV2, VGG16
from keras.layers import LSTM, ConvLSTM2D
import BuildModel_basic
import DatasetBuilder

from numpy.random import seed, shuffle

from tensorflow import set_random_seed
from collections import defaultdict
import plotHistory

class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.test_loss = []
        self.test_acc = []

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, batch_size=2, verbose=0)
        self.test_loss.append(loss)
        self.test_acc.append(acc)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


def train_eval_network(dataset_name, train_gen, validate_gen, test_x, test_y, seq_len, epochs, batch_size,
                       batch_epoch_ratio, initial_weights, size, cnn_arch, learning_rate,
                       optimizer, cnn_train_type, pre_weights, lstm_conf, len_train, len_valid, dropout, classes,
                       patience_es=15, patience_lr=5):
    """the function builds, compiles, fits and evaluates a certain architechtures on a dataset"""
    set_random_seed(2)
    seed(1)
    print("Experiment Runnning with CNN:",str(cnn_arch))
    result = dict(dataset=dataset_name, cnn_train=cnn_train_type,
                  cnn=cnn_arch.__name__, lstm=lstm_conf[0].__name__, epochs=epochs,
                  learning_rate=learning_rate, batch_size=batch_size, dropout=dropout,
                  optimizer=optimizer[0].__name__, initial_weights=initial_weights, seq_len=seq_len)
    print("run experimnt " + str(result))
    model = BuildModel_basic.build(size=size, seq_len=seq_len, learning_rate=learning_rate,
                                   optimizer_class=optimizer, initial_weights=initial_weights,
                                   cnn_class=cnn_arch, pre_weights=pre_weights, lstm_conf=lstm_conf,
                                   cnn_train_type=cnn_train_type, dropout=dropout, classes=classes)

    # the network is trained on data generators and apply the callacks when the validation loss is not improving:
    # 1. early stop to training after n iteration
    # 2. reducing the learning rate after k iteration where k < n
    test_history = TestCallback((test_x, test_y))
    history = model.fit_generator(
        steps_per_epoch=int(float(len_train) / float(batch_size * batch_epoch_ratio)),
        generator=train_gen,
        epochs=epochs,
        validation_data=validate_gen,
        validation_steps=int(float(len_valid) / float(batch_size)),
        callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.001, patience=patience_es, ),
                   ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience_lr, min_lr=1e-8, verbose=1),
                   test_history
                   ]
    )
    history_to_save = history.history
    history_to_save['test accuracy'] = test_history.test_acc
    history_to_save['test loss'] = test_history.test_loss

#get results
    model_name = ""
    for k, v in result.items():
        model_name = model_name + "_" + str(k) + "-" + str(v).replace(".", "d")
    model_path = os.path.join(res_path, model_name)
    pd.DataFrame(history_to_save).to_csv(model_path + "_train_results.csv")
    result['validation loss'] = min(history.history['val_loss'])
    result['validation accuracy'] = max(history.history['val_acc'])
    result['last validation loss'] = history.history['val_loss'][-1]
    result['last validation accuracy'] = history.history['val_acc'][-1]

    result['train accuracy'] = max(history.history['acc'])
    result['train loss'] = min(history.history['loss'])
    result['last train accuracy'] = history.history['acc'][-1]
    result['last train loss'] = history.history['loss'][-1]

    result['test accuracy'] = max(test_history.test_acc)
    result['test loss'] = min(test_history.test_loss)
    result['last test accuracy'] = test_history.test_acc[-1]
    result['last test loss'] = test_history.test_loss[-1]

    result['final lr'] = history.history['lr'][-1]
    result['total epochs'] = len(history.history['lr'])
    return result

#get train and validation data 
def get_generators(dataset_name, dataset_videos, datasets_frames, fix_len, figure_size, force, classes=1, use_aug=False,
                   use_crop=True):
    train_path, valid_path, test_path, \
    train_y, valid_y, test_y, \
    avg_length = DatasetBuilder.createDataset(dataset_videos, datasets_frames, fix_len,classes, force=force)

    if fix_len is not None:
        avg_length = fix_len
   

    len_train, len_valid = len(train_path), len(valid_path)
    train_gen = DatasetBuilder.data_generator(train_path, train_y, batch_size, figure_size, avg_length, use_aug=use_aug,
                                              use_crop=use_crop, classes=classes)
    validate_gen = DatasetBuilder.data_generator(valid_path, valid_y, batch_size, figure_size, avg_length,
                                                 use_aug=False, use_crop=False, classes=classes)
    test_x, test_y = DatasetBuilder.get_sequences(test_path, test_y, figure_size, avg_length, classes=classes)

    return train_gen, validate_gen, test_x, test_y, avg_length, len_train, len_valid



# select which dataset to work on
datasets_videos = dict(
    #hocky=dict(hocky="/content/drive/My Drive/ConvLSTM_violence/data/raw_videos/hocky"),
    #violentflow=dict(violentflow="/content/drive/My Drive/ConvLSTM_violence/data/raw_videos/violentflow"),
    #movies=dict(movies="/content/drive/My Drive/ConvLSTM_violence/data/raw_videos/movies"),
    crimes=dict(crimes="/content/drive/My Drive/ConvLSTM_violence/data/raw_videos/crimes")
)

#set path of extracted frames of the dataset 
datasets_frames = "/content/drive/My Drive/ConvLSTM_violence/data/raw_frames"
#set path of results of the experiment
res_path = "/content/drive/My Drive/ConvLSTM_violence/results/crimes_multi_results"
#set input image size to be resized
figure_size = 244
#train-test split ratio
#split_ratio = 0.1
#set batch-size for training
batch_size = 32
# batch_epoch_ratio = 0.5 #double the size because we use augmentation
#number of frames to be extracted per video
fix_len = 30
initial_weights = 'glorot_uniform'
weights = 'imagenet'
#set force to True if frames need to extracted afresh
force = False
#initializing convlstm model
lstm = (ConvLSTM2D, dict(filters=256, kernel_size=(3, 3), padding='same', return_sequences=False))
#number of classes of the dataset
classes = 14

results = []
#select which arch of cnn to use
cnn_arch = ResNet50
learning_rate = 0.0001
optimizer = (Adam, {})
#select static if pretrained weights are to be used for training, else select retrain
cnn_train_type = 'retrain'
dropout = 0.3
#set use_aug to True if augmentation methods are to be used
use_aug =  False


# apply selected architechture on selected dataset/datasets
for dataset_name, dataset_videos in datasets_videos.items():
    train_gen, validate_gen, test_x, test_y, seq_len, len_train, len_valid = get_generators(dataset_name,
                                                                                            dataset_videos,
                                                                                            datasets_frames, fix_len,
                                                                                            figure_size,
                                                                                            force=force,
                                                                                            classes=classes,
                                                                                            use_aug=use_aug,
                                                                                            use_crop=True)
    result = train_eval_network(epochs=30, dataset_name=dataset_name, train_gen=train_gen, validate_gen=validate_gen,
                                test_x=test_x, test_y=test_y, seq_len=seq_len, batch_size=batch_size,
                                batch_epoch_ratio=0.5, initial_weights=initial_weights, size=figure_size,
                                cnn_arch=cnn_arch, learning_rate=learning_rate,
                                optimizer=optimizer, cnn_train_type=cnn_train_type,
                                pre_weights=weights, lstm_conf=lstm, len_train=len_train, len_valid=len_valid,
                                dropout=dropout, classes=classes)
    plotHistory.plot_and_save_history(result, cnn_arch,res_path + '/' + cnn_arch + dataset_name + epochs + '--history.png')
    results.append(result)
    pd.DataFrame(results).to_csv("/content/drive/My Drive/ConvLSTM_violence/Exp Results/crimes_multi_results_Adam_Resnet50.csv")
    print(result)
pd.DataFrame(results).to_csv("/content/drive/My Drive/ConvLSTM_violence/Exp Results/crimes_multi_results_Adam_Resnet50.csv")
