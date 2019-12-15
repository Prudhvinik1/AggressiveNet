import numpy as np
from keras import backend as K
import os
import sys


def main():
    K.set_image_data_format('channels_last')
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from keras_video_classifier.library.utility.plot_utils import plot_and_save_history
    from keras_video_classifier.library.recurrent_networks import ResnetLSTMVideoClassifier
    from keras_video_classifier.library.utility.ucf.UCF101_loader import load_ucf
    #from keras_video_classifier.library.utility.crime.UCF_Crime_loader import load_ucf
    
    data_set_name = 'UCF-Anomaly-Detection-Dataset'
    input_dir_path = os.path.join(os.path.dirname(__file__), '/content/drive/My Drive')
    output_dir_path = os.path.join(os.path.dirname(__file__), '/content/drive/My Drive/models', data_set_name)
    report_dir_path = os.path.join(os.path.dirname(__file__), '/content/drive/My Drive/reports', data_set_name)

    np.random.seed(42)

    load_ucf(input_dir_path)

    classifier = ResnetLSTMVideoClassifier()

    history = classifier.fit(data_dir_path=input_dir_path, model_dir_path=output_dir_path, resnet_include_top=False, data_set_name=data_set_name, test_size=0.1)

    plot_and_save_history(history, ResnetLSTMVideoClassifier.model_name,
                          report_dir_path + '/' + ResnetLSTMVideoClassifier.model_name + '-hi-dim-history.png')
   
    plot_history_2win(history, ResnetLSTMVideoClassifier.model_name,report_dir_path + '/' + ResnetLSTMVideoClassifier.model_name + '-hi-dim-history2win.png')

if __name__ == '__main__':
    main()
