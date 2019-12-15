import urllib.request
import os
import sys
import patoolib

URL_LINK = 'http://crcv.ucf.edu/data/UCF101/UCF101.rar'

def scan_ucf(data_dir_path, limit):
    input_data_dir_path = data_dir_path + '/UCF-Anomaly-Detection-Dataset'

    result = dict()

    dir_count = 0
    for f in os.listdir(input_data_dir_path):
        file_path = input_data_dir_path + os.path.sep + f
        if not os.path.isfile(file_path):
            dir_count += 1
            for ff in os.listdir(file_path):
                video_file_path = file_path + os.path.sep + ff
                result[video_file_path] = f
        if dir_count == limit:
            break
    return result


def scan_ucf_with_labels(data_dir_path, labels, hard_split = False):
    input_data_dir_path = data_dir_path + '/UCF-Anomaly-Detection-Dataset'
    if (hard_split == True):
        input_data_dir_path = data_dir_path + '/UCF-101_Train'

    result = dict()

    dir_count = 0
    for label in labels:
        file_path = input_data_dir_path + os.path.sep + label
        if not os.path.isfile(file_path):
            dir_count += 1
            for ff in os.listdir(file_path):
                video_file_path = file_path + os.path.sep + ff
                result[video_file_path] = label
    return result



def load_ucf(data_dir_path):
    UFC101_data_dir_path = data_dir_path + "/UCF-Anomaly-Detection-Dataset"
    if not os.path.exists(UFC101_data_dir_path):
        print("data dir not set properly")


def main():
    data_dir_path = '/content/drive/My Drive'
    load_ucf(data_dir_path)


if __name__ == '__main__':
    main()
