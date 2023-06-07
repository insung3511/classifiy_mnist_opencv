from tqdm import tqdm

import numpy as np
import cv2
import os


def is_file_available(file_path: str):
    return os.path.isdir(file_path)


def get_file_list(file_path: str):
    """Get files lists by os module.

    :param file_path: str
    :return:
        File lists with params file_path + file_name
    """
    if is_file_available(file_path):
        file_path_list = os.listdir(file_path)
        return [file_path + file for file in file_path_list]

    else:
        Exception(f"{file_path} is not a directory. Please check again")


def get_image(image_path: str):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return 255 - image


class DataLoader:
    def __init__(self, data_path: str, labels: list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
        """Load datasets from params data_path. data_path have to be String. labels have to be List.
        Example)
            # Load datafrom ./data/mnist/, load 0, 1, 2 labeled images.
            dataloader = DataLoader('./data/mnist/', [0, 1, 2])

        Args:
            :param data_path:
            :param labels:
        """
        self.data_path = data_path
        self.labels = labels

    def get_total_image(self):
        labels_data = dict()

        each_label_path = [self.data_path + str(path) for path in self.labels]
        for idx, labels_path in enumerate(each_label_path):
            label_number = self.labels[idx]
            labels_data[label_number] = []

            for file in tqdm(get_file_list(labels_path + '/'), desc=f'Loading images... Label Number {label_number}'):
                labels_data[label_number].append(get_image(file))

        return labels_data

    def get_data_shape(self, data: dict):
        for key in data.keys():
            print(f"{key}th data shape : {np.shape(data[key])}")
