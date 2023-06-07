from feature_extraction import FeatureExtractor
from data_load import DataLoader

from datetime import datetime
from tqdm import tqdm

import numpy as np
import pickle
import gzip

np.random.seed(datetime.now().microsecond)

# Add Constant Variables
DATA_PATH = "./data/MNIST - JPG - training/"
LABELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
SPECTRUM_PATH = "./result/"


class Train:
    def train(self):
        # Load Data
        print(f"[INFO] Load MNIST datasets from {DATA_PATH}")
        dataloader = DataLoader(DATA_PATH)
        data = dataloader.get_total_image()
        dataloader.get_data_shape(data)

        print(f"[INFO] Convolution Computing...")
        extractor = FeatureExtractor()

        magnitude_dict = dict()

        for i in range(10):
            select_data_index = np.random.randint(0, len(data[i]), 5000)

            dst = list()
            for j in tqdm(range(len(select_data_index))):
                image_data = data[i][select_data_index[j]]
                image_spectrum = extractor.apply_extract(image_data)
                dst.append(image_spectrum[-1])

            magnitude_dict[i] = np.average(np.array(dst), axis=0)

        with gzip.open(SPECTRUM_PATH + "trained_threshold.pkl", "wb") as f:
            pickle.dump(magnitude_dict, f)
