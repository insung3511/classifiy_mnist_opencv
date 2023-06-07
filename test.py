from feature_extraction import FeatureExtractor

import numpy as np
import pickle
import gzip

def mean_squared_error(x, y):
    return np.sum((x - y) ** 2) / x.size


class Test:
    def __init__(self):
        self.trained_data = dict()
        with gzip.open("./result/trained_threshold.pkl", "rb") as f:
            self.trained_data = pickle.load(f)

    def test(self, test_img):
        extractor = FeatureExtractor()
        for i in range(10):
            mse = list()

            test_spectrum = extractor.apply_extract(image=test_img)
            for j in range(10):
                mag_mse = mean_squared_error(test_spectrum[-1], self.trained_data[j])
                mse.append(mag_mse)

        print(f"Predict: {np.argmin(mse)}, MSE: {mse[np.argmin(mse)]}")

        return test_spectrum
