from histogram_of_oriented_gradient import HistogramOfOrientedGradient
from convolution import Convolution

import numpy as np
import cv2

class FeatureExtractor:
    def __init__(self):
        self.conv = Convolution()
        self.hog = HistogramOfOrientedGradient()

    def apply_extract(self, image):
        x = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)[1]
        x2_1 = self.conv.left_to_right_filtering(x)
        x2_2 = self.conv.right_to_left_filtering(x)
        x2 = np.sqrt(x2_1 ** 2 + x2_2 ** 2)
        x3 = self.conv.laplacian_filtering(x2)
        x4 = self.hog.apply_hog(x3)

        return x, x2, x3, x4
    
