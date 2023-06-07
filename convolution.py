import numpy as np
import cv2

class Convolution:
    def apply_filter(self, filter, image):
        rows, cols = image.shape
        dst = np.zeros((rows, cols), np.float32)
        ycenter, xcenter = 1, 1

        for i in range(ycenter, rows - ycenter):
            for j in range(xcenter, cols - xcenter):
                y1, y2 = i - ycenter, i + ycenter + 1
                x1, x2 = j - xcenter, j + xcenter + 1
                roi = image[y1:y2, x1:x2].astype('float32')
                dst[i, j] = np.sum(roi * filter)

        return dst.astype('uint8')
    
    def laplacian_filtering(self, image):
        filter = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0],
        ], np.float32)
        dst = self.apply_filter(filter, image)
        dst = cv2.convertScaleAbs(dst)

        return dst
    
    def left_to_right_filtering(self, image):
        filter = np.array([
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, -1],
        ], np.float32)
        dst = self.apply_filter(filter, image)

        return dst
    
    def right_to_left_filtering(self, image):
        filter = np.array([
            [0, 0, 1],
            [0, 0, 0],
            [-1, 0, 0],
        ], np.float32)
        dst = self.apply_filter(filter, image)

        return dst