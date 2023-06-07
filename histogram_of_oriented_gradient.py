from skimage.feature import hog
from skimage import exposure

class HistogramOfOrientedGradient:
    def __init__(self, cell_blocks=4, block_size=2):
        self.cell_blocks = cell_blocks
        self.block_size = block_size

    def apply_hog(self, image):
        _, hog_image = hog(
            image, 
            orientations=image.shape[0], 
            pixels_per_cell=(self.cell_blocks, self.cell_blocks), 
            cells_per_block=(self.block_size, self.block_size), 
            visualize=True)
        
        hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        return hog_image
