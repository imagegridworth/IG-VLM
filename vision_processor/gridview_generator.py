import cv2
import glob
import numpy as np
import math
import os
import sys
import random

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from pipeline_processor.record import *
from .base_post_processor import *
from .fps_extractor import *


class GridViewCreator(BasePostProcessor):
    def __init__(self, func_max_per_row, quality=95):
        self.func_max_per_row = func_max_per_row
        self.quality = quality

    def post_process(self, *args, **kwargs):
        return self.create_grid_view_as_array_from_image_array(*args)

    def _extract_arguments(self, **kwargs):
        try:
            self.images = kwargs.get("images", [])
        except Exception as e:
            raise Exception(e)

    def create_grid_view_as_array_from_image_array(self, images):
        images = images
        self.max_images_per_row = self.func_max_per_row(len(images))

        min_width = min(img.shape[1] for img in images)
        min_height = min(img.shape[0] for img in images)
        resized_images = [cv2.resize(img, (min_width, min_height)) for img in images]

        while len(resized_images) % self.max_images_per_row != 0:
            resized_images.append(
                np.ones((min_height, min_width, 3), dtype=np.uint8) * 255
            )

        image_rows = [
            resized_images[i : i + self.max_images_per_row]
            for i in range(0, len(resized_images), self.max_images_per_row)
        ]
        concatenated_rows = [np.hstack(row) for row in image_rows]

        grid_image = np.vstack(concatenated_rows)
        image_array_rgb = cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB)
        if self.option == SaveOption.BASE64:
            return image_array_rgb
        else:
            return grid_image

    def _get_frame_number(self, file_path):
        numbers = file_path.split("/")[-1].split(".")[0]
        return int(numbers) if numbers else -1
