import os
import cv2
from abc import ABC, abstractmethod
import numpy as np
from enum import Enum

from PIL import Image

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from .decorator_processor import *
from .base_frame_extractor import *


class BasePostProcessor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def post_process(self, **kwargs):
        pass

    def _post_processo_on_option(self, *args, **kwargs):
        data = self.post_process(*args, **kwargs)
        return data

    def post_process_based_on_options(self, option, *args, **kwargs):
        self.option = option
        data = self._post_processo_on_option(*args, **kwargs)
        if option == SaveOption.BASE64:
            return self._save_data_to_base64(data, quality=self.quality)
        elif option == SaveOption.IMAGE:
            return Image.fromarray(data)
        else:
            raise ValueError("Invalid option: {}".format(option))

    @save_to_base64
    def _save_data_to_base64(self, data, quality=95):
        return data

    @save_to_file
    def _save_data_to_file(self, data, filename=None, quality=95):
        cv2.imwrite(filename, data, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        pass
