import os
import cv2
from abc import ABC, abstractmethod
import numpy as np
from enum import Enum
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from .decorator_processor import *
from .video_validation import VideoValidator


class SaveOption(Enum):
    BYTES = "bytes"
    FILE = "file"
    BASE64 = "base64"
    NUMPY = "numpy"
    IMAGE = "image"


class BaseFrameExtractor(ABC):
    def __init__(self, video_path_with_list):
        self.video_path = os.path.join(*video_path_with_list)
        self._check_video_valid()

    @abstractmethod
    def extract_frames(self, **kwargs):
        pass

    def _extract_frame_on_option(self, **kwargs):
        data = self.extract_frames(**kwargs)
        return data

    def save_data_based_on_option(self, option, filename=None, quality=95, **kwargs):
        data = self._extract_frame_on_option(**kwargs)
        if option == SaveOption.BYTES:
            return self._save_data(data)
        elif option == SaveOption.FILE:
            return self._save_data_to_file(data, filename, quality)
        elif option == SaveOption.BASE64:
            return self._save_data_to_base64(data)
        elif option == SaveOption.NUMPY:
            return np.array(data)
        else:
            raise ValueError("Invalid option: {}".format(option))

    def _check_video_valid(self):
        VideoValidator(self.video_path)

    @save_to_bytes
    def _save_data(self, data):
        return data

    @save_to_file
    def _save_data_to_file(self, data, filename, quality):
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        cv2.imwrite(filename, data, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        pass

    @save_to_base64
    def _save_data_to_base64(self, data):
        return data
