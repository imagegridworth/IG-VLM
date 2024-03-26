"""
 Copyright (c) 2024, Deep Representation Learning Research Group, Seoul National University.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from abc import ABC, abstractmethod
from enum import Enum


class BaseModelInference(ABC):
    def __init__(self, model_name, local_save_path):
        self.model_name = model_name
        self.local_save_path = local_save_path
        self.error_list = []

    @abstractmethod
    def load_model(self, **kwargs):
        pass

    @abstractmethod
    def inference(self, **kwargs):
        pass

    @abstractmethod
    def extract_answers(self):
        pass

    def save_local_file(self, answer):
        directory = os.path.dirname(self.local_save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(self.local_save_path, "w") as file:
            file.write(answer)

    def infer_and_save(self, **kwargs):
        try:
            self.inference(**kwargs)
            answer = self.extract_answers()
            return answer
        except Exception as e:
            self.error_list.append(e)
            print(e)
            return -1
