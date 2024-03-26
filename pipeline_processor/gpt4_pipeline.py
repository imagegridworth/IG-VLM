"""
 Copyright (c) 2024, Deep Representation Learning Research Group, Seoul National University.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import uuid
import pandas as pd
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model_processor.gpt4_model_processor import *
from vision_processor.fps_gridview_processor import *
from pipeline_processor.record import *


class Gpt4Pipeline:
    def __init__(self, path_qa, path_video_file_format, dir="./gpt4_pipeline_result/"):
        self.path_qa = path_qa
        self.path_result = dir
        self.path_video_file_format = path_video_file_format
        self.error_video_name = []
        self.make_video_file_list()

    def make_video_file_list(self):
        self._load_qa_file()
        self.df_qa["path_video"] = self.df_qa.apply(
            lambda x: (self.path_video_file_format % (x["video_name"])), axis=1
        )

    def set_component(
        self,
        api_key,
        system_prompt,
        user_prompt,
        func_user_prompt=lambda prompt, row: prompt % (row["question"]),
        calculate_max_row=lambda x: round(math.sqrt(x)),
        frame_fixed_number=6,
    ):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.func_user_prompt = func_user_prompt

        self.calculate_max_row = calculate_max_row
        self.frame_fixed_number = frame_fixed_number

        self.model = GPT4Inference("gpt4-inference")
        self.model.load_model(api_key=api_key)

        self.fps_data_processor = FpsDataProcessor(
            save_option=SaveOption.BASE64,
            calcualte_max_row=self.calculate_max_row,
            frame_fixed_number=self.frame_fixed_number,
        )

        extra_dir = "ffn=%s/" % (str(self.frame_fixed_number),)

        self._make_directory(extra_dir)

    def do_pipeline(self):

        for idx, row in tqdm(self.df_qa.iterrows()):
            question_id = str(row["question_id"])
            video_path = row["path_video"]
            video_extensions = ["avi", "mp4", "mkv", "webm", "gif"]

            if not os.path.exists(video_path):
                base_video_path, _ = os.path.splitext(video_path)
                for ext in video_extensions:
                    temp_path = f"{base_video_path}.{ext}"
                    if os.path.exists(temp_path):
                        video_path = temp_path
                        break

            if not os.path.exists(self._make_file_path(question_id)):
                try:
                    image_data = self.fps_data_processor.process([video_path])

                    answer = self.model.infer_and_save(
                        system_prompt=self.system_prompt,
                        user_prompt=self.func_user_prompt(self.user_prompt, row),
                        base64_img=image_data,
                    )

                    if -1 != answer:
                        self.write_result_file(question_id, answer)
                    else:
                        self.error_video_name.append(video_path)
                except Exception as e:
                    print(e)
                    print(video_path)
                    continue
        return self.merge_qa_and_answer()

    def write_result_file(self, question_id, answer, extension=".txt"):
        file_path = self._make_file_path(question_id, extension)
        with open(file_path, "w") as file:
            file.write(answer)

    def _make_file_path(self, question_id, extension=".txt"):
        return os.path.join(self.path_result, question_id + extension)

    def _load_qa_file(self):
        self.df_qa = pd.read_csv(self.path_qa, index_col=0)

    def _make_directory(self, extra_dir):
        self.path_result = self.path_result + extra_dir
        os.makedirs(self.path_result, exist_ok=True)

    def merge_qa_and_answer(self):
        self.df_qa["pred"] = None
        if "answer_string" in self.df_qa.columns:
            # in case of NextQA
            self.df_qa = self.df_qa.drop("answer", axis=1).rename(
                {"answer_string": "answer"}, axis=1
            )

        for idx, row in self.df_qa.iterrows():
            question_id = str(row["question_id"])
            file_path = self._make_file_path(
                question_id,
            )

            if os.path.exists(file_path):
                with open(file_path, "r") as file:
                    file_contents = file.read()
                self.df_qa.loc[idx, "pred"] = file_contents

        path_merged = os.path.join(self.path_result, "result.csv")
        self.df_qa.to_csv(path_merged)
        return self.df_qa, path_merged
