"""
 Copyright (c) 2024, Deep Representation Learning Research Group, Seoul National University.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import sys, os
import time
from io import BytesIO
import argparse
import re
import uuid

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from vision_processor.fps_gridview_processor import *

from pipeline_processor.llava_pipeline import *
from evaluation.direct_answer_eval import *


def infer_and_eval_model(args):
    path_qa = args.path_qa_pair_csv
    path_video = args.path_video
    path_result_dir = args.path_result
    llm_size = args.llm_size

    model_name, user_prompt = get_llava_and_prompt(llm_size)
    frame_fixed_number = 6

    # In case of NExT-QA, TVQA, IntentQA and EgoSchema, user the following codes.
    func_user_prompt = lambda prompt, row: prompt % (
        row["question"],
        row["a0"],
        row["a1"],
        row["a2"],
        row["a3"],
        row["a4"],
    )

    # In case of STAR benchamrk, use the following codes and select prompt according to llm size.
    """
    func_user_prompt = lambda prompt, row: prompt % (
        row["question"],
        row["a0"],
        row["a1"],
        row["a2"],
        row["a3"]
    )
    # 7b
    user_prompt = "Select correct option to answer the question. USER: <image>\nThe provided image arranges keyframes from a video in a grid view. Question: %s A:%s. B:%s. C:%s. D:%s. \nSelect the correct answer from the options. \nASSISTANT:\nAnswer:"
    # 13b
    user_prompt = "USER: <image>\nThe provided image arranges keyframes from a video in a grid view. Question: %s?\n A:%s. B:%s. C:%s. D:%s. \n Select the correct answer from the options(A,B,C,D). \nASSISTANT: \nAnswer:"
    # 34b 
    user_prompt = "<|im_start|>system\n Select correct option to answer the question.<|im_end|>\n<|im_start|>user\n <image>\n Question: %s? A:%s. B:%s. C:%s. D:%s. Select the correct answer from the options. <|im_end|>\n<|im_start|>assistant\nAnswer:"
    """

    print("loading [%s]" % (model_name))

    llavaPipeline = LlavaPipeline(
        model_name,
        path_qa,
        path_video,
        dir=path_result_dir,
    )
    llavaPipeline.set_component(
        user_prompt,
        frame_fixed_number=frame_fixed_number,
        func_user_prompt=func_user_prompt,
    )
    df_merged, path_df_merged = llavaPipeline.do_pipeline()

    print("llava prediction result : " + path_df_merged)
    print("start multiple-choice evaluation")

    eval_multiple_choice(df_merged)


def get_llava_and_prompt(llm_size):
    if llm_size in ["7b"]:
        prompt = "Select correct option to answer the question. USER: <image>\nThe provided image arranges keyframes from a video in a grid view. Question: %s A:%s. B:%s. C:%s. D:%s. E:%s. \nSelect the correct answer from the options. \nASSISTANT:\nAnswer:"
        model_name = "llava-v1.6-vicuna-%s" % (llm_size)
    elif llm_size in ["13b"]:
        prompt = "USER: <image>\nThe provided image arranges keyframes from a video in a grid view. Question: %s?\n A:%s. B:%s. C:%s. D:%s. E:%s. \n Select the correct answer from the options(A,B,C,D,E). \nASSISTANT: \nAnswer:"
        model_name = "llava-v1.6-vicuna-%s" % (llm_size)
    else:
        prompt = "<|im_start|>system\n Select correct option to answer the question.<|im_end|>\n<|im_start|>user\n <image>\n Question: %s? A:%s. B:%s. C:%s. D:%s. E: %s. Select the correct answer from the options. <|im_end|>\n<|im_start|>assistant\nAnswer:"
        model_name = "llava-v1.6-%s" % (llm_size)
    return model_name, prompt


def validate_llm_size(type_llm_size):
    if type_llm_size not in {"7b", "13b", "34b"}:
        raise argparse.ArgumentTypeError(f"No valid LLM size.")
    return type_llm_size


def validate_video_path(filename):
    pattern = r"\.(avi|mp4|mkv|gif|webm)$"  # %s.avi 또는 %s.mp4 형식을 따르는지 확인하는 정규 표현식
    if not re.search(pattern, filename):
        raise argparse.ArgumentTypeError(
            f"No valid video path. You must include %s and the extension of video file. (e.g., /tmp/%s.mp4)"
        )
    return filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaVA v1.6 with IG-VLM")
    parser.add_argument(
        "--path_qa_pair_csv",
        type=str,
        required=True,
        help="path of question and answer. It should be csv files",
    )
    parser.add_argument(
        "--path_video",
        type=validate_video_path,
        required=True,
        metavar="/tmp/%s.mp4",
        help="path of video files. You must include string format specifier and the extension of video file.",
    )
    parser.add_argument(
        "--path_result", type=str, required=True, help="path of output directory"
    )

    parser.add_argument(
        "--llm_size",
        type=validate_llm_size,
        default="7b",
        help="You can choose llm size of LLaVA. 7b | 13b | 34b",
    )

    args = parser.parse_args()

    infer_and_eval_model(args)
