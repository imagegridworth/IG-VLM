import sys, os
import time
from io import BytesIO
import argparse
import re
import uuid

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from vision_processor.fps_gridview_processor import *

from pipeline_processor.gpt4_pipeline import *
from evaluation.direct_answer_eval import *


def infer_and_eval_model(args):
    path_qa = args.path_qa_pair_csv
    path_video = args.path_video
    path_result_dir = args.path_result
    api_key = args.api_key

    system_prompt, user_prompt = get_prompt()
    frame_fixed_number = 6

    # In case of NExT-QA, TVQA, IntentQA and EgoSchema, These has five options on multiple-choice
    func_user_prompt = lambda prompt, row: prompt % (
        row["question"],
        row["a0"],
        row["a1"],
        row["a2"],
        row["a3"],
        row["a4"],
    )

    # In case of STAR benchamrk, with four options on multiple-choice, please use the following codes and select prompt according to llm size.
    """
    func_user_prompt = lambda prompt, row: prompt % (
        row["question"],
        row["a0"],
        row["a1"],
        row["a2"],
        row["a3"],
    )
    user_prompt = "The provided image arranges key frames from a video in a grid view. They are arranged in chronological order, holding temporal information from the top left to the bottom right. You need to choose one of the following five options to answer the question, '%s?' : A.'%s', B.'%s', C.'%s', D.'%s'. Please provide a single-character answer (A, B, C, or D) to the multiple-choice question, and your answer must be one of the letters (A, B, C or D). Your response must only contain one character without any other string."
    """

    print("loading model")

    gpt4vPipeline = Gpt4Pipeline(
        path_qa,
        path_video,
        dir=path_result_dir,
    )
    gpt4vPipeline.set_component(
        api_key,
        system_prompt,
        user_prompt,
        frame_fixed_number=frame_fixed_number,
        func_user_prompt=func_user_prompt,
    )
    df_merged, path_df_merged = gpt4vPipeline.do_pipeline()

    print("gpt4 prediction result : " + path_df_merged)
    print("start multiple-choice evaluation")

    eval_multiple_choice(df_merged)


def get_prompt():
    system_prompt = ""
    user_prompt = "The provided image arranges key frames from a video in a grid view. They are arranged in chronological order, holding temporal information from the top left to the bottom right. You need to choose one of the following five options to answer the question, '%s?' : A.'%s', B.'%s', C.'%s', D.'%s', E.'%s'. Please provide a single-character answer (A, B, C, D or E) to the multiple-choice question, and your answer must be one of the letters (A, B, C, D or E). Your response must only contain one character without any other string."
    return system_prompt, user_prompt


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
        "--api_key",
        type=str,
        required=True,
        help="api key for gpt-4v",
    )

    args = parser.parse_args()

    infer_and_eval_model(args)
