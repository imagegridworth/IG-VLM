import sys, os
import time
from io import BytesIO
import argparse
import re
import uuid

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from vision_processor.fps_gridview_processor import *
from pipeline_processor.gpt4_pipeline import *
from evaluation.gpt3_evaluation_utils import *


def infer_and_eval_model(args):
    path_qa = args.path_qa_pair_csv
    path_video = args.path_video
    path_result_dir = args.path_result
    api_key = args.api_key

    system_prompt, user_prompt = get_prompt()
    frame_fixed_number = 6

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
    )
    df_merged, path_df_merged = gpt4vPipeline.do_pipeline()

    print("GPT-4V prediction result : " + path_df_merged)
    print("start gpt3-evaluation")

    gpt3_dir = os.path.join(path_result_dir, "results_gpt3_evaluation")

    df_qa, path_merged = eval_gpt3(df_merged, gpt3_dir, api_key)

    print("Gpt-3-evaluation file : " + path_merged)
    yes_count = df_qa[df_qa["gpt3_pred"] == "yes"].shape[0]
    score = df_qa["gpt3_score"].mean()

    print("Acc : %s" % (str(yes_count / df_qa.shape[0])))
    print("Score : %s" % (str(score)))


def get_prompt():
    system_prompt = ""
    user_prompt = "The provided image arranges keyframes from a video in a grid view. Answer concisely with overall content and context of the video, highlighting any significant events, characters, or objects that appear throughout the frames. Question: %s?"
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
        help="api key for gpt-4v and gpt-3 evaluation",
    )
    args = parser.parse_args()

    infer_and_eval_model(args)
