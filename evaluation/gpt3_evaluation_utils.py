import math
import os
import glob
from tqdm import tqdm
import openai
import pandas as pd

from pipeline_processor.record import *


def eval_gpt3(df_merged, path_result, api_key, gpt_eval_type=EvaluationType.DEFAULT):

    os.makedirs(path_result, exist_ok=True)

    for idx, row in df_merged.iterrows():
        process_gpt3_evaluation_v2(
            row, path_result, api_key, gpt_eval_type=gpt_eval_type
        )

    result_path = os.path.join(path_result, "result.csv")

    if not os.path.exists(result_path):
        df_qa, path_merged = merge_qa_and_answer(df_merged, path_result)
        return df_qa, path_merged
    else:
        path_merged_already = result_path
        df_already = pd.read_csv(path_merged_already, index_col=0)
        return df_already, path_merged_already


def process_gpt3_evaluation_v2(
    row, path_result, api_key, gpt_eval_type=EvaluationType.DEFAULT
):
    client = openai.OpenAI(
        api_key=api_key,
    )
    file_path_saved = os.path.join(path_result, str(row["question_id"]) + ".txt")
    if not os.path.exists(file_path_saved):
        question = row["question"]
        answer = row["answer"]
        pred = row["pred"]
        message = make_messages(question, answer, pred, gpt_eval_type)
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=message
        )
        # Convert response to a Python dictionary.
        response_message = completion.choices[0].message.content
        with open(file_path_saved, "w") as f:
            f.write(response_message)
    else:
        print("exist")


def merge_qa_and_answer(df_qa, path_result):
    df_qa["gpt3_pred"] = None
    df_qa["gpt3_score"] = None

    for idx, row in df_qa.iterrows():
        file_path_saved = os.path.join(path_result, str(row["question_id"]) + ".txt")

        if os.path.exists(file_path_saved):
            with open(file_path_saved, "r") as file:
                try:
                    file_contents = file.read()
                    if file_contents.endswith("."):
                        file_contents = file_contents[:-1]
                    content_dict = eval(file_contents)

                    if "pred" in content_dict:
                        df_qa.loc[idx, "gpt3_pred"] = content_dict["pred"]
                    else:
                        df_qa.loc[idx, "gpt3_pred"] = ""

                    df_qa.loc[idx, "gpt3_score"] = content_dict["score"]
                except Exception as e:
                    print(e)
                    print(file_path_saved)
                    continue
        else:
            print(file_path_saved + " not exist")

    path_merged = os.path.join(path_result, "result.csv")
    df_qa.to_csv(path_merged)

    yes_count = df_qa[df_qa["gpt3_pred"] == "yes"].shape[0]
    print(yes_count / df_qa.shape[0])
    print(df_qa["gpt3_score"].describe())

    return df_qa, path_merged


def make_messages(question, answer, pred, gpt_eval_type):
    if gpt_eval_type == EvaluationType.DEFAULT:  # default
        return [
            {
                "role": "system",
                "content": "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                "------"
                "##INSTRUCTIONS: "
                "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                "- Consider synonyms or paraphrases as valid matches.\n"
                "- Evaluate the correctness of the prediction compared to the answer.",
            },
            {
                "role": "user",
                "content": "Please evaluate the following video-based question-answer pair:\n\n"
                f"Question: {question}\n"
                f"Correct Answer: {answer}\n"
                f"Predicted Answer: {pred}\n\n"
                "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}.",
            },
        ]
    elif gpt_eval_type == EvaluationType.CORRECTNESS:
        return [
            {
                "role": "system",
                "content": "You are an intelligent chatbot designed for evaluating the factual accuracy of generative outputs for video-based question-answer pairs. "
                "Your task is to compare the predicted answer with the correct answer and determine if they are factually consistent. Here's how you can accomplish the task:"
                "------"
                "##INSTRUCTIONS: "
                "- Focus on the factual consistency between the predicted answer and the correct answer. The predicted answer should not contain any misinterpretations or misinformation.\n"
                "- The predicted answer must be factually accurate and align with the video content.\n"
                "- Consider synonyms or paraphrases as valid matches.\n"
                "- Evaluate the factual accuracy of the prediction compared to the answer.",
            },
            {
                "role": "user",
                "content": "Please evaluate the following video-based question-answer pair:\n\n"
                f"Question: {question}\n"
                f"Correct Answer: {answer}\n"
                f"Predicted Answer: {pred}\n\n"
                "Provide your evaluation only as a factual accuracy score where the factual accuracy score is an integer value between 0 and 5, with 5 indicating the highest level of factual consistency. "
                "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the factual accuracy score in INTEGER, not STRING."
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: {''score': 4.8}.",
            },
        ]
    elif gpt_eval_type == EvaluationType.DETAILED_ORIENTATION:
        return [
            {
                "role": "system",
                "content": "You are an intelligent chatbot designed for evaluating the detail orientation of generative outputs for video-based question-answer pairs. "
                "Your task is to compare the predicted answer with the correct answer and determine its level of detail, considering both completeness and specificity. Here's how you can accomplish the task:"
                "------"
                "##INSTRUCTIONS: "
                "- Check if the predicted answer covers all major points from the video. The response should not leave out any key aspects.\n"
                "- Evaluate whether the predicted answer includes specific details rather than just generic points. It should provide comprehensive information that is tied to specific elements of the video.\n"
                "- Consider synonyms or paraphrases as valid matches.\n"
                "- Provide a single evaluation score that reflects the level of detail orientation of the prediction, considering both completeness and specificity.",
            },
            {
                "role": "user",
                "content": "Please evaluate the following video-based question-answer pair:\n\n"
                f"Question: {question}\n"
                f"Correct Answer: {answer}\n"
                f"Predicted Answer: {pred}\n\n"
                "Provide your evaluation only as a detail orientation score where the detail orientation score is an integer value between 0 and 5, with 5 indicating the highest level of detail orientation. "
                "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the detail orientation score in INTEGER, not STRING."
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: {''score': 4.8}.",
            },
        ]
    elif gpt_eval_type == EvaluationType.CONTEXT:
        return [
            {
                "role": "system",
                "content": "You are an intelligent chatbot designed for evaluating the contextual understanding of generative outputs for video-based question-answer pairs. "
                "Your task is to compare the predicted answer with the correct answer and determine if the generated response aligns with the overall context of the video content. Here's how you can accomplish the task:"
                "------"
                "##INSTRUCTIONS: "
                "- Evaluate whether the predicted answer aligns with the overall context of the video content. It should not provide information that is out of context or misaligned.\n"
                "- The predicted answer must capture the main themes and sentiments of the video.\n"
                "- Consider synonyms or paraphrases as valid matches.\n"
                "- Provide your evaluation of the contextual understanding of the prediction compared to the answer.",
            },
            {
                "role": "user",
                "content": "Please evaluate the following video-based question-answer pair:\n\n"
                f"Question: {question}\n"
                f"Correct Answer: {answer}\n"
                f"Predicted Answer: {pred}\n\n"
                "Provide your evaluation only as a contextual understanding score where the contextual understanding score is an integer value between 0 and 5, with 5 indicating the highest level of contextual understanding. "
                "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is contextual understanding score in INTEGER, not STRING."
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: {''score': 4.8}.",
            },
        ]
    elif gpt_eval_type == EvaluationType.TEMPORAL:
        return [
            {
                "role": "system",
                "content": "You are an intelligent chatbot designed for evaluating the temporal understanding of generative outputs for video-based question-answer pairs. "
                "Your task is to compare the predicted answer with the correct answer and determine if they correctly reflect the temporal sequence of events in the video content. Here's how you can accomplish the task:"
                "------"
                "##INSTRUCTIONS: "
                "- Focus on the temporal consistency between the predicted answer and the correct answer. The predicted answer should correctly reflect the sequence of events or details as they are presented in the video content.\n"
                "- Consider synonyms or paraphrases as valid matches, but only if the temporal order is maintained.\n"
                "- Evaluate the temporal accuracy of the prediction compared to the answer.",
            },
            {
                "role": "user",
                "content": "Please evaluate the following video-based question-answer pair:\n\n"
                f"Question: {question}\n"
                f"Correct Answer: {answer}\n"
                f"Predicted Answer: {pred}\n\n"
                "Provide your evaluation only as a temporal accuracy score where the temporal accuracy score is an integer value between 0 and 5, with 5 indicating the highest level of temporal consistency. "
                "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the temporal accuracy score in INTEGER, not STRING."
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: {''score': 4.8}.",
            },
        ]
