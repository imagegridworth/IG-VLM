"""
 Copyright (c) 2024, Deep Representation Learning Research Group, Seoul National University.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import math
import os
import glob
from tqdm import tqdm
import openai
import pandas as pd


def eval_gpt3_consistency(df_merged1, df_merged2, path_result, api_key):

    os.makedirs(path_result, exist_ok=True)

    for (idx1, row1), (idx2, row2) in zip(df_merged1.iterrows(), df_merged2.iterrows()):
        process_gpt3_evaluation_consistency(row1, row2, path_result, api_key)

    result_path = os.path.join(path_result, "result.csv")

    if not os.path.exists(result_path):
        df_qa, path_merged = merge_qa_and_answer_consistency(
            df_merged1, df_merged2, path_result
        )
        return df_qa, path_merged
    else:
        path_merged_already = result_path
        df_already = pd.read_csv(path_merged_already, index_col=0)
        return df_already, path_merged_already


def process_gpt3_evaluation_consistency(row1, row2, path_result, api_key):
    client = openai.OpenAI(api_key=api_key)
    file_path_saved = os.path.join(path_result, str(row1["question_id"]) + ".txt")
    if not os.path.exists(file_path_saved):
        question1 = row1["question"]
        question2 = row2["question"]
        answer = row1["answer"]
        pred1 = row1["pred"]
        pred2 = row2["pred"]
        message = [
            {
                "role": "system",
                "content": "You are an intelligent chatbot designed for evaluating the consistency of generative outputs for similar video-based question-answer pairs. "
                "You will be given two very similar questions, a common answer common to both the questions and predicted answers for the two questions ."
                "Your task is to compare the predicted answers for two very similar question, with a common correct answer and determine if they are consistent. Here's how you can accomplish the task:"
                "------"
                "##INSTRUCTIONS: "
                "- Focus on the consistency between the two predicted answers and the correct answer. Both predicted answers should correspond to the correct answer and to each other, and should not contain any contradictions or significant differences in the conveyed information.\n"
                "- Both predicted answers must be consistent with each other and the correct answer, in terms of the information they provide about the video content.\n"
                "- Consider synonyms or paraphrases as valid matches, but only if they maintain the consistency in the conveyed information.\n"
                "- Evaluate the consistency of the two predicted answers compared to the correct answer.",
            },
            {
                "role": "user",
                "content": "Please evaluate the following video-based question-answer pair:\n\n"
                f"Question 1: {question1}\n"
                f"Question 2: {question2}\n"
                f"Correct Answer: {answer}\n"
                f"Predicted Answer to Question 1: {pred1}\n"
                f"Predicted Answer to Question 2: {pred2}\n\n"
                "Provide your evaluation only as a consistency score where the consistency score is an integer value between 0 and 5, with 5 indicating the highest level of consistency. "
                "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the consistency score in INTEGER, not STRING."
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: {''score': 4.8}.",
            },
        ]
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=message
        )
        response_message = completion.choices[0].message.content

        with open(file_path_saved, "w") as f:
            f.write(response_message)
    else:
        print("exist")


def merge_qa_and_answer_consistency(df_qa1, df_qa2, path_result):
    df_qa = df_qa1.copy()  # Copy df_qa1 to create df_qa

    # Rename columns in df_qa
    df_qa.rename(columns={"question": "question1", "pred": "pred1"}, inplace=True)
    df_qa["gpt3_score"] = None
    df_qa["question2"] = df_qa2["question"]
    df_qa["pred2"] = df_qa2["pred"]

    for (idx1, row1), (idx2, row2) in zip(df_qa1.iterrows(), df_qa2.iterrows()):
        file_path_saved = path_result + str(row1["question_id"]) + ".txt"

        if os.path.exists(file_path_saved):
            with open(file_path_saved, "r") as file:
                try:
                    file_contents = file.read()
                    if file_contents.endswith("."):
                        file_contents = file_contents[:-1]
                    content_dict = eval(file_contents)

                    df_qa.loc[idx1, "gpt3_score"] = content_dict["score"]
                except Exception as e:
                    print(e)
                    print(file_path_saved)
                    continue
        else:
            print(file_path_saved + " not exist")

    path_merged = os.path.join(path_result, "result.csv")
    df_qa.to_csv(path_merged)

    print(df_qa["gpt3_score"].describe())

    return df_qa, path_merged
