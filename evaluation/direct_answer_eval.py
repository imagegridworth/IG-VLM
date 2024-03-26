import pandas as pd


def eval_multiple_choice(df):
    df["predicted_answer"] = df.apply(map_prediction_to_answer_v2, axis=1)
    df["is_correct"] = df["predicted_answer"] == df["answer"]
    total_accuracy = df["is_correct"].mean()

    print(f"Total Accuracy: {total_accuracy:.4f}")

    if "question_type" in df.columns:
        accuracy_report = df.groupby("question_type")["is_correct"].mean()
        print(accuracy_report)
        df["prefix"] = df["question_type"].apply(lambda x: x[0])
        grouped_accuracy = df.groupby("prefix")["is_correct"].mean()
        print(grouped_accuracy)


def map_prediction_to_answer_v2(row):
    answer_column = None
    if isinstance(row["pred"], str):
        prediction_letter = row["pred"][0]
        if prediction_letter in ["A", "B", "C", "D", "E"]:
            answer_column = "a" + str(ord(prediction_letter) - ord("A"))
        if "answer is " in row["pred"]:
            row["pred"] = row["pred"][row["pred"].index("answer is") :]
        if "A:" in row["pred"] or "A)" in row["pred"]:
            answer_column = "a0"
        elif "B:" in row["pred"] or "B)" in row["pred"]:
            answer_column = "a1"
        elif "C:" in row["pred"] or "C)" in row["pred"]:
            answer_column = "a2"
        elif "D:" in row["pred"] or "D)" in row["pred"]:
            answer_column = "a3"
        elif "E:" in row["pred"] or "E)" in row["pred"]:
            answer_column = "a4"
    if answer_column in ["a0", "a1", "a2", "a3", "a4"]:
        return row[answer_column]
    elif answer_column:
        print(prediction_letter)
    return "None"
