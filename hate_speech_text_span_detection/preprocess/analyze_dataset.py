# |-------------------------------------------------------------------------------------------|
# | Description: This script is used to analyze the dataset.                                  |
# | Note: It runs via the command line and operates independently of the Flask server.        |
# | The figures include:                                                                      |
# |     - Raw data:                                                                           |
# |         + Number of sentences in the dataset                                              |
# |         + Number of sentences which contain hate speech text spans                        |
# |     - Processed data:                                                                     |
# |         + Number of words in the dataset                                                  |
# |         + Number of words which is hate speech text                                       |
# |         + Hate speech text frequency                                                      |
# |-------------------------------------------------------------------------------------------|

from ast import literal_eval
from preprocess_utils import get_path
from typing import List
import pandas as pd


def analyze(title: str, data_file_names: List[str], callback):
    print(title)
    for file_name in data_file_names:
        print(f"   + {file_name[:-4]}: {callback(file_name)}")


def raw_data_analysis():
    parent_dir = "hate_speech_text_span_detection/data/raw"
    data_file_names = ["train.csv", "validation.csv", "test.csv"]

    print("-------------- Raw data analysis --------------")
    # Number of sentences in the dataset
    analyze(
        "Number of sentences in the dataset",
        data_file_names,
        lambda file_name: pd.read_csv(get_path(parent_dir, file_name)).shape[0],
    )

    # Number of sentences which contain hate speech text spans
    def count_hate_speech_text(file_name):
        count = sum(
            [
                1 if e else 0
                for e in pd.read_csv(get_path(parent_dir, file_name))[
                    "index_spans"
                ].apply(literal_eval)
            ]
        )
        return (
            count,
            f"{100 * count / pd.read_csv(get_path(parent_dir, file_name)).shape[0]:.2f} %",
        )

    analyze(
        "Number of sentences which contain hate speech text spans",
        data_file_names,
        count_hate_speech_text,
    )


def processed_data_analysis():
    parent_dir = "hate_speech_text_span_detection/data/processed"
    data_file_names = ["train.csv", "validation.csv", "test.csv"]

    print("-------------- Processed data analysis --------------")
    # Number of words in the dataset
    analyze(
        "Number of words in the dataset",
        data_file_names,
        lambda file_name: pd.read_csv(get_path(parent_dir, file_name)).shape[0],
    )

    # Number of words which is hate speech text
    def count_hate_speech_text_spans(file_name):
        count = sum(
            [
                1 if e == "B-T" or e == "I-T" else 0
                for e in pd.read_csv(get_path(parent_dir, file_name))["Tag"]
            ]
        )
        return (
            count,
            f"{100 * count / pd.read_csv(get_path(parent_dir, file_name)).shape[0]:.2f} %",
        )

    analyze(
        "Number of words which is hate speech text",
        data_file_names,
        count_hate_speech_text_spans,
    )

    # Hate speech text frequency
    def count_hate_speech_text_frequency(file_name):
        hos_text = []
        frequency = []
        df = pd.read_csv(get_path(parent_dir, file_name))

        for _, row in df.iterrows():
            if row["Tag"] == "O":
                continue

            if row["Word"] not in hos_text:
                hos_text.append(row["Word"])
                frequency.append(1)
            else:
                frequency[hos_text.index(row["Word"])] += 1

        df = pd.DataFrame({"HOS Text": hos_text, "Frequency": frequency})
        df = df.sort_values(by="Frequency", ascending=False)
        df = df.reset_index(drop=True)

        # Save the result to a csv file
        df.to_csv(
            get_path(
                "hate_speech_text_span_detection/data/analysis",
                f"{file_name[:-4]}_hos_text_frequency.csv",
            )
        )

        return df.head(3)

    analyze(
        "Hate speech text frequency",
        data_file_names,
        count_hate_speech_text_frequency,
    )


if __name__ == "__main__":
    raw_data_analysis()
    processed_data_analysis()
