#
# PLEASE DOWNLOAD VNCoreNLP WITH AVALIABLE SCRIPT IN SCRIPTS FOLDER
#

import numpy as np
import pandas as pd
from ast import literal_eval
from IPython.core.interactiveshell import InteractiveShell
from preprocess_utils import (
    get_path,
    norm_unicode,
    remove_duplicated_punctuations,
    find_consecutive_ranges,
    tokenize_word,
    annotate,
)


def load_raw_data(path: str) -> pd.DataFrame:
    raw_data = pd.read_csv(path)
    raw_data["index_spans"] = raw_data["index_spans"].apply(literal_eval)

    # normalize unicode
    for row_idx in range(len(raw_data)):
        raw_data.loc[row_idx, "content"] = norm_unicode(
            raw_data.loc[row_idx, "content"]
        )

    return raw_data


def load_text_spans_from_raw_data(raw_data: pd.DataFrame) -> list:
    data = []
    for _, row in raw_data.iterrows():
        text = row["content"]
        idx_spans = row["index_spans"]
        temp = []
        text_spans = []
        if idx_spans:
            segments = find_consecutive_ranges(idx_spans)
            for seg in segments:
                if len(seg) == 2 and seg[0] == seg[1]:  # single word
                    temp.append([seg[0]])
                    text_spans.append(
                        text[seg[0] : seg[-1] + 1]
                    )  # substring of text from seg[0] to seg[-1] + 1
                else:
                    temp.append([seg[0], seg[-1]])
                    text_spans.append(
                        text[seg[0] : seg[-1] + 1]
                    )  # substring of text from seg[0] to seg[-1] + 1
        data.append({"text": text, "index_HOS_spans": temp, "text_spans": text_spans})
    return data


def process_BIO_data(data: list) -> pd.DataFrame:
    formated_data = []
    for d in data:
        text = d["text"]
        pos = [i for i in range(len(text))]
        text, pos = remove_duplicated_punctuations(text, pos)
        tokens, alignment = tokenize_word(text, pos)
        annotations = annotate(d["index_HOS_spans"], alignment, tokens)
        ls = [[tokens[i], annotations[i]] for i in range(len(tokens))]

        formated_data.extend(ls)
        formated_data.append([None])  # add None to separate records
    df_final = pd.DataFrame(formated_data, columns=["Word", "Tag"])
    sentence_id = []
    sentence = 0  # sentence id is ordered from 0
    for i in range(len(df_final)):
        if df_final["Word"][i] != None:  # if not None, add sentence id
            sentence_id.append(sentence)
        else:  # if None (separate record data), add None to sentence id and increase sentence id
            sentence_id.append(np.nan)
            sentence += 1
    df_final["sentence_id"] = sentence_id
    df_final.dropna(inplace=True)  # drop None records
    df_final["sentence_id"] = df_final["sentence_id"].astype("int64")
    return df_final


def preprocess(data):
    raw_data_path = get_path(
        parent_dir="hate_speech_text_span_detection/data/raw", file_name=data
    )
    saved_data_path = get_path(
        parent_dir="hate_speech_text_span_detection/data/processed",
        file_name=data,
        remove_file_if_exist=True,
    )

    raw_data = load_raw_data(raw_data_path)
    raw_data_with_text_spans = load_text_spans_from_raw_data(raw_data)

    BIO_data = process_BIO_data(raw_data_with_text_spans)
    BIO_data.reset_index(inplace=True)
    BIO_data.to_csv(saved_data_path)


if __name__ == "__main__":
    InteractiveShell.ast_node_interactivity = "all"

    # Pandas global settings
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    pd.options.display.max_rows

    print("Preprocessing train data...")
    preprocess("train.csv")

    print("Preprocessing dev data...")
    preprocess("dev.csv")

    print("Preprocessing test data...")
    preprocess("test.csv")

    print("Preprocessing data successfully!")
