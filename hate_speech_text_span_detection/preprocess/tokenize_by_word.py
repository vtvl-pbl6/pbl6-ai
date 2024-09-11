#
# PLEASE DOWNLOAD VNCoreNLP WITH AVALIABLE SCRIPT IN SCRIPTS FOLDER
#

import numpy as np
import pandas as pd
from ast import literal_eval
from IPython.core.interactiveshell import InteractiveShell
from utils import (
    get_path,
    norm_unicode,
    preprocess,
    find_consecutive_ranges,
    tokenize_word,
    annotate,
)

InteractiveShell.ast_node_interactivity = "all"


# Pandas global settings
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
pd.options.display.max_rows

# Pandas load raw data
raw_train, raw_dev, raw_test = None, None, None
raw_data_dir = get_path("hate_speech_text_span_detection/data/raw", "")


def load_raw_data():
    global raw_train, raw_dev, raw_test
    raw_train = pd.read_csv(f"{raw_data_dir}/train.csv")
    raw_train["index_spans"] = raw_train["index_spans"].apply(literal_eval)
    norm_unicode(raw_train)

    raw_dev = pd.read_csv(f"{raw_data_dir}/dev.csv")
    raw_dev["index_spans"] = raw_dev["index_spans"].apply(literal_eval)
    norm_unicode(raw_dev)

    raw_test = pd.read_csv(f"{raw_data_dir}/test.csv")
    raw_test["index_spans"] = raw_test["index_spans"].apply(literal_eval)
    norm_unicode(raw_test)


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
        text, pos = preprocess(text, pos)
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


def save_BIO_data():
    save_path = get_path("hate_speech_text_span_detection/data/processed", "")

    BIO_train = process_BIO_data(load_text_spans_from_raw_data(raw_train))
    BIO_dev = process_BIO_data(load_text_spans_from_raw_data(raw_dev))
    BIO_test = process_BIO_data(load_text_spans_from_raw_data(raw_test))

    BIO_train.reset_index(inplace=True)
    BIO_dev.reset_index(inplace=True)
    BIO_test.reset_index(inplace=True)

    # Save to csv
    BIO_train.to_csv(f"{save_path}/train.csv")
    BIO_dev.to_csv(f"{save_path}/dev.csv")
    BIO_test.to_csv(f"{save_path}/test.csv")


if __name__ == "__main__":
    load_raw_data()
    save_BIO_data()
    print("Preprocess data successfully!")
