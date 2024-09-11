from itertools import groupby
from operator import itemgetter
from vncorenlp import VnCoreNLP
import os
import re
import numpy as np
import pandas as pd
from typing import List, Tuple


def get_path(
    parent_dir: str,
    file_name: str,
    mkdir_dir: bool = True,
    remove_file_if_exist: bool = False,
) -> str:
    parent_dir = f"{os.getcwd()}/{parent_dir}"
    file_path = f"{parent_dir}/{file_name}" if file_name else parent_dir

    if not os.path.exists(parent_dir) and mkdir_dir:
        os.makedirs(parent_dir)

    if os.path.exists(file_path) and remove_file_if_exist:
        os.remove(file_path)

    return file_path


# Initialize annotator
annotator = VnCoreNLP(
    get_path("hate_speech_text_span_detection/vncorenlp", "VnCoreNLP-1.1.1.jar"),
    annotators="wseg",
    max_heap_size="-Xmx500m",
)


def norm_unicode(df: pd.DataFrame) -> pd.DataFrame:
    col_name = "content"
    for row_idx in range(len(df)):
        df.loc[row_idx, col_name] = df.loc[row_idx, col_name].replace("òa", "oà")
        df.loc[row_idx, col_name] = df.loc[row_idx, col_name].replace("óa", "oá")
        df.loc[row_idx, col_name] = df.loc[row_idx, col_name].replace("ỏa", "oả")
        df.loc[row_idx, col_name] = df.loc[row_idx, col_name].replace("õa", "oã")
        df.loc[row_idx, col_name] = df.loc[row_idx, col_name].replace("ọa", "oạ")
        df.loc[row_idx, col_name] = df.loc[row_idx, col_name].replace("òe", "oè")
        df.loc[row_idx, col_name] = df.loc[row_idx, col_name].replace("óe", "oé")
        df.loc[row_idx, col_name] = df.loc[row_idx, col_name].replace("ỏe", "oẻ")
        df.loc[row_idx, col_name] = df.loc[row_idx, col_name].replace("õe", "oẽ")
        df.loc[row_idx, col_name] = df.loc[row_idx, col_name].replace("ọe", "oẹ")
        df.loc[row_idx, col_name] = df.loc[row_idx, col_name].replace("ùy", "uỳ")
        df.loc[row_idx, col_name] = df.loc[row_idx, col_name].replace("úy", "uý")
        df.loc[row_idx, col_name] = df.loc[row_idx, col_name].replace("ủy", "uỷ")
        df.loc[row_idx, col_name] = df.loc[row_idx, col_name].replace("ũy", "uỹ")
        df.loc[row_idx, col_name] = df.loc[row_idx, col_name].replace("ụy", "uỵ")
        df.loc[row_idx, col_name] = df.loc[row_idx, col_name].replace("Ủy", "Uỷ")
    return df


# Replace text with pattern by replacement and keep track of the position
# Ex: replace("Hello, world!!!", "o", "a", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14])
#    -> ("Hella, warld!!!", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14])
def replace(
    text: str, pattern: str, replacement: str, pos: List[int]
) -> Tuple[str, List[int]]:
    match_indices = [0]  # start from first index

    def capture_and_replace(match: re.Match, ret: str) -> str:
        match_indices.extend([match.start() + 1, match.end()])
        return ret

    last_idx = len(text)
    text = re.sub(
        pattern,
        lambda match: capture_and_replace(match, replacement),
        text,
        flags=re.IGNORECASE,
    )
    match_indices.append(last_idx)  # append the last index
    slices = np.array_split(match_indices, int(len(match_indices) / 2))
    res = []
    for s in slices:
        res += pos[s[0] : s[1]]
    assert len(text) == len(res)
    return text, res


# Preprocess text by collapsing duplicated punctuations
# Ex: "Hello, world!!!" -> "Hello, world!"
def preprocess(text: str, pos: List[int]) -> Tuple[str, List[int]]:
    # collapse duplicated punctuations
    punc = ",. !?\"'"
    for c in punc:
        pat = "([" + c + "]{2,})"
        text, pos = replace(text, pat, c, pos)
    assert len(text) == len(pos)
    return text, pos


# Find ranges of consecutive numbers
# Ex: [1, 2, 3, 5, 6, 7, 9] -> [(1, 3), (5, 7), (9, 9)]
def find_consecutive_ranges(numbers: List[int]) -> List[Tuple[int, int]]:
    ranges = []
    for _, group in groupby(
        enumerate(numbers), lambda x: x[0] - x[1]
    ):  # groupby consecutive numbers
        consecutive_group = list(map(itemgetter(1), group))
        ranges.append((consecutive_group[0], consecutive_group[-1]))
    return ranges


# Tokenize text by word
# Ex: tokenize_word("Hay quá Mac _. Lê- Nin", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
#     -> (['Hay', 'quá', 'Mac', '_', '.', 'Lê', '-', 'Nin'], [[0, 1, 2], [4, 5, 6], [8, 9, 10], [12], [13], [15, 16], [17], [19, 20, 21]])
def tokenize_word(text: str, pos: List[int]) -> Tuple[List[str], List[List[int]]]:
    annotator_text = annotator.tokenize(text)
    tokens = []
    for i in range(len(annotator_text)):
        for j in range(len(annotator_text[i])):
            if annotator_text[i][j] == "_":  # add "_" to tokens
                tokens.append(annotator_text[i][j])
            else:  # remove "_" from annotator_text, then split by "_" and add all syllables to tokens
                annotator_text[i][j] = annotator_text[i][j].lstrip("_")
                annotator_text[i][j] = annotator_text[i][j].rstrip("_")
                syllabel = annotator_text[i][j].split("_")
                for idx in range(len(syllabel)):
                    tokens.append(syllabel[idx])
    alignment = []
    start = 0
    for t in tokens:  # find and group positions of each token in the original text
        res = text.find(t, start)
        alignment.append(pos[res : res + len(t)])
        start = res + len(t)
    assert len(tokens) == len(alignment)
    return tokens, alignment


# Annotate tokens by BIO schema
# Explain:
#   B-T: Begin of token
#   I-T: Inside of token
#   O  : Outside of token
def annotate(
    index_HOS_spans: List[List[int]], alignment: List[int], tokens: List[str]
) -> pd.Series:
    Tag = []
    annotations = pd.DataFrame()
    annotations["Tokens"] = tokens
    for i in range(len(tokens)):
        Tag.append("O")
    annotations["Tag"] = Tag
    for indices in index_HOS_spans:
        i = 0
        while i < len(alignment):
            if (
                alignment[i][-1] < indices[0]
            ):  # if the last index of alignment[i] is less than the first index of indices => outside of token
                i += 1
            elif (
                alignment[i][0] <= indices[0] <= alignment[i][-1]
            ):  # if the first index of indices is in alignment[i] => begin of token
                annotations.loc[i, "Tag"] = "B-T"
                i += 1
            elif (
                indices[0] < alignment[i][0] <= indices[-1]
            ):  # if the first index of alignment[i] is in indices => inside of token
                annotations.loc[i, "Tag"] = "I-T"
                i += 1
            elif alignment[i][0] > indices[-1]:
                break
    return annotations["Tag"]
