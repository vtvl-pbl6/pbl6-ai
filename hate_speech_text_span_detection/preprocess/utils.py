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
def replace(
    text: str, pattern: str, replacement: str, pos: List[int]
) -> Tuple[str, List[int]]:
    matches = [0]

    def capture_and_replace(match: re.Match, ret: str) -> str:
        matches.extend([match.start() + 1, match.end()])
        return ret

    l = len(text)
    text = re.sub(
        pattern,
        lambda match: capture_and_replace(match, replacement),
        text,
        flags=re.IGNORECASE,
    )
    matches.append(l)
    slices = np.array_split(matches, int(len(matches) / 2))
    res = []
    for s in slices:
        res += pos[s[0] : s[1]]
    assert len(text) == len(res)
    return text, res


def preprocess(text: str, pos: List[int]) -> Tuple[str, List[int]]:
    # collapse duplicated punctuations
    punc = ",. !?\"'"
    for c in punc:
        pat = "([" + c + "]{2,})"
        text, pos = replace(text, pat, c, pos)
    assert len(text) == len(pos)
    return text, pos
