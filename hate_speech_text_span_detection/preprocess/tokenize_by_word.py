import pandas as pd
from ast import literal_eval
from IPython.core.interactiveshell import InteractiveShell
from utils import get_path, norm_unicode

InteractiveShell.ast_node_interactivity = "all"


# Pandas settings
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
pd.options.display.max_rows

# Pandas load data
data_dir = get_path("hate_speech_text_span_detection/data/raw", "")

train = pd.read_csv(f"{data_dir}/train.csv")
train["index_spans"] = train["index_spans"].apply(literal_eval)
norm_unicode(train)

dev = pd.read_csv(f"{data_dir}/dev.csv")
dev["index_spans"] = dev["index_spans"].apply(literal_eval)
norm_unicode(dev)

test = pd.read_csv(f"{data_dir}/test.csv")
test["index_spans"] = test["index_spans"].apply(literal_eval)
norm_unicode(test)

if __name__ == "__main__":
    print(test.head(2))
