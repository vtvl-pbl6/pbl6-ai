# |------------------------------------------------------------------------------------|
# | Description: This script is used to train the deep learning model.                 |
# | Note: It runs via the command line and operates independently of the Flask server. |
# |------------------------------------------------------------------------------------|

checkpoint_parent_dir = "hate_speech_text_span_detection/model/checkpoint"
data_parent_dir = "hate_speech_text_span_detection/data/processed"

"""# 1. Khai báo các thư viện cần thiết"""

from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from tqdm import tqdm
from IPython.display import clear_output
from typing import List, Tuple
import os
from os import walk
import matplotlib.pyplot as plt

# For training and evaluation
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import XLMRobertaModel, AutoTokenizer

# For data processing
import pandas as pd
from torch.utils.data import DataLoader, Dataset

"""### 1.1. Setup ban đầu"""

tf.keras.backend.clear_session()

# Clear memory
torch.cuda.empty_cache()

# Clear output of the cell
clear_output()

# Set runtime on GPU if available
device = torch.device("cpu")
device

"""### 1.2. Tạo input model"""

train_path = f"{data_parent_dir}/train_copy.csv"
validation_path = f"{data_parent_dir}/validation_copy.csv"
test_path = f"{data_parent_dir}/test_copy.csv"

input_model = XLMRobertaModel.from_pretrained(
    "xlm-roberta-base"
)  # load pre-trained model
tokenizer = AutoTokenizer.from_pretrained(
    "xlm-roberta-base"
)  # load pre-trained tokenizer
input_model.resize_token_embeddings(len(tokenizer))

clear_output()

"""# 2. Xử lý dữ liệu"""

clear_output()

"""### 2.1. Tiền xử lý data"""


def prepare_data(file_path: str) -> Tuple[List[str], List[List[int]]]:
    df = pd.read_csv(file_path)

    # Drop rows with missing values
    df = df.dropna()
    df = df.reset_index(drop=True)

    texts = df["Word"].tolist()
    spans = df["Tag"].tolist()

    # Convert spans to binary (0, 1)
    binary_spans = []
    for span in spans:
        binary_span = []
        span = span.split(" ")
        for s in span:
            if s == "O":
                binary_span.append(0)
            else:
                binary_span.append(1)
        binary_spans.append(binary_span)

    return texts, binary_spans


"""### 2.2. Tạo class TextDataset và hàm tạo dataloader"""


# Dataloader class
class TextDataset(Dataset):
    def __init__(
        self, tokenizer, texts: List[str], spans: List[List[int]], max_len: int
    ):
        # Tokenize text with output format {'input_ids': [], 'attention_mask': []}s
        self.texts = [
            tokenizer(
                text,
                padding="max_length",
                max_length=max_len,
                truncation=True,
                return_tensors="pt",
            )
            for text in texts
        ]
        self.spans = []

        for span in spans:  # Padding spans to max_len
            if len(span) < max_len:
                self.spans.append(span + [0] * (max_len - len(span)))
            else:
                self.spans.append(span[:max_len])

        self.spans = torch.tensor(self.spans)

    def __len__(self):
        return len(self.spans)

    def __getitem__(self, index):
        return self.texts[index], self.spans[index]


def create_dataloader(
    texts, spans, batch_size, tokenizer, max_len, shuffle=True
) -> DataLoader:
    dataset = TextDataset(tokenizer, texts, spans, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


"""### 2.3. Tạo dataloader cho dữ liệu **train**, **validation** và **test**"""

batch_size = 64  # devide data into batches
train_dataloader = create_dataloader(
    *prepare_data(train_path), batch_size=batch_size, tokenizer=tokenizer, max_len=64
)
validation_dataloader = create_dataloader(
    *prepare_data(validation_path),
    batch_size=batch_size,
    tokenizer=tokenizer,
    max_len=64,
    shuffle=False,
)
test_dataloader = create_dataloader(
    *prepare_data(test_path), batch_size=batch_size, tokenizer=tokenizer, max_len=64
)

"""# 3. Tạo mô hình huấn luyện

### 3.1. Tạo lớp mô hình huấn luyện
"""


class MultiTaskModel(nn.Module):
    def __init__(self, input_model):
        super(MultiTaskModel, self).__init__()
        self.bert = input_model
        self.span_classifier = nn.Linear(768, 1)  # Classification head
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        # Forward pass through the BERT model
        output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=False
        )
        last_hidden_state = output[
            0
        ]  # Hidden state of shape (batch_size, sequence_length, hidden_size)

        # Apply dropout
        last_hidden_state = self.dropout(last_hidden_state)

        # Apply the span classifier to get logits for each token (batch_size, sequence_length, 1)
        span_logits = self.span_classifier(last_hidden_state)

        span_logits = span_logits.permute(0, 2, 1)
        span_logits = torch.sigmoid(span_logits)
        span_logits = span_logits.permute(0, 2, 1)

        return span_logits

    def predict(self, input_ids, attention_mask, threshold: float) -> np.ndarray[tuple]:
        span_logits = self.forward(input_ids, attention_mask)  # Tensor
        span_logits = (
            span_logits.squeeze().cpu().detach().numpy()
        )  # Convert tensor to numpy array

        span_logits = (
            span_logits.reshape(1, 64) if span_logits.shape[0] == 1 else span_logits
        )

        span_logits = [max(e) for e in span_logits]

        # Convert logits to binary labels (batch_size, 1)
        predictions = [(1, e) if e > threshold else (0, e) for e in span_logits]

        return predictions


"""### 3.2. Tạo các hàm tính toán chung"""


def calculate_f1(preds, y):
    return f1_score(y, preds, average="macro")


def calculate_accuracy(preds, y):
    return accuracy_score(y, preds)


def save_checkpoint(model: MultiTaskModel, filename: str):
    torch.save(model.state_dict(), f"{checkpoint_parent_dir}/{filename}")


def load_checkpoint(model: MultiTaskModel, filename: str) -> MultiTaskModel:
    model.load_state_dict(torch.load(f"{checkpoint_parent_dir}/{filename}"))
    return model


def get_epoch_name(id) -> str:
    return f"epoch_{id}.pt"


def tensor_to_numpy(tensor):
    if torch.is_tensor(tensor):
        # Check if tensor is on GPU, move it to CPU first if necessary
        if tensor.is_cuda:
            tensor = tensor.cpu()
        return tensor.numpy()
    else:
        # If input is not a tensor, return it as is
        return tensor


def save_training_info(
    epoch,
    train_loss,
    train_acc,
    val_loss,
    val_acc,
    val_f1,
    epochs_without_improvement,
    clear=False,
    file_name="training_info.csv",
):
    df = pd.DataFrame(
        {
            "epoch": [epoch],
            "train_loss": [train_loss],
            "train_acc": [train_acc],
            "val_loss": [val_loss],
            "val_acc": [val_acc],
            "val_f1": [val_f1],
            "epochs_without_improvement": [epochs_without_improvement],
        }
    )

    # If `clear` is True, overwrite the file; otherwise, append to the file
    if clear:
        df.to_csv(
            f"{checkpoint_parent_dir}/{file_name}", mode="w", header=True, index=False
        )
    else:
        # Check if file exists, and append to it if it does, or create a new file otherwise
        file_exists = os.path.isfile(f"{checkpoint_parent_dir}/{file_name}")
        df.to_csv(
            f"{checkpoint_parent_dir}/{file_name}",
            mode="a",
            header=not file_exists,
            index=False,
        )


def load_training_info(file_name="training_info.csv"):
    if os.path.exists(f"{checkpoint_parent_dir}/{file_name}"):
        df = pd.read_csv(f"{checkpoint_parent_dir}/{file_name}")
        if not df.empty:
            # Convert the entire DataFrame to a list of dictionaries (each row as a dictionary)
            return df.to_dict(orient="records")
    return None


"""### 3.3. Hàm huấn luyện mô hình"""


def train(
    model: MultiTaskModel,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    criterion_span: nn.BCELoss,
    optimizer_spans: optim.Adam,
    device: torch.device,
    num_epochs: int,
    patience: int = 5,  # Number of epochs to wait for improvement
    info_file: str = "training_info.csv",
):
    model.train()  # Turn on training mode

    # Load the previous training info if it exists
    start_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    saved_info_list = load_training_info(info_file)
    if saved_info_list:
        saved_info = saved_info_list[-1]
        best_val_loss = saved_info["val_loss"]
        epochs_without_improvement = saved_info["epochs_without_improvement"]
        start_epoch = saved_info["epoch"]
        print(
            f"Resuming training from epoch {start_epoch} with {epochs_without_improvement} epochs without improvement."
        )

    if start_epoch >= num_epochs or epochs_without_improvement >= patience:
        return

    # Load checkpoint if start_epoch > 0
    if start_epoch > 0:
        model = load_checkpoint(model, get_epoch_name(start_epoch))

    for epoch in range(start_epoch, num_epochs):
        print("Epoch: ", epoch + 1)
        train_loss = 0
        train_span_preds = []
        train_span_targets = []

        # Training loop
        for texts, spans in tqdm(train_dataloader):
            input_ids = texts["input_ids"].squeeze(1).to(device)
            attention_mask = texts["attention_mask"].to(device)
            spans = spans.float().to(device)

            optimizer_spans.zero_grad()
            span_logits = model.forward(input_ids, attention_mask)  # Forward pass
            loss_span = criterion_span(span_logits.squeeze(), spans)

            loss = loss_span
            loss.backward()

            optimizer_spans.step()
            train_loss += loss.item()

            # Save the true labels and predicted labels for each sample
            train_span_preds.append(
                span_logits.squeeze().cpu().detach().numpy().flatten()
            )
            train_span_targets.append(spans.cpu().numpy().flatten())

        # Validation loop
        val_loss = 0
        val_span_preds = []
        val_span_targets = []

        for texts, spans in tqdm(validation_dataloader):
            input_ids = texts["input_ids"].squeeze(1).to(device)
            attention_mask = texts["attention_mask"].to(device)
            spans = spans.float().to(device)
            with torch.no_grad():
                span_logits = model.forward(input_ids, attention_mask)  # Forward pass
                loss_span = criterion_span(span_logits.squeeze(), spans)

                val_loss += loss_span

            # Save the true labels and predicted labels for each sample
            val_span_preds.append(span_logits.squeeze().cpu().numpy().flatten())
            val_span_targets.append(spans.cpu().numpy().flatten())

        # Validation loss, accuracy, F1-score
        val_span_preds = np.concatenate(val_span_preds)
        val_span_preds = (val_span_preds > 0.5).astype(int)
        val_span_targets = np.concatenate(val_span_targets)

        val_loss = tensor_to_numpy(val_loss / len(validation_dataloader))
        val_acc = tensor_to_numpy(calculate_accuracy(val_span_preds, val_span_targets))
        val_f1 = tensor_to_numpy(calculate_f1(val_span_preds, val_span_targets))

        # Train loss, accuracy
        train_span_preds = np.concatenate(train_span_preds)
        train_span_preds = (train_span_preds > 0.5).astype(int)
        train_span_targets = np.concatenate(train_span_targets)

        train_loss = tensor_to_numpy(train_loss / len(train_dataloader))
        train_acc = tensor_to_numpy(
            calculate_accuracy(train_span_preds, train_span_targets)
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0  # Reset patience counter
        else:
            epochs_without_improvement += 1

        print(
            f" -> Train loss: {train_loss}; Train acc: {train_acc} -- Val loss: {val_loss}; Val acc: {val_acc} -- Epochs without improvement: {epochs_without_improvement} -- F1-score: {val_f1}"
        )
        save_training_info(
            epoch + 1,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_f1,
            epochs_without_improvement,
            clear=(epoch == 0),
        )

        # Early stopping check
        if epochs_without_improvement >= patience:
            print("Early stopping triggered. Stopping training...")
            break

        # Save checkpoint
        if os.path.exists(
            f"{checkpoint_parent_dir}/{get_epoch_name(epoch)}"
        ):  # Remove previous checkpoint
            os.remove(f"{checkpoint_parent_dir}/{get_epoch_name(epoch)}")
        save_checkpoint(
            model,
            (
                get_epoch_name(epoch + 1)
                if epoch < num_epochs - 1
                else get_epoch_name("final")
            ),
        )  # Save current checkpoint


"""# 4. Huấn luyện và kiểm tra mô hình

### 4.1. Huấn luyện mô hình
"""

num_epochs = 2

# Create an instance of the multi-task model
model = MultiTaskModel(input_model=input_model)
model.to(device)

criterion_span = nn.BCELoss()

# Define the optimizer
optimizer_spans = optim.Adam(list(model.parameters()), lr=5e-6, weight_decay=1e-5)

train(
    model=model,
    train_dataloader=train_dataloader,
    validation_dataloader=validation_dataloader,
    criterion_span=criterion_span,
    optimizer_spans=optimizer_spans,
    device=device,
    num_epochs=num_epochs,
)

# """### 4.2. Hiển thị các biểu đồ"""


# def plot_training_results(
#     total_train_loss, total_train_acc, total_val_loss, total_val_acc
# ):
#     total_train_loss = [tensor_to_numpy(e) for e in total_train_loss]
#     total_train_acc = [tensor_to_numpy(e) for e in total_train_acc]
#     total_val_loss = [tensor_to_numpy(e) for e in total_val_loss]
#     total_val_acc = [tensor_to_numpy(e) for e in total_val_acc]

#     epochs = range(1, len(total_train_loss) + 1)

#     plt.figure(figsize=(12, 6))

#     # Plot training and validation loss
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs, total_train_loss, "b", label="Training Loss")
#     plt.plot(epochs, total_val_loss, "r", label="Validation Loss")
#     plt.title("Training and Validation Loss")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.legend()

#     # Plot training and validation accuracy
#     plt.subplot(1, 2, 2)
#     plt.plot(epochs, total_train_acc, "b", label="Training Accuracy")
#     plt.plot(epochs, total_val_acc, "r", label="Validation Accuracy")
#     plt.title("Training and Validation Accuracy")
#     plt.xlabel("Epochs")
#     plt.ylabel("Accuracy")
#     plt.legend()

#     plt.tight_layout()
#     plt.show()


# plot_training_results(total_train_loss, total_train_acc, total_val_loss, total_val_acc)

# """### 4.3. Kiểm tra mô hình trên tập dữ liệu test"""

# def test(model, test_dataloader, device):
#     model.eval()
#     span_preds = []
#     span_targets = []
#     for texts, spans in tqdm(test_dataloader):
#         input_ids = texts['input_ids'].squeeze(1).to(device)
#         attention_mask = texts['attention_mask'].to(device)
#         spans = spans.float().to(device)
#         with torch.no_grad():
#             span_logits = model(input_ids, attention_mask)

#         print("-----s")
#         print((span_logits.squeeze().cpu().numpy().flatten() > 0.5).astype(int))
#         print("-----e")

#         span_preds.append(span_logits.squeeze().cpu().numpy().flatten())
#         span_targets.append(spans.cpu().numpy().flatten())

#     span_preds = np.concatenate(span_preds)
#     span_targets = np.concatenate(span_targets)
#     span_preds = (span_preds > 0.5).astype(int)
#     span_f1 = f1_score(span_targets, span_preds, average='macro')

#     print("Span F1 Score: {:.4f}".format(span_f1))

# model = MultiTaskModel(input_model = input_model)
# model.load_state_dict(torch.load(f"{checkpoint_parent_dir}/epoch_26.pt", weights_only=True))
# model.to(device)

# test(model = model, test_dataloader = test_dataloader, device = device)
