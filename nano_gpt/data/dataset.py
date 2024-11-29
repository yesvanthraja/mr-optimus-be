from typing import List, Tuple
import os
import torch
from nano_gpt.models.neural_network import BigramLanguageModel


def read_data() -> str: 
    """
    Reads the input data from the file
    """
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, "input.txt")
    
    with open(file_path, 'r') as f:
        text = f.read()
    return text

def process_data(text: str) -> Tuple[List[str], int]:
    """
    Processes the data and returns the unique set of characters and the vocabulary size

    Args:
        text(str): input text

    Returns:
        Tuple[List[str], int]: List of unique characters and the vocabulary size
    """
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    return chars, vocab_size

def tokenize_input_text(chars: List[str]) -> Tuple[callable, callable]:
    """
    Tokenizes the input text, create a mapping from characters to integers and vice versa

    Args:
        chars(List[str]): List of unique characters

    Returns:
        Tuple[callable, callable]: Returns the encode and decode
    """
    str_to_int = {char: i for i, char in enumerate(chars)}
    int_to_str = {i: char for i, char in enumerate(chars)}
    encode = lambda x: [str_to_int[i] for i in x]
    decode = lambda y: ''.join([int_to_str[i] for i in y])
    return encode, decode, str_to_int, int_to_str

def encode_data_torch_tensor(text: str, chars: List[str]) -> Tuple[torch.Tensor, int]:
    """
    Encodes the input text into a torch tensor.

    Args:
        text (str): Input text to be encoded.
        chars (List[str]): List of unique characters for creating the encoding.

    Returns:
        Tuple[torch.Tensor, int]: Encoded text as a torch tensor and the vocabulary size.
    """
    encode, _, _, _ = tokenize_input_text(chars)
    encoded_text = encode(text)
    data = torch.tensor(encoded_text, dtype=torch.long)
    return data, len(chars)

def train_test_split(data: torch.Tensor, train_fraction: float = 0.9) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Splits the data into training and test set.

    Args:
        data (torch.Tensor): Input data to be split.
        train_fraction (float): Fraction of data to be used for training.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Training and test data.
    """
    n = int(train_fraction * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return {'train': train_data, 'validation': val_data}

def data_chunking(train_data: torch.Tensor, block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Chunk the data into blocks of size block_size.

    Args:
        data (torch.Tensor): Input data to be chunked.
        block_size (int): Size of the block.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Chunked data.
    """
    block_size = 8
    chunks = train_data[:block_size + 1]
    return chunks

def understand_chunking(x_train, block_size, batch_size):
    """
    Understand how the data is chunked.
    """
    x = x_train[:block_size]
    y = x_train[1:block_size + 1]
    for b in range(batch_size):
        for t in range(block_size):
            context = x[b :t+1]
            target = y[t]
            print(f"when input is {context} the target is {target}")

def data_batching(train_data: torch.Tensor, val_data: torch.Tensor ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create batches of data.

    Args:
        train_data (torch.Tensor): Training data.
        val_data (torch.Tensor): Validation data.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Batches of training and validation data.
    """
    torch.manual_seed(1337)
    batch_size = 4 # how many independent sequences will we process in parallel
    block_size = 8 # the number of tokens in the text that we will process at once

    def get_batch(split):
        # generate small batch of data of inputs for x and y
        data = train_data if split == 'train' else val_data
        start_points = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[start:start + block_size] for start in start_points])
        y = torch.stack([data[start + 1:start + block_size + 1] for start in start_points])
        return x, y
    return get_batch('train')

if __name__ == "__main__":
    text = read_data()
    chars, vocab_size = process_data(text)
    encode, decode, str_to_int, int_to_str = tokenize_input_text(chars)
    tensor_data, tensor_length = encode_data_torch_tensor(text, chars)
    train_test_data_split = train_test_split(tensor_data)
    chunking = data_chunking(train_test_data_split['train'], 8)
    
    xb, yb = data_batching(train_test_data_split['train'], None)
    print("inputs: ")
    print(xb.shape)
    print(xb)
    print("targets: ")
    print(yb.shape)
    print(yb)
    print("---------------")
    print(understand_chunking(train_test_data_split['train'], 8, 4))

    nueral_net = BigramLanguageModel(vocab_size)
    logits, loss = nueral_net(xb, yb)
    print(logits.shape)
    print(loss)

    
    # print()
    # print("Chunks", chunking)
    # print("Train Data", train_test_data_split)
    # print()
    # print("Tensor Data", tensor_data[:1000])
    # print(str_to_int)
    # print(encode('hello'))
    # print(''.join(chars))
    # print(vocab_size)
