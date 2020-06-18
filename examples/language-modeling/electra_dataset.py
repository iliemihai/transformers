import os
import mmap
import logging
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    PreTrainedTokenizer,
)
from electra_tokenizer import ElectROTokenizer

logger = logging.getLogger(__name__)

def remove_brackets(tok):
    tok=tok.replace("[","").replace("]","")
    return tok

def my_collate(batch):
    batch = filter(lambda x: x is not None, batch)
    return DataLoader.default_collate(list(batch))

class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size=512, local_rank=-1):
        assert os.path.isfile(file_path)
        
        logger.info("Reading file %s", file_path)

        # Memory mapping improves I/O performances because it does not involve a separate system
        # call for each access. The same memory is accessed in both kernel and user space
        with open(file_path, encoding="utf-8") as f:
            self.mmap_file = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
            self.offsets = [0]
            for line in iter(self.mmap_file.readline, b""):
                self.offsets.append(self.mmap_file.tell())

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, i):
        strtoks = list(map(remove_brackets, self.mmap_file[self.offsets[i]:self.offsets[i+1]].decode().split(", ")))
        #convert list of strings to list of integers
        inttoks = list(map(int, strtoks))
        if len(inttoks) > 512:
            print("MAI MARE LA INDEX", i, len(inttoks), inttoks)
        #    inttoks=inttoks[:512]
        #    inttoks[511]=3
        return torch.tensor(inttoks, dtype=torch.long)

