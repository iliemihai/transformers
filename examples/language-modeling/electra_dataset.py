import os
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
#from transformers.tokenization_electra import ElectraTokenizer

logger = logging.getLogger(__name__)

class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size=512, local_rank=-1):
        assert os.path.isfile(file_path)
        logger.info("Creating features from dataset file at %s", file_path)
        
        # -------------------------- TAKING LONG TIME, NEED TO DO IT PARALLEL ON BATCHES

        logger.info("Reading file %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        logger.info("Running tokenization")
        self.examples = tokenizer._tokenizer.encode_batch(lines)#[tokenizer.encode(line) for line in lines]
        
        # -------------------------- CHANGES END

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].ids, dtype=torch.long)

