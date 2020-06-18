import os, json, sys
from tqdm import tqdm
import multiprocessing
import logging, torch
from transformers import *
from electra_tokenizer import ElectROTokenizer
logging.basicConfig(level=logging.DEBUG)

VOCAB_PATH = "./ElectraConfig"
IN_PATH = "./corpus/ro_train.txt"
OUT_PATH = "./corpus/encoded_ro_train.txt"
BLOCK_SIZE = 512

def blocks(files, size=65536):
    while True:
        b = files.read(size)
        if not b: break
        yield b

def get_num_lines():
    with open(IN_PATH, "r",encoding="utf-8",errors='ignore') as f:
        return (sum(bl.count("\n") for bl in blocks(f)))

#tot_num_lines = get_num_lines())

class TokenizeCorpus:

    def __init__(self):
        self.fin = open(IN_PATH, "r", encoding="utf-8")
        self.fout = open(OUT_PATH, "w", encoding="utf-8")
        self.tokenizer = ElectROTokenizer(os.path.join(VOCAB_PATH, "vocab.txt"))
        self.buffer = ""
        self.arr = []

    def run(self):

        for line in tqdm(self.fin, total=90410523):
            if line.strip() == "":
                pass

            line_tokenized = self.tokenizer._tokenizer.encode(line, add_special_tokens=False).ids
            len_line_tokenized = len(line_tokenized)
            if len_line_tokenized > BLOCK_SIZE - 2:
                if len(self.buffer) > 0:
                    self.arr.append(self.tokenizer._tokenizer.encode(line).ids)
                    self.buffer = ""
                full_blocks, remainder_line = self.split_long_line(line, self.tokenizer, BLOCK_SIZE)
                if len(full_blocks) == 0:
                    continue

                for block in full_blocks:
                    self.fout.write("%s\n" % block)
                self.buffer = remainder_line
                continue

            len_buffer_tokenized = len(self.tokenizer._tokenizer.encode(self.buffer, add_special_tokens=False).ids)
            if len_buffer_tokenized + len_line_tokenized > BLOCK_SIZE - 2:
                self.arr.append(self.tokenizer._tokenizer.encode(self.buffer.strip()).ids)
                self.buffer = line.strip()
            else:
                self.buffer = self.buffer + " " + line.strip()

            if len(self.buffer.strip()) > 0:
                self.arr.append(self.tokenizer._tokenizer.encode(self.buffer.strip()).ids)

            for t in self.arr:
                self.fout.write(" %s\n" % t)

            self.arr = []
            self.buffer = ""

    def split_long_line(self, line, tokenizer, max_len):
        full_blocks = []
        cnt = 0
        while True:
            cnt += 1
            if cnt>20: # this means we have some sort of error and we are stuck in this loop
                return [], ""
            len_line_tokenized = len(tokenizer._tokenizer.encode(line, add_special_tokens=False).ids)

            if len_line_tokenized < max_len - 2:
                return full_blocks, line

            # search for the place to cut first part
            parts = line.split(" ")
            initial_split_point = 1 #int(len(parts) * max_len / len_line_tokenized) - 5
            i = 0
            for i in range(max(1,initial_split_point), len(parts)):
                possible_split =  " ".join(parts[:i])
                #logging.debug(possible_split)
                len_possible_split = len(tokenizer._tokenizer.encode(possible_split, add_special_tokens=False).ids)
                #logging.debug("\t\t Trying to split at word {}, with len {}".format(i, len_possible_split))
                if len_possible_split > max_len - 2: # we're over the limit
                    break

            if i == 0:
                return [], ""

            current_split = " ".join(parts[:i-1]).strip()
            full_blocks.append(tokenizer._tokenizer.encode(current_split).ids)
            line = line[len(current_split):].strip()
                
    def __del__(self):
        self.fin.close()
        self.fout.close()

tok = TokenizeCorpus()
tok.run()
