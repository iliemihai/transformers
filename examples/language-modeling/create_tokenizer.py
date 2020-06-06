# pip3 install tokenizers

from pathlib import Path

from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer

paths = [str(x) for x in Path("./raw_corpus/").glob("**/*.txt")]

# Initialize a tokenizer
tokenizer = BertWordPieceTokenizer(clean_text=True, handle_chinese_chars=True, strip_accents=True, lowercase=True)

# Customize training
tokenizer.train(files=paths, vocab_size=32000, min_frequency=3, show_progress=True, special_tokens=[
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
],wordpieces_prefix="##")

# Save files to disk
import os
OUT_DIR = "romanian_tokenizer_bpe_32k"
os.makedirs(OUT_DIR, exist_ok=True)
tokenizer.save(OUT_DIR, "ro")
