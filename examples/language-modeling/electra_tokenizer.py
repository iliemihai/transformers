from transformers.tokenization_utils import PreTrainedTokenizer, PreTrainedTokenizerFast
from tokenizers import BertWordPieceTokenizer

class ElectROTokenizer(PreTrainedTokenizerFast):

    def __init__(
        self,
        vocab_file,
        sep_token="[SEP]",
        cls_token="[CLS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        **kwargs
    ):
        super().__init__(
            BertWordPieceTokenizer(
                    vocab_file=vocab_file
            ),
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )
