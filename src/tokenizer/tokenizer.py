import os
import sentencepiece as spm

class BPETokenizer:
    def __init__(self, model_path: str=None):
        self.sp = None
        self.model_path = model_path

    def train(self, input_file: str, model_prefix: str="caption_bpe", vocab_size: int=6000):
        """Train tokenizer"""
        options = dict(
            # Input Specs:
            input=input_file,
            input_format="text",
            # Output Specs:
            model_prefix=model_prefix,
            # Algorithm Specs:
            model_type="bpe",
            vocab_size=vocab_size,
            # Rare Word Treatment:
            character_coverage=0.99995,
            byte_fallback=True,
            # Normalization:
            normalization_rule_name='identity',
            remove_extra_whitespaces=False,
            input_sentence_size=200000000,
            max_sentence_length=4192,
            seed_sentencepiece_size=1000000,
            shuffle_input_sentence=True,
            # Merge Rules:
            split_digits=True,
            split_by_unicode_script=True,
            split_by_whitespace=True,
            split_by_number=True,
            max_sentencepiece_length=16,
            add_dummy_prefix=True,
            allow_whitespace_only_pieces=True,
            # Special Tokens
            unk_id=0,
            bos_id=1,
            eos_id=2,
            pad_id=3,
            # Systems
            num_threads=os.cpu_count(),
        )
        spm.SentencePieceTrainer.train(**options)
        self.model_path = f"{model_prefix}.model"
        self.load_tokenizer()

    def load_tokenizer(self):
        """Load the tokenizer"""
        if self.model_path and os.path.exists(self.model_path):
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(self.model_path)
        else:
            raise ValueError("Tokenizer model not found. Train first or provide valid path.")

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs"""
        if self.sp is None:
            raise ValueError("Tokenizer not loaded")
        return self.sp.EncodeAsIds(text)
        

    def decode(self, token_ids: list[int]) -> int:
        """Decode token IDs"""
        if self.sp is None:
            raise ValueError("Tokenizer not loaded")
        return self.sp.DecodeIds(token_ids)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        
        if self.sp is None:
            raise ValueError("Tokenizer not loaded")
        
        return self.sp.GetPieceSize()

    def pad(self, token_ids: list[int], max_len: int = 25) -> list[int]:
        """Get token IDs with padding values"""
        if self.sp is None:
            raise ValueError("Tokenizer not loaded")
        
        pad_id = self.sp.pad_id()
        
        if len(token_ids) < max_len:
            return token_ids + [pad_id] * (max_len - len(token_ids))
        
        return token_ids[:max_len]
    
    def get_input_and_target(self, token_ids: list[int]) -> tuple:
        """Get input and target token IDs with padding values"""
        if self.sp is None:
            raise ValueError("Tokenizer not loaded")
        
        input_ids = self.pad([self.sp.bos_id()] + token_ids)
        target_ids = self.pad(token_ids + [self.sp.eos_id()])
        
        return input_ids, target_ids