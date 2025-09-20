import os
import sentencepiece as spm

class BPETokenizer:
    def __init__(self, model_path=None):
        self.sp = spm.SentencePieceProcessor()
        if model_path:
            self.load(model_path)

    def train(self, input_file, model_prefix="caption_bpe", vocab_size=6000):
        options = dict(
            input=input_file,
            input_format="text",
            model_prefix=model_prefix,
            model_type="bpe",
            vocab_size=vocab_size,
            character_coverage=0.99995,
            byte_fallback=True,
            unk_id=0,
            bos_id=1,
            eos_id=2,
            pad_id=3,
            num_threads=os.cpu_count(),
        )
        spm.SentencePieceTrainer.train(**options)
        self.load(f"{model_prefix}.model")

    def load(self, model_path):
        self.sp.load(model_path)

    def encode(self, text):
        ids = self.sp.encode(text, out_type=int)
        ids = [self.sp.bos_id()] + ids
        ids = ids + [self.sp.eos_id()]
        return ids

    def decode(self, ids):
        return self.sp.decode(ids)

    def pad(self, ids, max_len):
        pad_id = self.sp.pad_id()
        if len(ids) < max_len:
            return ids + [pad_id] * (max_len - len(ids))
        return ids[:max_len]

    def vocab_size(self):
        return self.sp.get_piece_size()