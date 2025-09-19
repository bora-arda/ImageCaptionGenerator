import sentencepiece as spm
import os
import sentencepiece as spm
import os

def encode(sentencePieceProcessor, captions):
    return sentencePieceProcessor.encode(captions, out_type=int)
def decode(sentencePieceProcessor, ids):
    return sentencePieceProcessor.decode(ids, out_type=str)

options = dict(
    # Input Specs:
    input = '../data/raw/train_captions.txt',
    input_format = 'text',
    # Output Specs:
    model_prefix = 'caption_bpe',
    # Algorithm Specs:
    model_type = 'bpe',
    vocab_size = '6000',
    # Rare Word Treatment
    character_coverage = 0.99995,
    byte_fallback = True,
    # Normalization:
    normalization_rule_name = 'identity',
    remove_extra_whitespaces = False,
    input_sentence_size = 200000000,
    max_sentence_length = 4192,
    seed_sentencepiece_size = 1000000,
    shuffle_input_sentence = True,
    # Merge Rules:
    split_digits = True,
    split_by_unicode_script = True,
    split_by_whitespace = True,
    split_by_number = True,
    max_sentencepiece_length = 16,
    add_dummy_prefix = True,
    allow_whitespace_only_pieces = True,
    # Special Tokens:
    unk_id = 0,
    bos_id = 1,
    eos_id = 2,
    pad_id = 3,
    # Systems:
    num_threads = os.cpu_count() # Use all system resources
)
spm.SentencePieceTrainer.train(**options)

sp = spm.SentencePieceProcessor()