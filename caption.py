"""
Module for loading, captioning, and saving images.
"""

import pickle
from keras.preprocessing.text import Tokenizer

def idx_to_word(integer: int, tokenizer_arg: Tokenizer) -> str | None:
    """
    Given an integer, return the corresponding word in the tokenizer's word index.
    ex: idx_to_word(1,tokenizer) -> 'startseq'
    """
    for word, index in tokenizer_arg.word_index.items():
        if index==integer:
            return word
    return None

def load_tokenizer() -> Tokenizer:
    """
    Load the tokenizer from the file.
    """
    with open('./saves/tokenizer.pickle', 'rb') as handle:
        tokenizer: Tokenizer = pickle.load(handle)
    return tokenizer
