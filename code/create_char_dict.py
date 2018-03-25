''' This code is used to create char dictionary using dataset'''
from __future__ import absolute_import
from __future__ import division
import os
import pickle
from tqdm import tqdm

_PAD = b"<pad>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _UNK]
PAD_ID = 0
UNK_ID = 1
PER_CUT = 0.01

MAIN_DIR = os.path.relpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # relative path of the main directory
DEFAULT_DATA_DIR = os.path.join(MAIN_DIR, "data") # relative path of data dir
EXPERIMENTS_DIR = os.path.join(MAIN_DIR, "experiments") # relative path of experiments dir

train_context_path = os.path.join(DEFAULT_DATA_DIR, "train.context")
train_qn_path = os.path.join(DEFAULT_DATA_DIR, "train.question")
dev_context_path = os.path.join(DEFAULT_DATA_DIR, "dev.context")
dev_qn_path = os.path.join(DEFAULT_DATA_DIR, "dev.question")

char_dict_pickle_file = os.path.join(os.path.join(MAIN_DIR, "code"), "chardict.p")

def get_char_dict(*args):
    """Creates Character Dictionary based on frequency stats and dumps to pickle.

    Input:
      *args: dataset filepaths

    Returns:
      char2id, id2char, char_vocab_size, max_word_len
    """

    print "Creating Character Mappings from training and dev sets..."
    txt = ""
    max_word_len = 0
    for arg in args:
        text_file = open(arg)
        # context_line, qn_line, ans_line = text_file.readline()
        for line in text_file:
            txt += line
            words = line.split()
            max_len = max([len(w) for w in words])
            if max_len>max_word_len: max_word_len = max_len
    # Drop low frequency chars
    char_frequency_dict = char_frequency(txt)
    total = sum(char_frequency_dict.values())
    for key in char_frequency_dict.keys():
        if char_frequency_dict[key]/total<PER_CUT/100:
            char_frequency_dict.pop(key)
    chars = char_frequency_dict.keys()
    chars.sort()
    chars = _START_VOCAB + chars
    char_vocab_size = len(chars)
    char2id = dict((c, i) for i, c in enumerate(chars))
    id2char = dict((i, c) for i, c in enumerate(chars))
    print('Total number of unique chars: {}. Maximum Length: {}'.format(char_vocab_size, max_word_len))
    print("char2d dict:", char2id)
    print("id2char dict:", id2char)
    pickle.dump([char2id, id2char, char_vocab_size], open(char_dict_pickle_file, "wb"))
    return char2id, id2char, char_vocab_size, max_word_len

def char_frequency(str1):
    """ Calculates frequency of characters in a string and returns a dictionary"""
    dict = {}
    for n in tqdm(str1):
        keys = dict.keys()
        if n in keys:
            dict[n] += 1
        else:
            dict[n] = 1
    return dict

if __name__ == "__main__":
    char2id, id2char, char_vocab_size , _ = get_char_dict(train_context_path, train_qn_path, dev_context_path, dev_qn_path)