"""
A Vocabulary maintains a mapping between words and corresponding unique
integers, holds special integers (tokens) for indicating start and end of
sequence, and offers functionality to map out-of-vocabulary words to the
corresponding token.
"""
import json
import os
from typing import List
from tqdm import tqdm
import numpy as np
import torch
import pickle 


class GloveVocabulary(object):
    """
    A simple Vocabulary class which maintains a mapping between words and
    integer tokens. Can be initialized either by word counts from the VisDial
    v1.0 train dataset, or a pre-saved vocabulary mapping.

    Parameters
    ----------
    word_counts_path: str
        Path to a json file containing counts of each word across captions,
        questions and answers of the VisDial v1.0 train dataset.
    min_count : int, optional (default=0)
        When initializing the vocabulary from word counts, you can specify a
        minimum count, and every token with a count less than this will be
        excluded from vocabulary.
    """

    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<S>"
    EOS_TOKEN = "</S>"
    UNK_TOKEN = "<UNK>"

    PAD_INDEX = 0
    SOS_INDEX = 1
    EOS_INDEX = 2
    UNK_INDEX = 3

    def __init__(self, word_counts_path: str, min_count: int, glove_weight_path: str, vec_size: int, glove_vec_num: int):
        if not os.path.exists(word_counts_path):
            raise FileNotFoundError(
                f"Word counts do not exist at {word_counts_path}"
            )
        if not os.path.exists(glove_weight_path):
            raise FileNotFoundError(
               f"Glove weight do not exist at {glove_weight_path}"
            )

        with open(word_counts_path, "r") as word_counts_file:
            word_counts = json.load(word_counts_file)

            # form a list of (word, count) tuples and apply min_count threshold
            word_counts = [
                (word, count)
                for word, count in word_counts.items()
                if count >= min_count
            ]
            # sort in descending order of word counts
            word_counts = sorted(word_counts, key=lambda wc: -wc[1])
            words = [w[0] for w in word_counts]

        golve_temp_path = os.path.join(
            os.path.dirname(glove_weight_path),
            os.path.basename(word_counts_path) + "_glove_temp.pkl"
        )
        if os.path.exists(golve_temp_path):
            with open(golve_temp_path, 'rb') as glove_temp_pkl:
                self.word2emb = pickle.load(glove_temp_pkl)
            print('Loaded pretrained glove weight temp from', golve_temp_path)
        else:
            self.word2emb = {}
            print("Reading in pretrained Glove weight...")
            with open(glove_weight_path, 'r', encoding='utf-8') as glove_weight_file:
                for line in tqdm(glove_weight_file, total=glove_vec_num):
                    array = line.split()
                    word = "".join(array[0:-vec_size]) 
                    emb = list(map(float, array[-vec_size:]))
                    if word in words:
                        self.word2emb[word] = emb
            with open(golve_temp_path, 'wb') as glove_temp_pkl:
                pickle.dump(self.word2emb, glove_temp_pkl)
            print('Saved pretrained glove weight temp to', golve_temp_path)

        self.word2index = {}
        self.word2index[self.PAD_TOKEN] = self.PAD_INDEX
        self.word2index[self.SOS_TOKEN] = self.SOS_INDEX
        self.word2index[self.EOS_TOKEN] = self.EOS_INDEX
        self.word2index[self.UNK_TOKEN] = self.UNK_INDEX
        for index, word in enumerate(self.word2emb.keys()):
            self.word2index[word] = index + 4

        #Add embedding for special words
        self.word2emb[self.PAD_TOKEN] = np.array([0.]*vec_size)
        self.word2emb[self.SOS_TOKEN] = np.array([0.]*vec_size)
        self.word2emb[self.EOS_TOKEN] = np.array([0.]*vec_size)
        self.word2emb[self.UNK_TOKEN] = np.array([0.]*vec_size)

        self.index2word = {
            index: word for word, index in self.word2index.items()
        }

    @classmethod
    def from_saved(cls, saved_vocabulary_path: str) -> "Vocabulary":
        """Build the vocabulary from a json file saved by ``save`` method.

        Parameters
        ----------
        saved_vocabulary_path : str
            Path to a json file containing word to integer mappings
            (saved vocabulary).
        """
        with open(saved_vocabulary_path, "r") as saved_vocabulary_file:
            cls.word2index = json.load(saved_vocabulary_file)
        cls.index2word = {
            index: word for word, index in cls.word2index.items()
        }

    def to_indices(self, words: List[str]) -> List[int]:
        return [self.word2index.get(word, self.UNK_INDEX) for word in words]

    def to_words(self, indices: List[int]) -> List[str]:
        return [
            self.index2word.get(index, self.UNK_TOKEN) for index in indices
        ]

    def save(self, save_vocabulary_path: str) -> None:
        with open(save_vocabulary_path, "w") as save_vocabulary_file:
            json.dump(self.word2index, save_vocabulary_file)

    def get_vocab_emb_tensors(self):
        '''
        return: pretrained weight in torch tensor.
        '''
        embs = np.array([self.word2emb[self.index2word[i]] for i in range(len(self.index2word))], dtype='float32') 
        return torch.from_numpy(embs)

    def __len__(self):
        return len(self.index2word)
