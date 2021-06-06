import torch
from transformers import BertTokenizer
from typing import List
import json
import os

class BertVocabulary(object):
      """
      A vocabulary class uses Bert tokenization. Bert has a vocabulary set consists of 30522 word pieces. That may be too 
      big to apply to the generation model. For flexible usage, we build our own tokenization system but maintain internal maps 
      to do multual conversion with Bert tokenization system.
      """
      PAD_TOKEN = "[PAD]" #bert
      SOS_TOKEN = "[S]"
      EOS_TOKEN = "[/S]"
      UNK_TOKEN = "[UNK]" #bert
      CLS_TOKEN = "[CLS]"
      SEP_TOKEN = "[SEP]"

      SOS_INDEX = 1 #bert unused
      EOS_INDEX = 2 #bert unused

      def __init__(self, token_counts_path: str, min_count: int = 2):
          if not os.path.exists(token_counts_path):
              raise FileNotFoundError(
                  f"Word counts do not exist at {token_counts_path}"
              )

          self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
          self.PAD_INDEX, self.UNK_INDEX, self.CLS_INDEX, self.SEP_INDEX = self.tokenizer.convert_tokens_to_ids(['[PAD]', '[UNK]', '[CLS]', '[SEP]'])
          with open(token_counts_path, "r") as token_counts_file:
              token_counts = json.load(token_counts_file)

              # form a list of (word, count) tuples and apply min_count threshold
              token_counts = [
                  (tok, count)
                  for tok, count in token_counts.items()
                  if count >= min_count
              ]
              # sort in descending order of word counts
              token_counts = sorted(token_counts, key=lambda tc: -tc[1])
              tokens = [tc[0] for tc in token_counts]
              bert_ids = self.tokenizer.convert_tokens_to_ids(tokens)
              self.max_bert_id = max(bert_ids)
          
          self.tok2id = {tok : bert_ids[i] for i, tok in enumerate(tokens)}
          self.tok2id[self.PAD_TOKEN] = self.PAD_INDEX
          self.tok2id[self.SOS_TOKEN] = self.SOS_INDEX
          self.tok2id[self.EOS_TOKEN] = self.EOS_INDEX
          self.tok2id[self.UNK_TOKEN] = self.UNK_INDEX
          self.tok2id[self.CLS_TOKEN] = self.CLS_INDEX
          self.tok2id[self.SEP_TOKEN] = self.SEP_INDEX

          self.id2tok = {
              bid: tok for tok, bid in self.tok2id.items()
          }

          self.bid2seqid = {
              list(self.id2tok.keys())[i]: i for i in range(len(self.id2tok))
          }
          self.PAD_SEQ_INDEX = self.bid2seqid[self.PAD_INDEX]

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
              cls.tok2id = json.load(saved_vocabulary_file)
          cls.id2tok = {
              bid: tok for tok, bid in cls.tok2id.items()
        }

      def to_indices(self, toks: List[str]) -> List[int]:
          return [self.tok2id.get(tok, self.UNK_INDEX) for tok in toks]

      def to_words(self, indices: List[int]) -> List[str]:
          return [
              self.id2tok.get(index, self.UNK_TOKEN) for index in indices
          ]

      def save(self, save_vocabulary_path: str) -> None:
          with open(save_vocabulary_path, "w") as save_vocabulary_file:
              json.dump(self.tok2id, save_vocabulary_file)

      def __len__(self):
          return len(self.tok2id)

