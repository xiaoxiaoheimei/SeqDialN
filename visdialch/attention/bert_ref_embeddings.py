import torch
from torch import nn
from transformers import BertTokenizer, BertModel
import pdb

class BertRefEmbedding(nn.Module):
      '''
      The layer of generate Bert contextual representation
      '''

      def __init__(self, bert_ref_model):
          '''
          Args:
            @bert_ref_model: A custom bert model which the first embedding layer will be referenced in this object.
          '''
          super(BertRefEmbedding, self).__init__()
          bert_layer0 = list(bert_ref_model.children())[0]
          self.bert_word_embed = list(bert_layer0.children())[0]
          tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
          self.PAD = tokenizer.convert_tokens_to_ids(['[PAD]'])[0] #ID of the Bert [PAD] token
          self.bert_hidden_size = bert_ref_model.config.hidden_size

      def forward(self, content_idxs):
          '''
          Args:
            @content_idxs (tensor): Bert IDs of the contents. (batch_size, max_seq_len) Note that the max_seq_len is a fixed number due to padding/clamping policy
          Return:
            bert_emb (tensor): contextual embedding condition on question. (batch_size, max_seq_len, bert_hidden_size)
          '''
          #pdb.set_trace()
          orig_PAD_mask = content_idxs == self.PAD
          pad_emb = torch.zeros(self.bert_hidden_size, device=content_idxs.device)
          bert_emb = self.bert_word_embed(content_idxs)#(batch_size, max_seq_len, bert_hidden_size)
          bert_emb[orig_PAD_mask] = pad_emb ##manually set the embedding of PAD token to be zero
          return bert_emb

 
