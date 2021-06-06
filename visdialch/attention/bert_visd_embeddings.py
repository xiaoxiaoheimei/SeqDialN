import torch as t
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import pdb

class BertVisdEmbedding(nn.Module):
      '''
      The layer of generate Bert contextual representation
      '''

      def __init__(self, config=None, device=t.device("cpu")):
          '''
          Args:
            @config: configuration file of internal Bert layer
          '''
          super(BertVisdEmbedding, self).__init__()
          if config is None:
              self.bert = BertModel.from_pretrained('bert-base-uncased')
          else:
              self.bert = BertModel(config=config)# transformers correspondence
          self.device = device
          self.bert_hidden_size = self.bert.config.hidden_size
          tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
          self.CLS = tokenizer.convert_tokens_to_ids(['[CLS]'])[0] #ID of the Bert [CLS] token
          self.SEP = tokenizer.convert_tokens_to_ids(['[SEP]'])[0] #ID of the Bert [SEP] token
          self.PAD = tokenizer.convert_tokens_to_ids(['[PAD]'])[0] #ID of the Bert [PAD] token

      def make_bert_input(self, content_idxs, content_type, seg_ids):
          '''
          Args:
            @content_idxs (tensor): Bert IDs of the content. (batch_size, max_seq_len) Note that the max_seq_len is a fixed number due to padding/clamping policy.
            @content_type (str): whether the content is "question", "history" or "answer".
            @the initial segment ID: for "question" and "answer", this should be None; for 'history', this is should be well-initialized [0,..,0,1,...,1].
          Return:
            cmp_idx (tensor): [CLS] context_idxs [SEP]. (batch_size, max_seq_len+2)
            segment_ids (tensor): for "question" and "answer", this should be "1,1,...,1"; for "history", this should be "seg_ids[0], seg_ids, seg_ids[-1]". (batch_size, max_seq_len+2)
            input_mask (tensor): attention of the real token in content. Note [CLS] and [SEP] are count as real token. (batch_size, q_len + ctx_len + 2)
          '''
          mask = content_idxs != self.PAD #get the mask indicating the non-padding tokens in the content
          if content_type == 'question' or content_type == 'answer': #question/answer type
             seg_ids = t.zeros_like(content_idxs, dtype=content_idxs.dtype, device=content_idxs.device)
          
          seq_len = mask.sum(dim = 1) #(batch_size, ) length of each sequence
          batch_size, _ = content_idxs.size()
          content_idxs = t.cat((content_idxs, t.tensor([[self.PAD]]*batch_size, device=content_idxs.device)), dim=1) #(batch_size, max_seq_len+1)
          content_idxs[t.arange(0, batch_size), seq_len] = self.SEP #append [SEP] token to obtain "content_idxs [SEP]"
          seg_last = seg_ids[t.arange(0, batch_size), seq_len-1] #get the last segment id of each sequence
          seg_ids = t.cat((seg_ids, t.tensor([[0]]*batch_size, device=content_idxs.device)), dim=1)#(batch_size, max_seq_len+1)
          seg_ids[t.arange(0, batch_size), seq_len] = seg_last#the segment id of the new appended [SEP]
          content_idxs = t.cat((t.tensor([[self.CLS]]*batch_size, device=content_idxs.device), content_idxs), dim=1) #(batch_size, max_seq_len+2)append [CLS] token to obtain "[CLS] content_idxs [SEP]"
          seg_ids = t.cat((seg_ids[:,0].view(-1,1), seg_ids), dim=1) #(batch_size, max_seq_len+2) extend the first column of the segment id
          input_mask = (content_idxs != self.PAD).long() #(batch_size, max_seq_len+2)

          return content_idxs, seg_ids, input_mask

      def parse_bert_output(self, bert_output, orig_PAD_mask):
          '''
          Args:
            @bert_output (tensor): Bert output with [CLS] and [SEP] embeddings. (batch_size, 1+max_seq_len+1, bert_hidden_size) 
            @orig_PAD_mask (tensor): 1 for PAD token, 0 for non-PAD token. (batch_size, max_seq_len)
          Return:
            bert_enc (tensor): Bert output without [CLS] and [SEP] embeddings, and with zero-embedding for all PAD tokens. (batch_size, max_seq_len, bert_hidden_size)
          '''
          bert_enc = bert_output[:,1:-1] #(batch_size, max_seq_len, bert_hidden_size)
          pad_emb = t.zeros(self.bert_hidden_size, device=bert_output.device) #manually set the embedding of PAD token to be zero
          #print(bert_enc.size(), orig_PAD_mask.size(), pad_emb.size(), bert_enc.device, orig_PAD_mask.device, pad_emb.device)
          bert_enc = bert_enc.contiguous()
          bert_enc[orig_PAD_mask] = pad_emb #set the PAD token embeddings to be zero.
          return bert_enc

      def forward(self, content_idxs, content_type, seg_ids=None):
          '''
          Args:
            @content_idxs (tensor): Bert IDs of the contents. (batch_size, max_seq_len) Note that the max_seq_len is a fixed number due to padding/clamping policy
            @content_type (str): whether the tensor is "question", "history" or "answer"
          Return:
            bert_ctx_emb (tensor): contextual embedding condition on question. (batch_size, max_seq_len, bert_hidden_size)
          '''
          orig_PAD_mask = content_idxs == self.PAD
          cmp_idxs, segment_ids, bert_att = self.make_bert_input(content_idxs, content_type, seg_ids)
          outputs = self.bert(cmp_idxs, segment_ids, bert_att)
          bert_output = outputs[0]
          bert_enc = self.parse_bert_output(bert_output, orig_PAD_mask)
          return bert_enc

      def train(self, mode=True):
          '''
          Specifically set self.bert into training mode
          '''
          self.training = mode
          self.bert.train(mode)
          return self

      def eval(self):
          '''
          Specifically set self.bert into evaluation mode 
          '''
          return self.train(False)

      def to(self, *args, **kwargs):
          '''
          Override to() interface.
          '''
          print("bert emd to() called!")
          self = super().to(*args, **kwargs)
          self.bert = self.bert.to(*args, **kwargs)
          return self

 
