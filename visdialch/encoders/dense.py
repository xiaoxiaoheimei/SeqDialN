import torch
from torch import nn
from torch.nn import functional as F

from visdialch.attention import DenseCoAttEncoder
from visdialch.attention import BertVisdEmbedding, DistilBertVisdEmbedding
from visdialch.attention import VisdTransformer
from visdialch.utils import DynamicRNN
from visdialch.fusions.factory import factory as factory_fusion


class DenseEncoder(nn.ModuleList):
    def __init__(self, config, vocabulary):
        super().__init__()
        self.config = config
        if config['word_embedding_type'] == 'glove':
           self.word_embed = nn.Embedding.from_pretrained(vocabulary.get_vocab_emb_tensors())
        elif config['word_embedding_type'] == 'bert':
           self.word_embed = DistilBertVisdEmbedding()
        else:
           self.word_embed = nn.Embedding(
              len(vocabulary),
              config["word_embedding_size"],
              padding_idx=vocabulary.PAD_INDEX,
           )

        self.DCAEncoder = DenseCoAttEncoder(config)
        self.fea_dim = self.DCAEncoder.get_feature_dim()

        self.reason_mode = config['reason_mode']
        if self.reason_mode == 'transformer':
            trans_config = config['transformer']
            self.reason_trans = VisdTransformer(config=trans_config)

        # We use DCAEncoder to encode both question and dialog and then concat them together, 
        # we use this project to project the 2*fea_dim vector to fea_dim feature
        self.question_dialog_projection = nn.Linear(
            2*self.fea_dim, config['decoder_lstm_hidden_size']
        )

        # collect batch details when dump_details is set to True
        self.save_details = config.get('save_details', False)
        print(f"encoder save_details is {self.save_details}")

        self.details = [] if self.save_details else None

    def forward(self, batch):
        # shape: (batch_size, img_feature_size) - CNN fc7 features
        # shape: (batch_size, num_proposals, img_feature_size) - RCNN features
        img = batch["img_feat"]
        # shape: (batch_size, 10, max_sequence_length)
        ques = batch["ques"]
        # shape: (batch_size, 10, max_sequence_length * 2)                     
        # concatenated qa * 10 rounds
        dialog = batch["hist"]
        if self.config['word_embedding_type'] == 'bert':
            dialog_seg_id = batch["hist_seg"]
        # num_rounds = 10, even for test (padded dialog rounds at the end)
        batch_size, num_rounds, max_sequence_length = ques.size()

        # embed questions
        ques = ques.view(batch_size * num_rounds, max_sequence_length)
        if self.config['word_embedding_type'] == 'bert':
          ques_embed = self.word_embed(ques, 'question') #(batch_size*num_rounds, max_sequence_length, word_emb_size)
        else:
          ques_embed = self.word_embed(ques) #(batch_size*num_rounds, max_sequence_length, word_emb_size)
        ques_embed = ques_embed.view(batch_size, num_rounds, max_sequence_length, -1)
  
        #embed history
        dialog = dialog.view(batch_size*num_rounds, -1)
        if self.config['word_embedding_type'] == 'bert':
          dialog_embed = self.word_embed(dialog, 'history', dialog_seg_id.view(batch_size*num_rounds, -1)) #(batch_size*num_rounds, 2*max_sequence_length, word_emb_size)
        else:
          dialog_embed = self.word_embed(dialog) #(batch_size*num_rounds, 2*max_sequence_length, word_emb_size)
        dialog_embed = dialog_embed.view(batch_size, num_rounds, dialog.size(-1), -1)

        ques_len = batch['ques_len'].view(batch_size, num_rounds)
        ques_img_fea = self.DCAEncoder(ques_embed, ques_len, img, 'question') #get the question/image co-attended feature. (batch_size, num_rounds, self.fea_dim)

        dialog_len = batch['hist_len'].view(batch_size, num_rounds)
        dialog_img_fea = self.DCAEncoder(dialog_embed, dialog_len, img, 'dialog') #get the history/image co-attended featuer. (batch_size, num_rounds, self.fea_dim)
        if self.reason_mode == 'transformer':
            dialog_img_fea, ques_img_fea, q2dq_w = self.reason_trans(dialog_img_fea, ques_img_fea)

        if self.reason_mode == 'transformer' and self.save_details:
            for img_id, one_q2dq_w in zip(batch["img_ids"], q2dq_w):
                self.details.append(
                    {
                        "image_id": img_id.item(),
                        "q2dq_w": one_q2dq_w.tolist()
                    }
                )

        # Concat the question and dialog features to obtain a 2*self.fea_dim
        question_dialog_fea = torch.cat((ques_img_fea, dialog_img_fea), -1) #(batch_size, num_rounds, 2*self.fea_dim)

        image_QA_fea = self.question_dialog_projection(question_dialog_fea) #(batch_size, num_rounds, lstm_hidden_size)

        return image_QA_fea

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.DCAEncoder.to(*args, **kwargs)
        if self.config['word_embedding_type'] == 'bert':
            self.word_embed.to(*args, **kwargs)
        return self

