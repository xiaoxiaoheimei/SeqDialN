import torch
import numpy as np
import sys
import yaml
import torch.nn as nn
sys.path.append("..")
sys.path.append("../..")
from dense_att_enc import DenseCoAttEncoder
from visdialch.encoders import DenseEncoder
from visdialch.data.bert_dataset import BertVisDialDataset
from torch.utils.data import DataLoader
from bert_visd_embeddings import BertVisdEmbedding
from distilbert_visd_embeddings import DistilBertVisdEmbedding
from bert_ref_embeddings import BertRefEmbedding
from transformers import BertConfig
import json

import argparse
import time
import pdb

parser = argparse.ArgumentParser('Bert Dataset/Embedding Test')
parser.add_argument('--gpu-ids',
                     nargs="+",
                     type=int,
                     default=-1,
                     help='GPU ID on which the test running on'
                   )
parser.add_argument('--config-yml',
                    default='../../configs/dense_disc_faster_rcnn_x101.yml',
                    help='configuration file to test the DenseEncoder'
                  )
parser.add_argument('--data-json',
                    default='../../data/visdial_1.0_train.json',
                    help='data file'
                  )
parser.add_argument('--overfit',
                    action='store_true',
                    help='whether generate overfit dataset.'
                  )
parser.add_argument(
                   "--in-memory",
                   action="store_true",
                   help="Load the whole dataset and pre-extracted image features in memory. "
                   "Use only in presence of large RAM, atleast few tens of GBs.",
)

parser.add_argument(
                    "--val-dense-json",
                    default="../../data/visdial_1.0_val_dense_annotations.json",
                    help="Path to json file containing VisDial v1.0 validation dense ground "
                    "truth annotations.",
)



def test_dataset(args):
    config = yaml.load(open(args.config_yml))
    dataset = BertVisDialDataset(
                config["dataset"],
                args.data_json,
                args.val_dense_json,
                overfit=args.overfit,
                in_memory=args.in_memory,
                return_options=True if config["model"]["decoder"] == "disc" else False,
                add_boundary_toks=False if config["model"]["decoder"] == "disc" else True,
              )
    dataloader = DataLoader(
                  dataset,
                  batch_size=config["solver"]["batch_size"],
                  num_workers=4,
                  shuffle=True,
                )
    data = {"batch{}".format(i): batch for i, batch in enumerate(dataloader) if i < 5}
    torch.save(data, 'test_data.pt')

def test_ref_emb(args):
    visd_emb = DistilBertVisdEmbedding()
    visd_emb.eval()
    dataset = torch.load("test_data.pt")
    ref_emb = BertRefEmbedding(visd_emb.bert)
    model = nn.ModuleList([visd_emb, ref_emb])
    model = model.to(torch.device("cuda:0"))
    '''
    if -1 not in args.gpu_ids:
       model = nn.DataParallel(model, args.gpu_ids)
    '''
    for idx, batch in dataset.items():
        answer_idx = batch["opt"][0:32] 
        answer_idx = answer_idx.to("cuda:0")
        batch, rounds, options, seq_len = answer_idx.size()
        answer_idx1 = answer_idx.view(-1, seq_len)
        ans_emb = model[1](answer_idx1)
        ans_emb = ans_emb.view(batch, rounds, options, seq_len, -1)
        ans_pad_mask = answer_idx == 0
        assert ans_emb[ans_pad_mask].abs().sum().item() == 0.
        print("Pass answer embeding value-sanity check for batch_{}".format(idx))

def test_bert_emb(args):
    pdb.set_trace()
    config = yaml.load(open(args.config_yml))
    dataset = torch.load("test_data.pt")
    config1 = BertConfig(vocab_size_or_config_json_file=30522, num_hidden_layers=3, hidden_size=192, num_attention_heads=3, intermediate_size=384)
    model = DistilBertVisdEmbedding()
    model = model.to(torch.device("cuda:0"))
    if -1 not in args.gpu_ids:
       model = nn.DataParallel(model, args.gpu_ids)
    model.eval()
    for idx, batch in dataset.items():
      ques_idx = batch['ques'][0:32]
      hist_idx = batch['hist'][0:32]
      hist_seg_idx = batch['hist_seg'][0:32]
      answer_idx = batch['opt'][0:16]
      ques_idx = ques_idx.to("cuda:0")
      hist_idx = hist_idx.to("cuda:0")
      hist_seg_idx = hist_seg_idx.to("cuda:0")

      batch, rounds, seq_len = ques_idx.size()
      ques_idx1 = ques_idx.view(-1, seq_len)
      ques_emb = model(ques_idx1, "question")
      ques_emb = ques_emb.view(batch, rounds, seq_len, -1)
      assert tuple(ques_idx.size()) == tuple(ques_emb.size())[0:3]
      print("Pass question embedding shape-sanity check for batch_{}.".format(idx))
      ques_pad_mask = ques_idx == 0
      ques_pad_emb = ques_emb[ques_pad_mask]
      assert ques_pad_emb.abs().sum().item() == 0.
      print("Pass question embedding value-sanity check for batch_{}.".format(idx))
      del ques_pad_mask
      del ques_pad_emb
      batch, rounds, seq_len = hist_idx.size()
      hist_idx1 = hist_idx.view(-1, seq_len)
      hist_seg_idx1 = hist_seg_idx.view(-1, seq_len)
      hist_emb = model(hist_idx1, "history", hist_seg_idx1)
      hist_emb = hist_emb.view(batch, rounds, seq_len, -1)
      assert tuple(hist_idx.size()) == tuple(hist_emb.size())[0:3]
      print("Pass history embedding shape-sanity check for batch_{}.".format(idx))
      hist_pad_mask = hist_idx == 0
      hist_pad_emb = hist_emb[hist_pad_mask]
      assert hist_pad_emb.abs().sum().item() == 0.
      print("Pass history embeding value-sanity check for batch_{}.".format(idx))
      del hist_pad_mask
      del hist_pad_emb
      del ques_idx
      del hist_idx
      del hist_seg_idx
      del ques_idx1
      del hist_idx1
      del hist_seg_idx1
      del ques_emb
      del hist_emb
      torch.cuda.empty_cache()
      continue
      slice_num = 100
      answer_idx = answer_idx.to("cuda:0").contiguous()
      batch, rounds, options, seq_len = answer_idx.size()
      answer_pad_mask = answer_idx == 0
      for i in range(1):
          answer_idx_t = answer_idx[:,:,i*slice_num:(i+1)*slice_num].contiguous().view(-1, seq_len)
          answer_emb_t = model(answer_idx_t, "answer")
          answer_emb_t = answer_emb_t.view(batch, rounds, slice_num, seq_len, -1)
          if i == 0:
             answer_emb = answer_emb_t
          else:
             answer_emb = torch.cat((answer_emb, answer_emb_t), dim=2)
          del answer_emb_t
          del answer_idx_t
          torch.cuda.empty_cache()
      answer_pad_emb = answer_emb[answer_pad_mask[:,:,0:slice_num]]
      assert answer_pad_emb.abs().sum().item() == 0.
      print("Pass answer embedding value-sanity check for batch_{}.".format(idx))



if __name__ == "__main__":
    args = parser.parse_args()
    test_dataset(args)
    test_bert_emb(args)
    test_ref_emb(args)
