import torch
import numpy as np
import sys
import yaml
sys.path.append("..")
sys.path.append("../..")
from dense_att_enc import DenseCoAttEncoder
from visdialch.encoders import DenseEncoder
import argparse
import time
import pdb

parser = argparse.ArgumentParser('Dense Co-Attention Encoder Test')
parser.add_argument('--word_emb_dim',
                    type=int,
                    default=300,
                    help='Number of word embedding dimension.'
                   )
parser.add_argument('--lstm_hidden_dim',
                    type=int,
                    default=512,
                    help='Number of the lstm hidden dimension.'
                   )
parser.add_argument('--img_fea_dim',
                    type=int,
                    default=2048,
                    help='Number of the image feature dimension.'
                   )
parser.add_argument('--dropout',
                    type=float,
                    default=0.2,
                    help='dropout rate'
                   )
parser.add_argument('--sub_maps',
                    type=int,
                    default=4,
                    help='Number of slices in parallel attention step.'
                   )
parser.add_argument('--K',
                    type=int,
                    default=2,
                    help='Number of nowhere-to-attend positions.'
                   )
parser.add_argument('--stack_depth',
                    type=int,
                    default=2,
                    help='Number of dense-co-attention layers.'
                   )
parser.add_argument('--gpu-id',
                     type=int,
                     default=-1,
                     help='GPU ID on which the test running on'
                   )
parser.add_argument('--config-yml',
                    default='../../configs/dense_disc_faster_rcnn_x101.yml',
                    help='configuration file to test the DenseEncoder'
                  )


def main(args):
    batch = 12
    rounds = 10
    max_seq_len = 20
    region_num = 100

    config = {}
    config['word_emb_dim'] = args.word_emb_dim
    config['lstm_hidden_dim'] = args.lstm_hidden_dim
    config['img_fea_dim'] = args.img_fea_dim
    config['dropout'] = args.dropout
    config['K'] = args.K
    config['stack_depth'] = args.stack_depth

    device = torch.device("cuda", args.gpu_id) if args.gpu_id >= 0 else torch.device('cpu')
    torch.cuda.set_device(device)
    att_encoder = DenseCoAttEncoder(config)
    att_encoder = att_encoder.to(device)
    att_encoder.eval()

    img_fea = torch.randn(batch, region_num, args.img_fea_dim).to(device)
    dialog_content = torch.randn(batch, rounds, 2*max_seq_len, args.word_emb_dim).to(device)
    dialog_len = torch.randint(max_seq_len, 2*max_seq_len+1, (batch, rounds)).to(device)

    start = time.time()
    att_fea = att_encoder(dialog_content, dialog_len, img_fea, 'dialog')
    end = time.time()
    assert tuple(att_fea.size()) == (batch, rounds, 2*args.lstm_hidden_dim)
    print(("Pass dialog sanity check. The case of {}x{}x{}x{} word sequence, "
            "{}x{}x{} image feature inputs costs {}/ms."
            "").format(batch, rounds, 2*max_seq_len, args.word_emb_dim,
            batch, region_num, args.img_fea_dim, str((end-start)*1000)))

    del dialog_content
    question_content = torch.randn(batch, rounds, max_seq_len, args.word_emb_dim).to(device)
    question_len = torch.randint(max_seq_len//2, max_seq_len+1, (batch, rounds)).to(device)

    start = time.time()
    att_fea = att_encoder(question_content, question_len, img_fea, 'question')
    end = time.time()
    assert tuple(att_fea.size()) == (batch, rounds, 2*args.lstm_hidden_dim)
    print(("Pass question sanity check. The case of {}x{}x{}x{} word sequence, "
            "{}x{}x{} image feature inputs costs {}/ms."
            "").format(batch, rounds, max_seq_len, args.word_emb_dim,
            batch, region_num, args.img_fea_dim, str((end-start)*1000)))

    del img_fea
    del question_content

    batch = 12
    rounds = 10
    img_fea = torch.randn(batch, region_num, args.img_fea_dim).to(device)
    answer_content = torch.randn(batch, rounds, max_seq_len, args.word_emb_dim).to(device)
    answer_len = torch.randint(max_seq_len//2, max_seq_len+1, (batch, rounds)).to(device)

    start = time.time()
    att_fea = att_encoder(answer_content, answer_len, img_fea, 'answer')
    end = time.time()
    assert tuple(att_fea.size()) == (batch, rounds, 2*args.lstm_hidden_dim)
    print(("Pass answer sanity check. The case of {}x{}x{}x{} word sequence, "
            "{}x{}x{} image feature inputs costs {}/ms."
            "").format(batch, rounds, max_seq_len, args.word_emb_dim,
            batch, region_num, args.img_fea_dim, str((end-start)*1000)))

    del answer_content
    del att_fea
    #pdb.set_trace()
    batch = 12
    rounds = 10
    vocab = ['test'] * 1000
    config = yaml.load(open(args.config_yml))
    config['dataset']['use_glove'] = False
    config['model']['use_glove'] = False
    dense_enc = DenseEncoder(config['model'], vocab).to(device)
    data = {}
    img_fea = torch.randn(batch, region_num, config['model']['img_feature_size']).to(device)
    data['img_feat'] = img_fea
    data['ques'] = torch.randint(0, 999, (batch, rounds, config['dataset']['max_sequence_length'])).to(device)
    data['ques_len'] = torch.randint(1, config['dataset']['max_sequence_length'], (batch, rounds)).to(device)
    data['hist'] = torch.randint(0, 999, (batch, rounds, 2*config['dataset']['max_sequence_length'])).to(device)
    data['hist_len'] = torch.randint(1, 2*config['dataset']['max_sequence_length'], (batch, rounds)).to(device)
    start = time.time()
    img_QA_fea = dense_enc(data)
    end = time.time()
    assert tuple(img_QA_fea.size()) ==  (batch, rounds, config['model']['lstm_hidden_size'])
    print(("Pass DenseEncoder sanity check. The case of {}x{}x{} question sequence, "
            "{}x{}x{} dialog sequence, "
            "{}x{}x{} image feature inputs costs {}/ms."
            "").format(batch, rounds, config['dataset']['max_sequence_length'],
                       batch, rounds, 2*config['dataset']['max_sequence_length'],
                       batch, region_num, args.img_fea_dim, str((end-start)*1000)))

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
