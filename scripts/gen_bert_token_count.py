import json
from transformers import BertTokenizer
from collections import Counter
import argparse
from tqdm import tqdm
import os
import pdb

parser = argparse.ArgumentParser("Compute Bert Token Counts")
parser.add_argument('--train_corpus',
                    default='data/visdial_1.0_train.json',
                    help='path of json file of the training corpus' 
                   )

parser.add_argument('--val_corpus',
                    default='data/visdial_1.0_val.json',
                    help='path of json file of the validation corpus' 
                   )

parser.add_argument('--test_corpus',
                    default='data/visdial_1.0_test.json',
                    help='path of json file of the test corpus' 
                   )
parser.add_argument('--out_path',
                    default='data/visdial_1.0_bert_token_count.json',
                    help='path of output token count json file.'
                   )
def parse_corpus(corpus_path, token2count):
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(
                f"Word counts do not exist at {corpus_path}"
              )
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    with open(corpus_path, 'r') as corpus_file:
        corpus_data = json.load(corpus_file)
        split = corpus_data["split"]
        if split in ['train', 'test2018', 'val2018']:
            '''
            Only valid to compute word count appearing in training corpus
            '''
            questions = corpus_data['data']['questions'] 
            answers = corpus_data['data']['answers'] 
            dialogs = corpus_data['data']['dialogs']
            print('Count {} question tokens...'.format(split))
            for i in tqdm(range(len(questions))):
                tokens = tokenizer.tokenize(questions[i] + "?")
                for tok in tokens:
                    token2count[tok] += 1
            print('Count {} answer tokens...'.format(split))
            for i in tqdm(range(len(answers))):
                tokens = tokenizer.tokenize(answers[i])
                for tok in tokens:
                    token2count[tok] += 1
            print('Count {} caption tokens...'.format(split))
            for i in tqdm(range(len(dialogs))):
                tokens = tokenizer.tokenize(dialogs[i]['caption'])
                for tok in tokens:
                    token2count[tok] += 1
        else:
            raise
    return token2count

if __name__ == "__main__":
    args = parser.parse_args()
    train_corpus = args.train_corpus
    val_corpus = args.val_corpus
    test_corpus = args.test_corpus
    train_t2c = parse_corpus(train_corpus, Counter())
    val_t2c = parse_corpus(val_corpus, Counter())
    test_t2c = parse_corpus(test_corpus, Counter())
    t2c = train_t2c + val_t2c + test_t2c
    with open(args.out_path, 'w') as out_file:
         json.dump(t2c, out_file)
    
