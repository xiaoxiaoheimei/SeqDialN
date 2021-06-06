SeqDialN
====================================

Code for reproducing results in our paper **SeqDialN: Sequential Visual Dialog Networks in Joint Visual-Linguistic Representation Space**.

  * [Setup and Dependencies](#setup-and-dependencies)
  * [Download Data](#download-data)
  * [Preprocess Data](#preprocess-data)
  * [Training](#training)
  * [Evaluation](#evaluation)
  * [Ensemble](#ensemble)
  * [Acknowledgements](#acknowledgements)

If you find this work is useful in your research, please kindly consider cite our paper:
```
@misc{yang2020seqdialn,
      title={SeqDialN: Sequential Visual Dialog Networks in Joint Visual-Linguistic Representation Space}, 
      author={Liu Yang and Fanqi Meng and Ming-Kuang Daniel Wu and Vicent Ying and Xianchao Xu},
      year={2020},
      eprint={2008.00397},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

Setup and Dependencies
----------------------

This code is implemented using PyTorch v1.2, and we recommend using Anaconda or Miniconda to setup the environment.

1. Install Anaconda or Miniconda distribution based on Python3+ from their [downloads' site][1].

2. Create a environment named `visdial` and install all dependencies with the `environment.yml` file.
```sh
conda env create -f environment.yml
conda activate visdial
```

Download Data
-------------

1. Download the VisDial v1.0 dialog json files: [training set][2], [validation set][3] and [test set][4].

2. Download the dense annotations for [validation set][5] and [training subset][6].

3. Download the word counts for VisDial v1.0 train split [here][7]. They are used to build the vocabulary.

4. Download pre-trained GloVe word vectors form [here][8] and unzip it. 

5. Download pre-extracted image features of VisDial v1.0 images, using a Faster-RCNN pre-trained on Visual Genome. Extracted features for v1.0 train, val and test are available for download at these links.

  * [`features_faster_rcnn_x101_train.h5`][9]: Bottom-up features of 36 proposals from images of `train` split.
  * [`features_faster_rcnn_x101_val.h5`][10]: Bottom-up features of 36 proposals from images of `val` split.
  * [`features_faster_rcnn_x101_test.h5`][11]: Bottom-up features of 36 proposals from images of `test` split.

6. Check all the files we needed and their location should as follow for default arguments to work effectively:
```
$PROJECT_ROOT/data/glove.840B.300d/glove.840B.300d.txt
$PROJECT_ROOT/data/features_faster_rcnn_x101_test.h5
$PROJECT_ROOT/data/features_faster_rcnn_x101_train.h5
$PROJECT_ROOT/data/features_faster_rcnn_x101_val.h5
$PROJECT_ROOT/data/visdial_1.0_test.json
$PROJECT_ROOT/data/visdial_1.0_train_dense_sample.json
$PROJECT_ROOT/data/visdial_1.0_train.json
$PROJECT_ROOT/data/visdial_1.0_val_dense_annotations.json
$PROJECT_ROOT/data/visdial_1.0_val.json
$PROJECT_ROOT/data/visdial_1.0_word_counts_train.json
```


Preprocess Data
---------------

### Generate token counts for DistillBERT
In order to use DistilBERT embeddiing, we should generate bert token counts json file used to build the vocabulary:
```
python scripts/gen_bert_token_count.py
```
The output json file should appear as `$PROJECT_ROOT/data/visdial_1.0_bert_token_count.json`.

### Extract training subset with adjusted dense annotations
Only subset of the training set have dense annotations. We should extract these subset and adjust the gt_relevance values to fine-tune with dense annotations using our re-weight method.
```
python scripts/extract_train.py --adjust-gt-relevance
```
Two new json files will appear as `$PROJECT_ROOT/data/visdial_1.0_train_dense_sub.json` and `$PROJECT_ROOT/data/visdial_1.0_train_dense_sample_adjusted.json`


Training
--------

### Base model training

To train the base model (no finetuning on dense annotations):
```sh
python train.py \
  --config-yml configs/disc_mrn_be.yml \
  --gpu-ids 0 \
  --cpu-workers 8 \
  --validate \
  --save-dirpath checkpoints/disc_mrn_be/
```
Train different type of models by passing different configuration file path to `--config-yml`. The model type and corresponding configuration files as show in the tables below:

Discriminative:

|   SeqIPN-GE-D   |   SeqIPN-BE-D   |   SeqMRN-GE-D   |   SeqMRN-BE-D   |
|:---------------:|:---------------:|:---------------:|:---------------:|
| disc_ipn_ge.yml | disc_ipn_be.yml | disc_mrn_ge.yml | disc_mrn_be.yml |

Generative:

|   SeqIPN-GE-G  |   SeqIPN-BE-G  |   SeqMRN-GE-G  |   SeqMRN-BE-G  |
|:--------------:|:--------------:|:--------------:|:--------------:|
| gen_ipn_ge.yml | gen_ipn_be.yml | gen_mrn_ge.yml | gen_mrn_be.yml |

Provide more ids to `--gpu-ids` to use multi-GPU execution. For example `--gpu-ids 0 1 2 3` will use 4 GPUs to train the model.

### Saving model checkpoints

This script will save model checkpoints at every epoch as per path specified by `--save-dirpath`. 

### Logging

We use Tensorboard for logging training progress. Execute 
```
tensorboard --logdir checkpoints/ --port 8008
```
 and visit `localhost:8008` in the browser.

### Fine-tune with dense annotations

To fine-tune the base model with dense annotations:
```
python train_stage2.py \
  --config-yml configs/disc_mrn_be_ft.yml \
  --gpu-ids 0 \
  --cpu-workers 8 \
  --validate \
  --load-pthpath checkpoints/disc_mrn_be/checkpoint_12.pth \
  --save-dirpath checkpoints/disc_mrn_be_ft/
```
You should specify the corresponding base model checkpoint path with `--load-pthpath`.

Both discriminative and generative base model could fine-tune with dense annotations, but fine-tuning can't help generative model boost NDCG much. Model type and corresponding fine-tune configuration files as show in the tables below:

Discriminative:

|     SeqIPN-GE-D    |     SeqIPN-BE-D    |     SeqMRN-GE-D    |     SeqMRN-BE-D    |
|:------------------:|:------------------:|:------------------:|:------------------:|
| disc_ipn_ge_ft.yml | disc_ipn_be_ft.yml | disc_mrn_ge_ft.yml | disc_mrn_be_ft.yml |

Generative:
|   SeqIPN-GE-G  |   SeqIPN-BE-G  |   SeqMRN-GE-G  |   SeqMRN-BE-G  |
|:--------------:|:--------------:|:--------------:|:--------------:|
| gen_ipn_ge_ft.yml | gen_ipn_be_ft.yml | gen_mrn_ge_ft.yml | gen_mrn_be_ft.yml |


Evaluation
----------

Evaluation of a trained model checkpoint can be done as follows:

```sh
python evaluate.py \
  --config-yml checkpoints/disc_mrn_be_ft/config.yml \
  --split val \
  --gpu-ids 0 \
  --cpu-workers 8 \
  --load-pthpath checkpoints/disc_mrn_be_ft/checkpoint_2.pth \
  --save-ranks-path results/ranks/disc_mrn_be_ft.json \
  --save-preds-path results/preds/disc_mrn_be_ft_preds.h5
```
This will report metrics form the Visual Dialog paper: R@{1, 5, 10}, Mean rank (mean), Mean reciprocal rank (MRR) and Normalized Discounted Cumulative Gain (NDCG).

If `--save-ranks-path` was specified, it will generate an EvalAI submission json file.

If `--save-preds-path` was specified, it will save the model's raw predict scores to a `.h5` file which could be used to ensemble models.

The metrics reported here would be the same as those reported through EvalAI by making a submission in `val` phase. 

To generate a submission file or raw predict results `.h5` file for `test-std` or `test-challenge` phase, replace `--split val` with `--split test`.

Ensemble
--------

In order to ensemble several models' predict resuls, you should evaluate these models seperatly using the evaluate.py script mentioned above and save each model's raw predict scores to a `.h5` file. Make sure these `.h5` file belong to same split (`val` or `test`) and in the same folder. Then:
```
python ensemble.py \
  --preds-folder results/preds/ \
  --split val \
  --method sa \
  --norm-order none \
  --save-ranks-path results/ranks/ensmble.json
```
This will search all `.h5` files in the folder specified in `--preds-folder` and ensemble the results using method specified in `--method`. Four ensemble method (`sa` for "Score Average", `pa` for "Probability Average", `ra` for "Rank Average" and `rra` for "Reciprocal Rank Average") support now.

`--norm-order` shold be `none` or a `int` number and this argument only work when `--method` is `sa`. When it is `none` we average different model's predict scores directly. When it is a `int` number we normlize the predict scores before average. For example, if `--norm-order` is `2`, we will normlize the predict scores using L2Norm before average.

For `val` split it will report all metrics mentioned above and `--save-ranks-path` is optinal. For `test` split you have to specify `--save-ranks-path` to save ensembled predict ranks to a json file.

Acknowledgements
----------------

This code began as a fork of [batra-mlp-lab/visdial-challenge-starter-pytorch][12].

[1]: https://conda.io/docs/user-guide/install/download.html
[2]: https://www.dropbox.com/s/ix8keeudqrd8hn8/visdial_1.0_train.zip?dl=0
[3]: https://www.dropbox.com/s/ibs3a0zhw74zisc/visdial_1.0_val.zip?dl=0
[4]: https://www.dropbox.com/s/o7mucbre2zm7i5n/visdial_1.0_test.zip?dl=0
[5]: https://www.dropbox.com/s/3knyk09ko4xekmc/visdial_1.0_val_dense_annotations.json?dl=0
[6]: https://www.dropbox.com/s/1ajjfpepzyt3q4m/visdial_1.0_train_dense_sample.json?dl=0
[7]: https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/visdial_1.0_word_counts_train.json
[8]: http://nlp.stanford.edu/data/glove.840B.300d.zip
[9]: https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_faster_rcnn_x101_train.h5
[10]: https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_faster_rcnn_x101_val.h5
[11]: https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/features_faster_rcnn_x101_test.h5
[12]: https://github.com/batra-mlp-lab/visdial-challenge-starter-pytorch

