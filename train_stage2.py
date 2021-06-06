import argparse
import itertools
import math
import os 
import logging 

from tensorboardX import SummaryWriter
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from bisect import bisect
from pathlib import Path 

from visdialch.data.dataset import VisDialDataset
from visdialch.data.bert_dataset import BertVisDialDataset
from visdialch.encoders import Encoder
from visdialch.decoders import Decoder
from visdialch.metrics import SparseGTMetrics, NDCG
from visdialch.model import EncoderDecoderModel
from visdialch.utils.checkpointing import CheckpointManager, load_checkpoint
from visdialch.loss import NpairLoss, GeneralizedCrossEntropyLoss

from transformers import AdamW, WarmupCosineWithHardRestartsSchedule
from transformers import WarmupCosineSchedule, WarmupLinearSchedule 

import pdb


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config-yml",
    default="configs/dense_disc_glove_lstm_stage2.yml",
    help="Path to a config file listing reader, model and solver parameters.",
)
parser.add_argument(
    "--train-json",
    default="data/visdial_1.0_train_dense_sub.json",
    help="Path to json file containing VisDial v1.0 training data.",
)
parser.add_argument(
    "--train-dense-json",
    default="data/visdial_1.0_train_dense_sample_adjusted.json",
    help="Path to json file containing VisDial v1.0 training dense ground "
    "truth annotations.",
)
parser.add_argument(
    "--train2-json",
    default="",
    help="Path to json file containing the second part of the training data "
    "that if used."
)
parser.add_argument(
    "--train2-dense-json",
    default="",
    help="Path to json file containing the second part of training annotations"
         " if used."
)
parser.add_argument(
    "--val-json",
    default="data/visdial_1.0_val.json",
    help="Path to json file containing VisDial v1.0 validation data.",
)
parser.add_argument(
    "--val-dense-json",
    default="data/visdial_1.0_val_dense_annotations.json",
    help="Path to json file containing VisDial v1.0 validation dense ground "
    "truth annotations.",
)


parser.add_argument_group(
    "Arguments independent of experiment reproducibility"
)
parser.add_argument(
    "--gpu-ids",
    nargs="+",
    type=int,
    default=0,
    help="List of ids of GPUs to use.",
)
parser.add_argument(
    "--cpu-workers",
    type=int,
    default=4,
    help="Number of CPU workers for dataloader.",
)
parser.add_argument(
    "--overfit",
    action="store_true",
    help="Overfit model on 5 examples, meant for debugging.",
)
parser.add_argument(
    "--validate",
    action="store_true",
    help="Whether to validate on val split after every epoch.",
)
parser.add_argument(
    "--validate-train",
    action="store_true",
    help="Whether to validate on train split after every epoch.",
)
parser.add_argument(
    "--in-memory",
    action="store_true",
    help="Load the whole dataset and pre-extracted image features in memory. "
    "Use only in presence of large RAM, atleast few tens of GBs.",
)


parser.add_argument_group("Checkpointing related arguments")
parser.add_argument(
    "--load-pthpath",
    default="",
    help="To continue training, path to .pth file of saved checkpoint."
    "In stage2, this should be the stage1 pre-trained model checkpoint.",
)
parser.add_argument(
    "--save-dirpath",
    default="checkpoints/",
    help="Path of directory to create checkpoint directory and save "
    "checkpoints.",
)


# For reproducibility.
# Refer https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# =============================================================================
#   INPUT ARGUMENTS AND CONFIG
# =============================================================================

args = parser.parse_args()

def prepare_logger(dirpath):
    Path(dirpath).mkdir(parents=True, exist_ok=True)
    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(os.path.join(dirpath, 'log.txt'), mode='w')

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(message)s')
    f_format = logging.Formatter('%(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

logger = prepare_logger(args.save_dirpath)

# keys: {"dataset", "model", "solver"}
config = yaml.load(open(args.config_yml))

if isinstance(args.gpu_ids, int):
    args.gpu_ids = [args.gpu_ids]
device = (
    torch.device("cuda", args.gpu_ids[0])
    if args.gpu_ids[0] >= 0
    else torch.device("cpu")
)

# Print config and args.
logger.info(yaml.dump(config, default_flow_style=False))
for arg in vars(args):
    logger.info("{:<20}: {}".format(arg, getattr(args, arg)))


# =============================================================================
#   SETUP DATASET, DATALOADER, MODEL, CRITERION, OPTIMIZER, SCHEDULER
# =============================================================================

word_embedding_type = config['dataset']['word_embedding_type']
if word_embedding_type not in ['init', 'glove', 'bert']:
    raise NotImplementedError()
logger.info('Word embedding type: {}'.format(word_embedding_type))

if word_embedding_type == 'bert':
    train_dataset = BertVisDialDataset(
        config["dataset"],
        args.train_json,
        args.train_dense_json,
        return_adjusted_gt_relevance=config["dataset"]["use_adjusted"],
        overfit=args.overfit,
        in_memory=args.in_memory,
        return_options=True,
        add_boundary_toks=config["model"]["decoder"] == "gen",
        proj_to_senq_id=config["model"]["decoder"] == "gen"
    )
    val_dataset = BertVisDialDataset(
        config["dataset"],
        args.val_json,
        args.val_dense_json,
        overfit=args.overfit,
        in_memory=args.in_memory,
        return_options=True,
        add_boundary_toks=config["model"]["decoder"] == "gen",
        proj_to_senq_id=config["model"]["decoder"] == "gen"
    )
else:
    train_dataset = VisDialDataset(
        config["dataset"],
        args.train_json,
        args.train_dense_json,
        return_adjusted_gt_relevance=config["dataset"]["use_adjusted"],
        overfit=args.overfit,
        in_memory=args.in_memory,
        return_options=True,
        add_boundary_toks=config["model"]["decoder"] == "gen"
    )
    val_dataset = VisDialDataset(
        config["dataset"],
        args.val_json,
        args.val_dense_json,
        overfit=args.overfit,
        in_memory=args.in_memory,
        return_options=True,
        add_boundary_toks=config["model"]["decoder"] == "gen"
    )

assert((config["solver"]['batch_size'] 
        % config["solver"]["accumulation_steps"]) == 0)
actual_batch_size = (config["solver"]['batch_size'] 
                     // config["solver"]["accumulation_steps"])
logger.info('Actual batch size: {}'.format(actual_batch_size))
logger.info('Gradient accumulation steps: {}'.format(config['solver']['accumulation_steps']))
logger.info('Effective batch size: {}'.format(config['solver']['batch_size']))

train_dataloader = DataLoader(
    train_dataset,
    batch_size=actual_batch_size,
    num_workers=args.cpu_workers,
    shuffle=True,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=actual_batch_size
    if config["model"]["decoder"] == "disc"
    else len(args.gpu_ids),
    num_workers=args.cpu_workers,
)
logger.info('Training data use adjusted gt_relavance: {}'.format(config["dataset"]["use_adjusted"]))

if args.train2_json != "":
    if word_embedding_type == 'bert':
        train2_dataset = BertVisDialDataset(
            config['dataset'],
            args.train2_json,
            args.train2_dense_json,
            return_adjusted_gt_relevance=config["dataset"]["use_adjusted"],
            overfit=args.overfit,
            in_memory=args.in_memory,
            return_options=config["model"]["decoder"] == "disc",
            add_boundary_toks=config["model"]["decoder"] == "gen",
            proj_to_senq_id=config["model"]["decoder"] == "gen"
        )
    else:
        train2_dataset = VisDialDataset(
            config['dataset'],
            args.train2_json,
            args.train2_dense_json,
            return_adjusted_gt_relevance=config["dataset"]["use_adjusted"],
            overfit=args.overfit,
            in_memory=args.in_memory,
            return_options=config["model"]["decoder"] == "disc",
            add_boundary_toks=config["model"]["decoder"] == "gen"
        )
    train2_dataloader = DataLoader(
        train2_dataset,
        batch_size=actual_batch_size,
        num_workers=args.cpu_workers,
        shuffle=True,
    )
    logger.info('Training data 2 use adjusted gt_relavance: {}'.format(config["dataset"]["use_adjusted"]))


# Pass vocabulary to construct Embedding layer.
encoder = Encoder(config["model"], train_dataset.vocabulary)
if word_embedding_type == 'bert':
    decoder = Decoder(
        config["model"], train_dataset.vocabulary, 
        bert_model=encoder.word_embed.bert,
        stage=2
    )
else:
    decoder = Decoder(
        config["model"], train_dataset.vocabulary,
        stage=2
    )
logger.info("Encoder: {}".format(config["model"]["encoder"]))
logger.info("Decoder: {}".format(config["model"]["decoder"]))

# Share word embedding between encoder and decoder.
if not word_embedding_type == 'bert':
    decoder.word_embed = encoder.word_embed

# Wrap encoder and decoder in a model.
model = EncoderDecoderModel(encoder, decoder).to(device)
if -1 not in args.gpu_ids:
    model = nn.DataParallel(model, args.gpu_ids)

# Loss function.
if config['model']['loss'] == 'gce':
    criterion = GeneralizedCrossEntropyLoss()
else:
    raise NotImplementedError


if config["solver"]["training_splits"] == "trainval":
    data_iterations = len(train_dataloader) + len(val_dataloader)
else:
    data_iterations = len(train_dataloader)
if args.train2_json != "":
    data_iterations += len(train2_dataloader)
iterations = int(math.ceil(data_iterations / config['solver']['accumulation_steps']))


def lr_lambda_fun(current_iteration: int) -> float:
    """Returns a learning rate multiplier.

    Till `warmup_epochs`, learning rate linearly increases to `initial_lr`,
    and then gets multiplied by `lr_gamma` every time a milestone is crossed.
    """
    current_epoch = float(current_iteration) / iterations
    if current_epoch <= config["solver"]["warmup_epochs"]:
        alpha = current_epoch / float(config["solver"]["warmup_epochs"])
        return config["solver"]["warmup_factor"] * (1.0 - alpha) + alpha
    else:
        idx = bisect(config["solver"]["lr_milestones"], current_epoch)
        return pow(config["solver"]["lr_gamma"], idx)

 
reason_mode = config['model']['reason_mode']
logger.info('Reason mode: {}'.format(reason_mode))

bert_params = [item for item in list(model.named_parameters()) if 'bert' in item[0]]
not_bert_params = [item for item in list(model.named_parameters()) if 'bert' not in item[0]]

if 'no_decay' in config['solver'].keys():
    no_decay = config['solver']['no_decay']
else:
    no_decay = []

grouped_parameters = [
    {
        'params': [p for n, p in not_bert_params if not any(nd in n for nd in no_decay)], 
        'weight_decay': config['solver']['weight_decay']
    }, 
    {
        'params': [p for n, p in bert_params if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01
    },
    {
        'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
        'weight_decay': 0.0
    }
]

if config['solver']['optimizer'] == 'Adamax':
    optimizer = optim.Adamax(
        grouped_parameters, 
        lr=config["solver"]["initial_lr"],
        weight_decay=config['solver']['weight_decay']
    )
elif config['solver']['optimizer'] == 'AdamW':
    optimizer = AdamW(
        grouped_parameters, 
        lr=config["solver"]["initial_lr"], 
        weight_decay=config['solver']['weight_decay']
    )
else:
    raise NotImplementedError()
logger.info('Optimizer: {}'.format(config['solver']['optimizer']))
    
if config['solver']['lr_schedule'] == 'warmup_reduce_on_milestones':
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_fun)
elif config['solver']['lr_schedule'] == 'warmup_linear':
    warmup_steps = iterations * config["solver"]["warmup_epochs"]
    t_total = iterations * config["solver"]["num_epochs"]
    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=warmup_steps, t_total=t_total
    )
elif config['solver']['lr_schedule'] == 'warmup_cosine':
    warmup_steps = iterations * config["solver"]["warmup_epochs"]
    t_total = iterations * config["solver"]["num_epochs"]
    scheduler = WarmupCosineSchedule(
        optimizer, warmup_steps=warmup_steps, t_total=t_total
    )
elif config['solver']['lr_schedule'] == 'warmup_cosine_with_hard_restarts':
    warmup_steps = iterations * config["solver"]["warmup_epochs"]
    t_total = iterations * config["solver"]["num_epochs"]
    cycles = config['solver']['cycles']
    scheduler = WarmupCosineWithHardRestartsSchedule(
        optimizer, warmup_steps=warmup_steps, t_total=t_total, cycles=cycles
    )
else:
    raise NotImplementedError()
logger.info('Learning rate schedule: {}'.format(config['solver']['lr_schedule']))


# =============================================================================
#   SETUP BEFORE TRAINING LOOP
# =============================================================================

summary_writer = SummaryWriter(log_dir=args.save_dirpath)
checkpoint_manager = CheckpointManager(
    model, optimizer, args.save_dirpath, config=config
)
sparse_metrics = SparseGTMetrics()
ndcg = NDCG()

# Is stage 2, we should load the stage 1 pre-trained weights.
model_state_dict, _ = load_checkpoint(args.load_pthpath)
if isinstance(model, nn.DataParallel):
    model.module.load_state_dict(model_state_dict)
else:
    model.load_state_dict(model_state_dict)
logger.info("Loaded stage 1 model from {}".format(args.load_pthpath))


# =============================================================================
#   TRAINING LOOP
# =============================================================================

# Forever increasing counter to keep track of iterations (for tensorboard log).
start_epoch = 0
global_iteration_step = start_epoch * iterations
model.train()

for epoch in range(start_epoch, config["solver"]["num_epochs"]):

    # -------------------------------------------------------------------------
    #   ON EPOCH START  (combine dataloaders if training on train + val)
    # -------------------------------------------------------------------------
    if config["solver"]["training_splits"] == "trainval":
        combined_dataloader = (
            itertools.chain(train_dataloader, val_dataloader)
            if args.train2_json == ""
            else itertools.chain(train_dataloader, train2_dataloader, val_dataloader)
        )
    else:
        combined_dataloader = (
            itertools.chain(train_dataloader)
            if args.train2_json == ""
            else itertools.chain(train_dataloader, train2_dataloader)
        )

    logger.info(f"\nTraining for epoch {epoch}:")
    #pdb.set_trace()
    optimizer.zero_grad()
    for i, batch in enumerate(tqdm(combined_dataloader)):
        for key in batch:
            batch[key] = batch[key].to(device)

        output = model(batch)
        target = (
            batch['adjusted_gt_relevance']
            if config["dataset"]["use_adjusted"]
            else batch['gt_relevance']
        )
        batch_loss = criterion(
            output[torch.arange(output.size(0)), batch["round_id"] - 1, :], 
            target
        )
        batch_loss.backward()

        if ((i+1) % config['solver']['accumulation_steps'] == 0
            or (i+1) == data_iterations):
            if config['solver']['max_grad_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                            config["solver"]["max_grad_norm"])
            optimizer.step()
            optimizer.zero_grad()

            summary_writer.add_scalar(
                "train/loss", batch_loss, global_iteration_step
            )
            summary_writer.add_scalar(
                "train/lr", optimizer.param_groups[0]["lr"], global_iteration_step
            )

            scheduler.step(global_iteration_step)
            global_iteration_step += 1
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    #   ON EPOCH END  (checkpointing and validation)
    # -------------------------------------------------------------------------
    checkpoint_manager.step()

    # Validate and report automatic metrics.
    if args.validate:

        # Switch dropout, batchnorm etc to the correct mode.
        model.eval()

        # Evaluate validation set
        logger.info(f"\nValidation after epoch {epoch}:")
        for i, batch in enumerate(tqdm(val_dataloader)):
            for key in batch:
                batch[key] = batch[key].to(device)
            with torch.no_grad():
                output = model(batch)
            sparse_metrics.observe(output, batch["ans_ind"])
            if "gt_relevance" in batch:
                output = output[
                    torch.arange(output.size(0)), batch["round_id"] - 1, :
                ]
                ndcg.observe(output, batch["gt_relevance"])

        all_metrics = {}
        all_metrics.update(sparse_metrics.retrieve(reset=True))
        all_metrics.update(ndcg.retrieve(reset=True))
        for metric_name, metric_value in all_metrics.items():
            logger.info(f"{metric_name}: {metric_value}")
        summary_writer.add_scalars(
            "metrics_val", all_metrics, global_iteration_step
        )

        model.train()
        torch.cuda.empty_cache()
        summary_writer.flush()

    if args.validate_train:

        # Switch dropout, batchnorm etc to the correct mode.
        model.eval()

        # Evaluate training set.
        logger.info(f"\nValidation of training set after epoch {epoch}:")
        for i, batch in enumerate(tqdm(train_dataloader)):
            for key in batch:
                batch[key] = batch[key].to(device)
            with torch.no_grad():
                output = model(batch)
            sparse_metrics.observe(output, batch["ans_ind"])
            if "gt_relevance" in batch:
                output = output[
                    torch.arange(output.size(0)), batch["round_id"] - 1, :
                ]
                ndcg.observe(output, batch["gt_relevance"])
        all_metrics = {}
        all_metrics.update(sparse_metrics.retrieve(reset=True))
        all_metrics.update(ndcg.retrieve(reset=True))
        for metric_name, metric_value in all_metrics.items():
            logger.info(f"{metric_name}: {metric_value}")
        summary_writer.add_scalars(
            "metrics_train", all_metrics, global_iteration_step
        )

        # Evaluate part2 training set if exist.
        if args.train2_json != "":
            logger.info(f"\nValidation of fake training set after epoch {epoch}:")
            for i, batch in enumerate(tqdm(train2_dataloader)):
                for key in batch:
                    batch[key] = batch[key].to(device)
                with torch.no_grad():
                    output = model(batch)
                sparse_metrics.observe(output, batch['ans_ind'])
                if "gt_relevance" in batch:
                    output = output[
                        torch.arange(output.size(0)), batch['round_id'] - 1, :
                    ]
                    ndcg.observe(output, batch['gt_relevance'])
            all_metrics = {}
            all_metrics.update(sparse_metrics.retrieve(reset=True))
            all_metrics.update(ndcg.retrieve(reset=True))
            for metric_name, metric_value in all_metrics.items():
                logger.info(f"{metric_name}: {metric_value}")
            summary_writer.add_scalars(
                "metrics_train2", all_metrics, global_iteration_step
            )

        model.train()
        torch.cuda.empty_cache()
        summary_writer.flush()

summary_writer.close()