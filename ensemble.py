import argparse
import json
import os
import numpy as np 
import h5py 
import torch 
from glob import glob 
from scipy.special import softmax 
from tqdm import tqdm
from visdialch.metrics import SparseGTMetrics, NDCG, scores_to_ranks


parser = argparse.ArgumentParser(
    "Evaluate and/or generate EvalAI submission file by ensemble several "
    "model's predict results."
)
parser.add_argument(
    "--preds-folder",
    default="checkpoints/ensemble_models/",
    help="The path to the folder of the predict results of different models "
         "we want to ensemble.",
)
parser.add_argument(
    "--method",
    default="sa",
    choices=["sa", "pa", "ra", "rra"],
    help="Ensemble method. "
        "sa - Score Average; pa - Probability Average; "
        "ra - Rank Average; rra - Reciprocal Rank Average."
)
parser.add_argument(
    "--norm-order",
    default="2",
    help="The normlize order used in score average method. 'none' or int. "
        "Only work when method is 'sa'."
)
parser.add_argument(
    "--temp",
    default=1,
    type=float,
    help="The temperature used in temperature shaping when average. "
        "It's same as normal average if temp == 1."
)
parser.add_argument(
    "--split",
    default="val",
    choices=["val", "test"],
    help="Which split to ensemble upon.",
)
parser.add_argument(
    "--save-ranks-path",
    default="",
    help="The path to save ensembled results."
)


def score_average(preds, norm_order=None, temp=1):
    """
    Ensemble several model's predicts by average their predicted scores.

    Parameters:
    -----------
    preds: ndarray of shape (n_models, n_samples, n_rounds, n_options). 
        Several model's predict results.
    norm_order: {non-zero int, inf, -inf, ‘fro’, ‘nuc’}, optional. The normlize
        order used in numpy.linalg.norm() function. When it's None, we don't 
        normlize the predicts. When it's not None, the normlize will be apply 
        to last axis (option axis) which mean that we will nromlize different 
        round's option score to same scale.
    temp: float, optinal, default is 1.
        The temperature used in temperature sharping. It's same as normal 
        average when temp is 1. Otherwise, the preds 'p' will be powered by 
        temperature 't' elementwise (p=p**t) before average.
    
    Returns:
    --------
    final_pred: ndarray of shape (n_sample, n_rounds, n_options). The ensembled
        result.
    """
    print('Norm order:', norm_order)
    if norm_order is None:
        preds = (preds - preds.min()) / (preds.max() - preds.min())
        return (preds**temp).mean(axis=0)
    else:
        preds_norm = np.linalg.norm(
            preds, axis=-1, ord=norm_order, keepdims=True
        ) + 1e-8
        preds = preds / preds_norm
        preds = (preds - preds.min()) / (preds.max() - preds.min())
        return (preds**temp).mean(axis=0)


def prob_average(preds, temp=1):
    """
    Ensemble several model's predicts by converting predict scores to 
    probability using softmax function and average those probabilities.

    Parameters:
    -----------
    preds: ndarray of shape (n_models, n_samples, n_rounds, n_options). 
        Several model's predict results.

    Returns:
    -------
    final_pred: ndarray of shape (n_sample, n_rounds, n_options). 
        The ensembled result.
    """
    return (softmax(preds, axis=-1)**temp).mean(axis=0)


def rank_average(preds, temp=1):
    """
    Ensemble several model's predicts by average their predicted ranks.

    Parameters:
    -----------
    preds: ndarray of shape (n_models, n_samples, n_rounds, n_options).
        Several model's predict results.

    Returns:
    --------
    final_pred: ndarray of shape (n_sample, n_rounds, n_options). 
        The ensembled result.
    """
    ranks = np.array([
        scores_to_ranks(torch.tensor(pred)).cpu().numpy()
        for pred in preds
    ])
    ranks = (ranks - ranks.min()) / (ranks.max() - ranks.min())
    return 1.0 - (ranks**temp).mean(axis=0)


def reciprocal_rank_average(preds, temp=1):
    """
    Ensemble several model's predicts by average their predicted 
        reciprocal ranks.

    Parameters:
    -----------
    preds: ndarray of shape (n_models, n_samples, n_rounds, n_options).
        Several model's predict results.

    Returns:
    --------
    final_pred: ndarray of shape (n_sample, n_rounds, n_options). 
        The ensembled result.
    """
    ranks = np.array([
        1.0 / scores_to_ranks(torch.tensor(pred)).cpu().numpy()
        for pred in preds
    ])
    ranks = (ranks - ranks.min()) / (ranks.max() - ranks.min())
    return (ranks**temp).mean(axis=0)


def load_data(preds_folder_path, split, sort_ids=False):
    """
    Load all the predict results in a folder into one ndarray.

    Parameters:
    -----------
    preds_folder_path: The path to the foler contaning predict results h5 files
        generated form evaluate.py script.
    split: Which split those predict results belong to. 'val' or 'test'.
    sort_ids: bool, default is False. Whther sort the image ids or not. Could 
        used to make sure that we could ensemble differet models that evaluate 
        the dataset with different sample order.

    Returns:
    --------
    data: dict with keys ['image_ids', 'preds', 'round_ids'] for val test and 
        additional ['answer_indexes', 'gt_relevances'] for val split.
    """
    preds_paths = glob(os.path.join(preds_folder_path, "*.h5"))
    print("Found {} models' predict results".format(len(preds_paths)))
    preds = []
    previous_image_ids = None
    round_ids = None
    if split == 'val':
        answer_indexes = None
        gt_relevances = None

    for path in preds_paths:
        print(path)
        h5 = h5py.File(path)
        assert(split == h5.attrs['split'])
        image_ids = np.array(h5['image_ids'])
        if sort_ids:
            if previous_image_ids is not None:
                np.testing.assert_array_equal(
                    np.sort(image_ids), np.sort(previous_image_ids)
                )
            previous_image_ids = image_ids
            sorted_index = image_ids.argsort()
            preds.append(np.array(h5['pred_scores'])[sorted_index])
            if round_ids is None:
                round_ids = np.array(h5['round_ids'])[sorted_index]
            if split == 'val':
                if answer_indexes is None:
                    answer_indexes = np.array(h5['answer_indexes'])[sorted_index]
                if gt_relevances is None:
                    gt_relevances = np.array(h5['gt_relevances'])[sorted_index]
            image_ids = np.sort(image_ids)
        else:
            if previous_image_ids is not None:
                np.testing.assert_array_equal(
                    image_ids, previous_image_ids
                )
            previous_image_ids = image_ids
            preds.append(np.array(h5['pred_scores']))
            if round_ids is None:
                round_ids = np.array(h5['round_ids'])
            if split == 'val':
                if answer_indexes is None:
                    answer_indexes = np.array(h5['answer_indexes'])
                if gt_relevances is None:
                    gt_relevances = np.array(h5['gt_relevances'])

    if split == 'val':
        return {
            'image_ids': image_ids,
            'preds': np.array(preds),
            'round_ids': round_ids,
            'answer_indexes': answer_indexes,
            'gt_relevances': gt_relevances
        }
    else:
        return {
            'image_ids': image_ids,
            'preds': np.array(preds),
            'round_ids': round_ids,
        }


def eval_pred(pred, answer_index, round_id, gt_relevance):
    """
    Evaluate the predict results and report metrices. Only for val split.

    Parameters:
    -----------
    pred: ndarray of shape (n_samples, n_rounds, n_options).
    answer_index: ndarray of shape (n_sample, n_rounds).
    round_id: ndarray of shape (n_samples, ).
    gt_relevance: ndarray of shape (n_samples, n_options).

    Returns:
    --------
    None
    """
    # Convert them to torch tensor to use visdialch.metrics
    pred = torch.Tensor(pred)
    answer_index = torch.Tensor(answer_index).long()
    round_id = torch.Tensor(round_id).long()
    gt_relevance = torch.Tensor(gt_relevance)

    sparse_metrics = SparseGTMetrics()
    ndcg = NDCG()

    sparse_metrics.observe(pred, answer_index)
    pred = pred[torch.arange(pred.size(0)), round_id - 1, :]
    ndcg.observe(pred, gt_relevance)

    all_metrics = {}
    all_metrics.update(sparse_metrics.retrieve(reset=True))
    all_metrics.update(ndcg.retrieve(reset=True))
    for metric_name, metric_value in all_metrics.items():
        print(f"{metric_name}: {metric_value}")


def write_to_json(pred, image_id, round_id, json_path, split):
    """
    Write predict results to json file.

    Parameters:
    -----------
    pred: ndarray of shape (n_samples, n_rounds, n_options).
    image_id: ndarray of shape (n_sample, ).
    round_id: ndarray of shape (n_sample, ).
    json_path: The path used to save results.

    Retures:
    --------
    None
    """
    pred = torch.Tensor(pred)
    ranks = scores_to_ranks(pred)
    ranks_json = []
    for i, img_id in enumerate(image_id):
        if split == 'test':
            ranks_json.append({
                'image_id': int(img_id),
                'round_id': int(round_id[i]),
                'ranks': [
                    rank.item()
                    for rank in ranks[i][round_id[i] - 1]
                ]
            })
        elif split == 'val':
            for j in range(10):
                ranks_json.append(
                    {
                        "image_id": int(img_id),
                        "round_id": int(j + 1),
                        "ranks": [rank.item() for rank in ranks[i][j]],
                    }
                )
    json.dump(ranks_json, open(json_path, 'w'))


if __name__ == '__main__':
    args = parser.parse_args()
    data = load_data(args.preds_folder, args.split)

    if args.norm_order == 'none':
        norm_order = None
    else:
        norm_order = int(args.norm_order)

    if args.method == 'sa':
        print('Using score average.')
        final_pred = score_average(
            preds=data['preds'], 
            norm_order=norm_order, 
            temp=args.temp
        )
    elif args.method == 'pa':
        print('Using prob average.')
        final_pred = prob_average(data['preds'], temp=args.temp)
    elif args.method == 'ra':
        print('Using rank average.')
        final_pred = rank_average(data['preds'], temp=args.temp)
    elif args.method == 'rra':
        print('Using reciprocal rank average.')
        final_pred = reciprocal_rank_average(data['preds'], temp=args.temp)
    else:
        raise NotImplementedError()

    print('Temp:', args.temp)

    if args.split == 'val':
        print('Evaluate metrices of ensembled results...')
        eval_pred(
            pred=final_pred, 
            answer_index=data['answer_indexes'],
            round_id=data['round_ids'],
            gt_relevance=data['gt_relevances']
        )
        if args.save_ranks_path != "":
            print('Write ensembled results to', args.save_ranks_path)
            write_to_json(
                pred=final_pred,
                image_id=data['image_ids'],
                round_id=data['round_ids'],
                json_path=args.save_ranks_path,
                split='val'
            )
    else:
        print('Write ensembled results to', args.save_ranks_path)
        write_to_json(
            pred=final_pred,
            image_id=data['image_ids'],
            round_id=data['round_ids'],
            json_path=args.save_ranks_path,
            split='test'
        )
