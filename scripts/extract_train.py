import argparse 
import json 
import copy 

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train-json",
    default="data/visdial_1.0_train.json",
    help="Path to json file containing VisDial v1.0 training data.",
)
parser.add_argument(
    "--train-dense-json",
    default="data/visdial_1.0_train_dense_sample.json",
    help="Path to json file containing VisDial v1.0 train subset dense ground "
    "truth annotations.",
)
parser.add_argument(
    "--sub-train-path",
    default="data/visdial_1.0_train_dense_sub.json",
    help="The file path of new train subset json file."
)
parser.add_argument(
    "--new-split-name",
    default="train_dense_subset",
    help="The split name used in new generated dataset's split attribute."
)
parser.add_argument(
    "--adjust-gt-relevance",
    action="store_true",
    help="Add the new dataset's adjusted gt_relevance values or not."
)
parser.add_argument(
    "--new-train-dense-json",
    default="data/visdial_1.0_train_dense_sample_adjusted.json",
    help="Path to save the adjusted gt_relevance added dense annotations. "
        "Only work when '--adjust-gt-relevance' set to True." 
)


def adjust_dense(old_dense, gt_index):
    new_dense = copy.deepcopy(old_dense)
    new_dense[gt_index] += 2.0
    new_dense = [x * (1.0 / 3.0) for x in new_dense]
    return new_dense


def build_new_json(old_json, old_dense_json, add_adjusted_dense=False, split_name=""):
    dense_ids = [x['image_id'] for x in old_dense_json]
    new_json = {}
    new_json['version'] = old_json['version']
    new_json['data'] = {}
    new_json['split'] = old_json['split'] if split_name == "" else split_name
    new_json['data']['dialogs'] = []
    new_json['data']['answers'] = old_json['data']['answers']
    new_json['data']['questions'] = old_json['data']['questions']
    new_json['data']['dialogs'] = [
        dialog for dialog in old_json['data']['dialogs'] 
        if dialog['image_id'] in dense_ids
    ]
    new_ids = [dialog['image_id'] for dialog in new_json['data']['dialogs']]
    new_dense_json = copy.deepcopy(old_dense_json)
    if add_adjusted_dense:
        for x in new_dense_json:
            j = new_ids.index(x['image_id'])
            assert(x['image_id'] == new_json['data']['dialogs'][j]['image_id'])
            x['gt_relevance'] = x['relevance']
            x['adjusted_gt_relevance'] = adjust_dense(
                x['gt_relevance'], 
                new_json['data']['dialogs'][j]['dialog'][x['round_id']-1]['gt_index']
            )
    return new_json, new_dense_json

if __name__ == '__main__':
    args = parser.parse_args()
    train_json = json.load(open(args.train_json, 'r'))
    train_dense_json = json.load(open(args.train_dense_json, 'r'))
    train_subset, new_dense_json = build_new_json(
        train_json, 
        train_dense_json, 
        args.adjust_gt_relevance, 
        args.new_split_name
    )
    json.dump(train_subset, open(args.sub_train_path, 'w'))
    json.dump(new_dense_json, open(args.new_train_dense_json, 'w'))
