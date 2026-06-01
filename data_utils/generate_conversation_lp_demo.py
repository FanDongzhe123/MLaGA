import os
import sys
import argparse
import json
from typing import Optional

# Allow ``from data_loader import ...`` when this file is executed as
# ``python generate_conversation_lp_demo.py`` from the data_utils/ directory.
_sys_dir = os.path.dirname(os.path.abspath(__file__))
if _sys_dir not in sys.path:
    sys.path.insert(0, _sys_dir)

import torch
from tqdm import tqdm

from data_loader import load_data


# =====================================================================
# Movies LP demo template
# =====================================================================

def create_json_template_Movies_lp_demo(id_value, gpt_value, demos, text, temp_demo_text, temp_demo_label, task_type):
    value_dict = {1: 'Yes', 0: 'No'}
    demo_text = []
    demo_label = []
    for text_pair in temp_demo_text:
        cat_text = "Product 1: " + text_pair[0] + ";" + " Product 2: " + text_pair[1]
        demo_text.append(cat_text)
    for label in temp_demo_label:
        demo_label.append(value_dict[label])

    centrol_node_text = "Product 1: " + text[0] + ";" + " Product 2: " + text[1]

    data = {
        "id": id_value,
        "demo_id": demos,
        "task_type": task_type,
        "conversations": [
            {
                "from": "human",
                "value":
                    f"Given two products sold in Aamzon Movies & TV category. The concat text description of these two products is {centrol_node_text}. The concat image description of these two products is <image>. The concat graph-aware feature of these two products is <graph>. We need to predict whether these two products are purchased or reviewed together. Here is top-1 similar products pair associated with their truth connections:\nPair 1's multimodal features are: the concat text description of these two products is : {demo_text[0]}, the concat image description of these two products is: <image>, the concat graph-aware feature of these two products is <graph>; Purchased or reviewed together: {demo_label[0]}.\nPlease tell me whether these two products should be purchased or reviewed together by considering their multimodal features and the given demonstrations."

            },
            {
                "from": "gpt",
                "value": gpt_value
            }
        ]
    }
    return data


# =====================================================================
# Arts LP demo template
# =====================================================================

def create_json_template_Arts_lp_demo(id_value, gpt_value, demos, text, temp_demo_text, temp_demo_label, task_type):
    value_dict = {1: 'Yes', 0: 'No'}
    demo_text = []
    demo_label = []
    for text_pair in temp_demo_text:
        cat_text = "Product 1: " + text_pair[0] + ";" + " Product 2: " + text_pair[1]
        demo_text.append(cat_text)
    for label in temp_demo_label:
        demo_label.append(value_dict[label])

    centrol_node_text = "Product 1: " + text[0] + ";" + " Product 2: " + text[1]

    data = {
        "id": id_value,
        "demo_id": demos,
        "task_type": task_type,
        "conversations": [
            {
                "from": "human",
                "value":
                    f"Given two products sold in Aamzon Arts, Crafts & Sewing category. The concat text description of these two products is {centrol_node_text}. The concat image description of these two products is <image>. The concat graph-aware feature of these two products is <graph>. We need to predict whether these two products are purchased or reviewed together. Here are top-2 similar products pair calculated by PageRank algorithm associated with their truth connections:\nPair 1's multimodal features are: the concat text description of these two products is : {demo_text[0]}, the concat image description of these two products is: <image>, the concat graph-aware feature of these two products is <graph>; Purchased or reviewed together: {demo_label[0]}.\nPlease tell me whether these two products should be purchased or reviewed together by considering their multimodal features and the given demonstrations."

            },
            {
                "from": "gpt",
                "value": gpt_value
            }
        ]
    }
    return data


# =====================================================================
# VideoGames LP demo template
# =====================================================================

def create_json_template_VideoGames_lp_demo(id_value, gpt_value, demos, text, temp_demo_text, temp_demo_label, task_type):
    value_dict = {1: 'Yes', 0: 'No'}
    demo_text = []
    demo_label = []
    for text_pair in temp_demo_text:
        cat_text = "Product 1: " + text_pair[0] + ";" + " Product 2: " + text_pair[1]
        demo_text.append(cat_text)
    for label in temp_demo_label:
        demo_label.append(value_dict[label])

    centrol_node_text = "Product 1: " + text[0] + ";" + " Product 2: " + text[1]

    data = {
        "id": id_value,
        "demo_id": demos,
        "task_type": task_type,
        "conversations": [
            {
                "from": "human",
                "value":
                    f"Given two products sold in Aamzon VideoGames category. The concat text description of these two products is {centrol_node_text}. The concat image description of these two products is <image>. The concat graph-aware feature of these two products is <graph>. We need to predict whether these two products are purchased or reviewed together. Here is top-1 similar products pair associated with their truth connections:\nPair 1's multimodal features are: the concat text description of these two products is : {demo_text[0]}, the concat image description of these two products is: <image>, the concat graph-aware feature of these two products is <graph>; Purchased or reviewed together: {demo_label[0]}\nPlease tell me whether these two products should be purchased or reviewed together by considering their multimodal features and the given demonstrations."

            },
            {
                "from": "gpt",
                "value": gpt_value
            }
        ]
    }
    return data


# =====================================================================
# RedditS LP demo template
# =====================================================================

def create_json_template_RedditS_lp_demo(id_value, gpt_value, demos, text, temp_demo_text, temp_demo_label, task_type):
    value_dict = {1: 'Yes', 0: 'No'}
    demo_text = []
    demo_label = []
    for text_pair in temp_demo_text:
        cat_text = "Product 1: " + text_pair[0] + ";" + " Product 2: " + text_pair[1]
        demo_text.append(cat_text)
    for label in temp_demo_label:
        demo_label.append(value_dict[label])

    centrol_node_text = "Product 1: " + text[0] + ";" + " Product 2: " + text[1]

    data = {
        "id": id_value,
        "demo_id": demos,
        "task_type": task_type,
        "conversations": [
            {
                "from": "human",
                "value":
                    f"Given two posts on Reddit Website. The concat text description of these two posts is: {centrol_node_text}. The concat image description of these two posts is: <image>. The concat graph-aware feature of these two posts is: <graph>. We need to predict whether these two posts are commented by a same user. Here is top-1 similar posts pair associated with their truth connections:\nPair 1's multimodal features are: the concat text description of these two posts is : {demo_text[0]}, the concat image description of these two posts is: <image>, the concat graph-aware feature of these two posts is <graph>; commented by a same user: {demo_label[0]}.\nPlease tell me whether these two products should be commented by a same user by considering their multimodal features and the given demonstrations."

            },
            {
                "from": "gpt",
                "value": gpt_value
            }
        ]
    }
    return data


# =====================================================================
# LP demo JSONL generation
# =====================================================================

SUPPORTED_DATASETS = ("Movies", "Arts", "VideoGames", "RedditS")

LP_DEMO_TEMPLATE_BUILDERS = {
    "Movies": create_json_template_Movies_lp_demo,
    "Arts": create_json_template_Arts_lp_demo,
    "VideoGames": create_json_template_VideoGames_lp_demo,
    "RedditS": create_json_template_RedditS_lp_demo,
}

LABEL_DICT = {0: "No", 1: "Yes"}


def generate_lp_demo_jsonl(
    dataset_name: str,
    split: str = "test",
    task_type: str = "lp",
    max_samples: Optional[int] = 0,
    graph_path: Optional[str] = None,
    demos_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> str:
    """Generate link-prediction demo templates for the given dataset.

    Dispatches to the corresponding ``create_json_template_<Dataset>_lp_demo``
    builder based on ``dataset_name`` and writes one JSON object per line.

    Parameters
    ----------
    dataset_name : str
        One of ``Movies`` / ``Arts`` / ``VideoGames`` / ``RedditS``.
    split : str
        ``train`` or ``test``.
    task_type : str
        Value stored as ``task_type`` inside each conversation entry.
        Defaults to ``"lp"`` for link-prediction.
    max_samples : Optional[int]
        Take at most this many edges from the split (in original order).
        ``None`` or ``<= 0`` means no limit (use full split).
    graph_path : Optional[str]
        Path to the LP graph ``.pt`` file. Defaults to
        ``./<dataset>Graph.pt``.
    demos_path : Optional[str]
        Path to the LP demos ``.pt`` file produced by the PageRank demo
        selection step. Defaults to
        ``./demos_lp/<dataset>Graph_demos_new_3.pt``.
    output_dir : Optional[str]
        Directory to write the JSONL file into. Defaults to
        ``./<dataset>`` (created if missing).

    Returns
    -------
    str
        Path of the written JSONL file.
    """
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"dataset_name must be one of {SUPPORTED_DATASETS}, got: {dataset_name}"
        )
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got: {split}")

    builder = LP_DEMO_TEMPLATE_BUILDERS[dataset_name]

    _, text = load_data(dataset_name=dataset_name)

    g_path = graph_path if graph_path is not None else f"./{dataset_name}Graph.pt"
    d_path = (
        demos_path
        if demos_path is not None
        else f"./demos_lp/{dataset_name}Graph_demos_new_3.pt"
    )

    data = torch.load(g_path)
    data_with_demo = torch.load(d_path)

    if split == "train":
        id_pairs = data.train_edges.tolist()
        labels = data.train_labels.tolist()
        demos_list = data_with_demo["train_demos"]
    else:
        id_pairs = data.test_edges.tolist()
        labels = data.test_labels.tolist()
        demos_list = data_with_demo["test_demos"]

    if max_samples is not None and max_samples > 0:
        id_pairs = id_pairs[:max_samples]
        labels = labels[:max_samples]

    conversations = []
    for idx, id_pair in tqdm(
        enumerate(id_pairs), total=len(id_pairs), desc=f"{dataset_name}/{split}"
    ):
        centrol_node_pair_text = [text[node_id] for node_id in id_pair]

        demos_dict = demos_list[tuple(id_pair)]
        demos = demos_dict["demos"]
        demos_label = demos_dict["labels"]

        pair_text = []
        for demo_pair in demos:
            if isinstance(demo_pair, int):
                print(demo_pair)
            pair_text.append([text[d] for d in demo_pair])

        label = LABEL_DICT[labels[idx]]
        template = builder(
            id_pair,
            label,
            demos,
            centrol_node_pair_text,
            pair_text,
            demos_label,
            task_type,
        )
        conversations.append(template)

    out_dir = output_dir if output_dir is not None else f"./{dataset_name}"
    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, f"{dataset_name}_{split}_TIQ_lp_demo.jsonl")
    with open(output_file, "w") as f:
        for conversation in conversations:
            json.dump(conversation, f, indent=None)
            f.write("\n")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate LP demo JSONL files for Movies / Arts / VideoGames / RedditS."
    )
    parser.add_argument("--dataset", choices=list(SUPPORTED_DATASETS), required=True)
    parser.add_argument(
        "--split",
        choices=["train", "test", "both"],
        default="both",
        help="Which split(s) to generate. 'both' generates train then test. Default: both.",
    )
    parser.add_argument("--task_type", default="lp")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max number of samples (first N in split order). Default 0 = full split.",
    )
    parser.add_argument(
        "--graph-path",
        default=None,
        help="Override default ./<dataset>Graph.pt path.",
    )
    parser.add_argument(
        "--demos-path",
        default=None,
        help="Override default ./demos_lp/<dataset>Graph_demos_new_3.pt path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: ./<dataset>",
    )
    args = parser.parse_args()

    cap = None if args.max_samples <= 0 else args.max_samples
    splits = ["train", "test"] if args.split == "both" else [args.split]

    for split in splits:
        output_file = generate_lp_demo_jsonl(
            dataset_name=args.dataset,
            split=split,
            task_type=args.task_type,
            max_samples=cap,
            graph_path=args.graph_path,
            demos_path=args.demos_path,
            output_dir=args.output_dir,
        )
        print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
