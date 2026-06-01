import os
import sys
import argparse
from typing import Optional

# Allow ``from Graphs...`` when this file is run as ``python generate_conversation.py`` from data_utils/
_sys_dir = os.path.dirname(os.path.abspath(__file__))
if _sys_dir not in sys.path:
    sys.path.insert(0, _sys_dir)

from data_loader import load_data
import torch
import json
from tqdm import tqdm
import numpy as np

# =====================================================================
# Movies demo templates
# =====================================================================

def create_json_template_Movies_demo(id_value, demo_ids, gpt_value, text, demo_text, demo_label, task_type):
    data = {
        "id": id_value,
        "demo_id": demo_ids,
        "task_type": task_type,
        "conversations": [
            {
                "from": "human",
                "value":
                    f"Given an Amazon product in the Movies & TV category. The text description of this product is {text}. The image description of this product is <image>. This product is also co-purchased with other products, based on which we obtain the graph-aware feature: <graph>. The task is to classify this product into 19 classes: Movies, Genre for Featured Categories, Studio Specials, Musicals & Performing Arts, A&E Home Video, TV, Science Fiction & Fantasy, Boxed Sets, Walt Disney Studios Home Entertainment, Paramount Home Entertainment, Blu-ray, Art House & International, Criterion Collection, Holidays & Seasonal, Music Artists, BBC, Fully Loaded DVDs, Independently Distributed, HBO, Classics. Here are top-3 similar products calculated by PageRank algorithm associated with their truth class:\nProduct 1's multimodal features are: Text feature: {demo_text[0]}, Image feature :<image>, Graph-aware feature: <graph>; it belongs to: {demo_label[0]}\nProduct 2's multimodal features are: Text feature: {demo_text[1]}, Image feature :<image>, Graph-aware feature: <graph>; it belongs to: {demo_label[1]}\nProduct 3's multimodal features are: Text feature: {demo_text[2]}, Image feature :<image>, Graph-aware feature: <graph>; it belongs to: {demo_label[2]}\nPlease tell me which class the given product should belong to by considering its own multimodal features and the given demonstrations?"

            },
            {
                "from": "gpt",
                "value": gpt_value
            }
        ]
    }
    return data




# =====================================================================
# Arts demo templates
# =====================================================================

def create_json_template_Arts_demo(id_value, demo_ids, gpt_value, text, demo_text, demo_label, task_type):
    data = {
        "id": id_value,
        "demo_id": demo_ids[0:3],
        "task_type": task_type,
        "conversations": [
            {
                "from": "human",
                "value": (
                    f"Given an Amazon product in the Arts, Crafts & Sewing category. The text description of this product is {text}. The image description of this product is <image>. This product is also co-purchased with other products, based on which we obtain the graph-aware feature: <graph>. The task is to classify this product into 7 classes: Painting, Drawing & Art Supplies, Crafting, Sewing, Beading & Jewelry Making, Scrapbooking & Stamping, Knitting & Crochet, Model & Hobby Building. Here are top-3 similar products calculated by PageRank algorithm associated with their truth class:\nProduct 1's multimodal features are: Text feature: {demo_text[0]}, Image feature :<image>, Graph-aware feature: <graph>; it belongs to: {demo_label[0]}\nProduct 2's multimodal features are: Text feature: {demo_text[1]}, Image feature :<image>, Graph-aware feature: <graph>; it belongs to: {demo_label[1]}\nProduct 3's multimodal features are: Text feature: {demo_text[2]}, Image feature :<image>, Graph-aware feature: <graph>; it belongs to: {demo_label[2]}\nPlease tell me which class the given product should belong to by considering its own multimodal features and the given demonstrations?"
                )
            },
            {
                "from": "gpt",
                "value": gpt_value
            }
        ]
    }
    return data





# =====================================================================
# VideoGames demo templates
# =====================================================================

def create_json_template_VideoGames_demo(id_value, demo_ids, gpt_value, text, demo_text, demo_label, task_type):
    data = {
        "id": id_value,
        "demo_id": demo_ids,
        "task_type": task_type,
        "conversations": [
            {
                "from": "human",
                "value": (
                    f"Given an Amazon product in the VideoGames category. The text description of this product is {text}. The image description of this product is <image>. This product is also co-purchased with other products, based on which we obtain the graph-aware feature: <graph>. The task is to classify this product into 6 classes: Legacy Systems, PC, Nintendo Switch, PlayStation 4, PlayStation 5, Xbox One. Here are top-3 similar products calculated by PageRank algorithm associated with their truth class:\nProduct 1's multimodal features are: Text feature: {demo_text[0]}, Image feature :<image>, Graph-aware feature: <graph>; it belongs to: {demo_label[0]}\nProduct 2's multimodal features are: Text feature: {demo_text[1]}, Image feature :<image>, Graph-aware feature: <graph>; it belongs to: {demo_label[1]}\nProduct 3's multimodal features are: Text feature: {demo_text[2]}, Image feature :<image>, Graph-aware feature: <graph>; it belongs to: {demo_label[2]}\nPlease tell me which class the given product should belong to by considering its own multimodal features and the given demonstrations?"
                )
            },
            {
                "from": "gpt",
                "value": gpt_value
            }
        ]
    }
    return data




# =====================================================================
# RedditS demo templates
# =====================================================================

def create_json_template_RedditS_demo(id_value, demo_ids, gpt_value, text, demo_text, demo_label, task_type):
    data = {
        "id": id_value,
        "demo_id": demo_ids,
        "task_type": task_type,
        "conversations": [
            {
                "from": "human",
                "value": (
                    f"Given a post on the Reddit Website. The user comment is: {text}. The image of this post is: <image>. The user of this post also comments on other posts, based on which we obtain the graph-aware feature: <graph>. The task is to classify this post into 20 classes: beekeeping, birdphotography, chefknives, designmyroom, desksetup, duck, exposureporn, gardening, houseporn, machineporn, malelivingspace, natureisfuckinglit, orchids, pics, pottery, ruralporn, squirrels, urbanhell, volumeeating, woodcarving. Here are top-3 similar posts calculated by PageRank algorithm associated with their truth class:\nProduct 1's multimodal features are: Text feature: {demo_text[0]}, Image feature :<image>, Graph-aware feature: <graph>; it belongs to: {demo_label[0]}\nProduct 2's multimodal features are: Text feature: {demo_text[1]}, Image feature :<image>, Graph-aware feature: <graph>; it belongs to: {demo_label[1]}\nProduct 3's multimodal features are: Text feature: {demo_text[2]}, Image feature :<image>, Graph-aware feature: <graph>; it belongs to: {demo_label[2]}\nPlease tell me which class the given post should belong to by considering its own multimodal features and the given demonstrations?"
                )
            },
            {
                "from": "gpt",
                "value": gpt_value
            }
        ]
    }
    return data




# =====================================================================
# Demo JSONL generation
# =====================================================================

SUPPORTED_DATASETS = ("Movies", "Arts", "VideoGames", "RedditS")

DEMO_TEMPLATE_BUILDERS = {
    "Movies": create_json_template_Movies_demo,
    "Arts": create_json_template_Arts_demo,
    "VideoGames": create_json_template_VideoGames_demo,
    "RedditS": create_json_template_RedditS_demo,
}


def generate_demo_jsonl(
    dataset_name: str,
    split: str = "test",
    task_type: str = "nc",
    max_samples: Optional[int] = 1000,
    demos_dir: str = "./demos",
    output_dir: Optional[str] = None,
) -> str:
    """Generate class-label demo templates for the given dataset.

    Dispatches to the corresponding ``create_json_template_<Dataset>_demo``
    builder based on ``dataset_name`` and writes one JSON object per line.

    Parameters
    ----------
    dataset_name : str
        One of ``Movies`` / ``Arts`` / ``VideoGames`` / ``RedditS``.
    split : str
        ``train`` or ``test``.
    task_type : str
        Value stored as ``task_type`` inside each conversation entry.
    max_samples : Optional[int]
        Take at most this many nodes from the split (in original order).
        ``None`` or ``<= 0`` means no limit (use full split).
    demos_dir : str
        Directory containing ``<dataset_name>_demo.pt`` produced by the
        PageRank demo selection step.
    output_dir : Optional[str]
        Directory to write the JSONL file into. Defaults to
        ``./<dataset_name>`` (created if missing).

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

    builder = DEMO_TEMPLATE_BUILDERS[dataset_name]

    data, text = load_data(dataset_name=dataset_name)

    text_label = getattr(data, "text_label", None)
    if text_label is None:
        raise AttributeError(
            f"Loaded data for {dataset_name} is missing `text_label`; cannot build demo gpt values."
        )

    split_idx = data.train_id if split == "train" else data.test_id
    if isinstance(split_idx, np.ndarray):
        idx = split_idx.tolist()
    elif isinstance(split_idx, torch.Tensor):
        idx = split_idx.tolist()
    else:
        idx = list(split_idx)

    if max_samples is not None and max_samples > 0:
        idx = idx[:max_samples]

    demo_path = os.path.join(demos_dir, f"{dataset_name}_demo.pt")
    demos = torch.load(demo_path).tolist()

    conversations = []
    for node_id in tqdm(idx, desc=f"{dataset_name}/{split}"):
        demo_ids = demos[node_id]
        demo_text = [text[d] for d in demo_ids]
        demo_label = [text_label[d] for d in demo_ids]

        template = builder(
            node_id,
            demo_ids,
            text_label[node_id],
            text[node_id],
            demo_text,
            demo_label,
            task_type,
        )
        conversations.append(template)

    out_dir = output_dir if output_dir is not None else f"./{dataset_name}"
    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, f"{dataset_name}_{split}_TIQ_demo.jsonl")
    with open(output_file, "w") as f:
        for conversation in conversations:
            json.dump(conversation, f, indent=None)
            f.write("\n")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate demo JSONL files for Movies / Arts / VideoGames / RedditS."
    )
    parser.add_argument("--dataset", choices=list(SUPPORTED_DATASETS), required=True)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--task_type", default="nc")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Max number of samples (first N in split order). Default 1000. Use 0 for full split.",
    )
    parser.add_argument(
        "--demos-dir",
        default="./demos",
        help="Directory containing <dataset>_demo.pt files. Default: ./demos",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: ./<dataset>",
    )
    args = parser.parse_args()

    cap = None if args.max_samples <= 0 else args.max_samples
    output_file = generate_demo_jsonl(
        dataset_name=args.dataset,
        split=args.split,
        task_type=args.task_type,
        max_samples=cap,
        demos_dir=args.demos_dir,
        output_dir=args.output_dir,
    )
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
