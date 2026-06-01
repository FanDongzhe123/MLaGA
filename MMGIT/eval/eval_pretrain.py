import sys
sys.path.append("..")
sys.path.append(".")
sys.path.append("../..")
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import pdb
from data_utils.data_loader import load_data
import copy

from MMGIT.constants import GRAPH_TOKEN_INDEX, DEFAULT_GRAPH_TOKEN, DEFAULT_GRAPH_PAD_ID, DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN, IMAGE_TOKEN_INDEX, GRAPH_TOKEN_INDEX
from MMGIT.conversation import conv_templates, SeparatorStyle
from MMGIT.model.builder import load_pretrained_model
from MMGIT.utils import disable_torch_init, tokenizer_graph_token, get_model_name_from_path
from torch_geometric.utils import k_hop_subgraph, degree, remove_self_loops, add_self_loops
from torch_geometric.nn import MessagePassing
import math
import transformers

SMALL_DATASETS=["pubmed", "cora"]


class MP(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


# def get_chunk(lst, n, k):
#     chunks = split_list(lst, n)
#     return chunks[k]
def preprocess_llama3(
    source,
    tokenizer: transformers.PreTrainedTokenizer,
    has_graph: bool = False,
    max_len=2048,
    system_message: str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
    
):
    roles = {"human": "user", "gpt": "assistant"}
    tokenizer = copy.deepcopy(tokenizer)
    if has_graph:
        tokenizer.add_tokens(["<graph>"], special_tokens=True)
        tokenizer.add_tokens(["<image>"], special_tokens=True)

    graph_token_index = tokenizer.convert_tokens_to_ids("<graph>")
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    chat_template = "{% for message in messages %}{{'<|begin_of_text|><|start_header_id|>' + message['role'] + '<|end_header_id|>' + '\n' + message['content'] + '<|eot_id|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # unmask_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "\n\n"]
    # unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    
    nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")
    input_ids, targets = [], []

    if roles[source["from"]] != roles["human"]:
        source = source[1:]

    input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
    input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])

            # Make sure llava data can load
    try:
        role = source["role"]
        content = source["content"]
    except:
        role = source["from"]
        content = source["value"]

    role =  roles.get(role, role)
            
    conv = [{"role" : role, "content" : content}]
            # First is bos token we don't need here
    encode_id = tokenizer.apply_chat_template(conv, add_generation_prompt=True)[1:]
    input_id += encode_id

        

                    

        
    for idx, encode_id in enumerate(input_id):
            # if encode_id in unmask_tokens_idx:
            #     target[idx] = encode_id
        if encode_id == image_token_index:
            input_id[idx] = IMAGE_TOKEN_INDEX
        if encode_id == graph_token_index:
            input_id[idx] = GRAPH_TOKEN_INDEX

    input_ids.append(input_id)
    input_ids = torch.tensor(input_ids, dtype=torch.long)

    return input_ids

def load_pretrain_embedding_graph(data_dir, pretrained_embedding_type):


    if pretrained_embedding_type == "clip":
        graph_emb = torch.load(os.path.join(data_dir, "clip.pt"))
        image_emb = torch.load(os.path.join(data_dir, "clip_image_new.pt"))
        text_emb = torch.load(os.path.join(data_dir, "clip_text_new.pt"))
        if graph_emb.is_cuda:
            graph_emb = graph_emb.to("cpu")
        if image_emb.is_cuda:
            image_emb = image_emb.to("cpu")
        if text_emb.is_cuda:
            text_emb = text_emb.to("cpu")
        return graph_emb, image_emb, text_emb
    elif pretrained_embedding_type == "clip_mix":
        graph_emb = torch.load(os.path.join(data_dir, "clip_mix.pt"))
        image_emb = torch.load(os.path.join(data_dir, "clip_image_new.pt"))
        text_emb = torch.load(os.path.join(data_dir, "clip_text_new.pt"))
        if graph_emb.is_cuda:
            graph_emb = graph_emb.to("cpu")
        if image_emb.is_cuda:
            image_emb = image_emb.to("cpu")
        if text_emb.is_cuda:
            text_emb = text_emb.to("cpu")
        return graph_emb, image_emb, text_emb
    elif pretrained_embedding_type == "clip_mix_new":
        graph_emb = torch.load(os.path.join(data_dir, "clip_mix_new.pt"))
        image_emb = torch.load(os.path.join(data_dir, "clip_image_new.pt"))
        text_emb = torch.load(os.path.join(data_dir, "clip_text_new.pt"))
        if graph_emb.is_cuda:
            graph_emb = graph_emb.to("cpu")
        if image_emb.is_cuda:
            image_emb = image_emb.to("cpu")
        if text_emb.is_cuda:
            text_emb = text_emb.to("cpu")
        return graph_emb, image_emb, text_emb
    elif pretrained_embedding_type == "clip_sbert":
        graph_emb = torch.load(os.path.join(data_dir, "clip_sbert.pt"))
        image_emb = torch.load(os.path.join(data_dir, "clip_image_new.pt"))
        text_emb = torch.load(os.path.join(data_dir, "clip_text_new.pt"))
        if graph_emb.is_cuda:
            graph_emb = graph_emb.to("cpu")
        if image_emb.is_cuda:
            image_emb = image_emb.to("cpu")
        if text_emb.is_cuda:
            text_emb = text_emb.to("cpu")
        return graph_emb, image_emb, text_emb
    elif pretrained_embedding_type == "clip_sbert_blip":
        graph_emb = torch.load(os.path.join(data_dir, "clip_sbert_blip.pt"))
        image_emb = torch.load(os.path.join(data_dir, "clip_image_new.pt"))
        text_emb = torch.load(os.path.join(data_dir, "clip_text_new.pt"))
        if graph_emb.is_cuda:
            graph_emb = graph_emb.to("cpu")
        if image_emb.is_cuda:
            image_emb = image_emb.to("cpu")
        if text_emb.is_cuda:
            text_emb = text_emb.to("cpu")
        return graph_emb, image_emb, text_emb
    elif pretrained_embedding_type == 'clip_qquery':
        graph_emb = torch.load(os.path.join(data_dir, "clip.pt"))
        if graph_emb.is_cuda:
            graph_emb = graph_emb.to("cpu")
        return graph_emb, None, None
    elif pretrained_embedding_type == 'all':
        graph_emb = torch.load(os.path.join(data_dir, "query_token_all.pt"))
        if graph_emb.is_cuda:
            graph_emb = graph_emb.to("cpu")
        return graph_emb, None, None
    elif pretrained_embedding_type == 'all_cat':
        graph_emb = torch.load(os.path.join(data_dir, "query_token_all.pt"))
        if graph_emb.is_cuda:
            graph_emb = graph_emb.to("cpu")
        graph_emb = graph_emb.view(graph_emb.shape[0], -1)
        return graph_emb, None, None
    elif pretrained_embedding_type == 'clip_text':
        graph_emb = torch.load(os.path.join(data_dir, "clip_text_new.pt"))
        return graph_emb, None, None
    elif pretrained_embedding_type == 'clip_image':
        graph_emb = torch.load(os.path.join(data_dir, "clip_image_new.pt"))
        return graph_emb, None, None
    elif pretrained_embedding_type == 'clip_image_text':
        image_emb = torch.load(os.path.join(data_dir, "clip_image_new.pt"))
        text_emb = torch.load(os.path.join(data_dir, "clip_text_new.pt"))
        graph_emb = torch.cat([text_emb, image_emb], dim=1)
        return graph_emb, None, None
    elif pretrained_embedding_type == 'clip_all':
        graph_emb = torch.load(os.path.join(data_dir, "clip.pt"))
        image_emb = torch.load(os.path.join(data_dir, "clip_image_new.pt"))
        text_emb = torch.load(os.path.join(data_dir, "clip_text_new.pt"))
        if graph_emb.is_cuda:
            graph_emb = graph_emb.to("cpu")
        if image_emb.is_cuda:
            image_emb = image_emb.to("cpu")
        if text_emb.is_cuda:
            text_emb = text_emb.to("cpu")
        return graph_emb, image_emb, text_emb
    else:
        pretrained_emb = torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_x.pt"))
    return graph_emb, image_emb, text_emb

def load_pretrain_embedding_hop(data_dir, pretrained_embedding_type, hop, mask):
    if pretrained_embedding_type == "simteg":
        simteg_sbert=[torch.load(os.path.join(data_dir, f"simteg_sbert_x.pt"))[mask]] + [torch.load(os.path.join(data_dir, f"simteg_sbert_{i}hop_x.pt"))[mask] for i in range(1, hop + 1)]
        simteg_roberta = [torch.load(os.path.join(data_dir, f"simteg_roberta_x.pt"))[mask]] + [torch.load(os.path.join(data_dir, f"simteg_roberta_{i}hop_x.pt"))[mask] for i in range(1, hop + 1)]
        simteg_e5 = [torch.load(os.path.join(data_dir, f"simteg_e5_x.pt"))[mask]] + [torch.load(os.path.join(data_dir, f"simteg_e5_{i}hop_x.pt"))[mask] for i in range(1, hop + 1)]
        pretrained_embs = [torch.cat([simteg_sbert[i], simteg_roberta[i], simteg_e5[i]], dim=-1) for i in range(hop + 1)]
    else:
        pretrained_embs = [torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_x.pt"))[mask]]+  [torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_{i}hop_x.pt"))[mask] for i in range(1, hop+1)]

    return pretrained_embs

def load_pretrain_embedding_hop_lp(data_dir, pretrained_embedding_type, hop):
    mask = torch.load(os.path.join(data_dir, f"no_test_link_mask.pt"))
    if pretrained_embedding_type == "simteg":
        simteg_sbert=[torch.load(os.path.join(data_dir, f"simteg_sbert_x.pt"))[mask]] + [torch.load(os.path.join(data_dir, f"simteg_sbert_{i}hop_x_notestlink.pt")) for i in range(1, hop + 1)]
        simteg_roberta = [torch.load(os.path.join(data_dir, f"simteg_roberta_x.pt"))[mask]] + [torch.load(os.path.join(data_dir, f"simteg_roberta_{i}hop_x_notestlink.pt")) for i in range(1, hop + 1)]
        simteg_e5 = [torch.load(os.path.join(data_dir, f"simteg_e5_x.pt"))[mask]] + [torch.load(os.path.join(data_dir, f"simteg_e5_{i}hop_x_notestlink.pt")) for i in range(1, hop + 1)]
        pretrained_embs = [torch.cat([simteg_sbert[i], simteg_roberta[i], simteg_e5[i]], dim=-1) for i in range(hop + 1)]
    else:
        pretrained_embs = [torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_x.pt"))[mask]]+  [torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_{i}hop_x_notestlink.pt")) for i in range(1, hop+1)]

    return pretrained_embs, mask

def eval_model(args):
    # Model
    disable_torch_init()
    # pdb.set_trace()
    use_text_cls = False
    use_neighbors = False
    use_all = False
    graph_only = False

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(f"Loaded from {model_path}. Model Base: {args.model_base}")
    tokenizer, model, context_len = load_pretrained_model(model_path, args.model_base, model_name, args.pretrained_lora_path,
                                                          cache_dir=args.cache_dir)
    model = model.to(torch.float16).cuda()
    # data_dir=os.path.expanduser(args.data_dir)
    if args.dataset == "Movies":
        data_dir = "dataset/Movies"
    elif args.dataset == "Toys":
        data_dir = "dataset/Toys"
    elif args.dataset == "Grocery":
        data_dir = "dataset/Grocery"
    elif args.dataset == "VideoGames":
        data_dir = "dataset/VideoGames"
    elif args.dataset == "Health":
        data_dir = "dataset/Health/"
    elif args.dataset == "Beauty":
        data_dir = "dataset/Beauty/"
    elif args.dataset == "CD":
        data_dir = "dataset/CD/"
    elif args.dataset == "Arts":
        data_dir = "dataset/Arts/"
    elif args.dataset == "Automotive":
        data_dir = "dataset/Automotive/"
    elif args.dataset == "RedditS":
        data_dir = "dataset/RedditS/"
    elif args.dataset == "Goodreads":
        data_dir = "/vast/df2362/Goodreads"
    elif args.dataset == "Art500K":
        data_dir = "dataset/Art500K/"
    elif args.dataset == "Reddit":
        data_dir = "dataset/Reddit/"
    else:
        print(f"{args.dataset} not exists")
        raise ValueError
    if args.task in  ["nc", "nd", "nda", "nctext"]:
        if args.template == "HO":
            prompt_file = os.path.join(data_dir, f"{args.dataset}_test.jsonl")
        elif args.template == "TIQ":
            prompt_file = os.path.join(data_dir, f"{args.dataset}_test_TIQ.jsonl")
        elif args.template == "TIQ_semi":
            prompt_file = os.path.join(data_dir, f"{args.dataset}_test_TIQ_semi.jsonl")
        elif args.template == "TIQ_v2":
            prompt_file = os.path.join(data_dir, f"{args.dataset}_test_TIQ_v2.jsonl")
        elif args.template == "TIQ_demo":
            prompt_file = os.path.join(data_dir, f"{args.dataset}_test_TIQ_demo_degree.jsonl")
        elif args.template == "TIQ_demo_1":
            prompt_file = os.path.join(data_dir, f"{args.dataset}_test_TIQ_demo_1.jsonl")
        elif args.template == "TIQ_demo_2":
            prompt_file = os.path.join(data_dir, f"{args.dataset}_test_TIQ_demo_2.jsonl")
        elif args.template == "TIQ_demo_bootstrap":
            prompt_file = os.path.join(data_dir, f"{args.dataset}_test_TIQ_demo_bootstrap_{args.bootstrap}.jsonl")
        elif args.template == "TIQ_demo_text_only":
            prompt_file = os.path.join(data_dir, f"{args.dataset}_test_TIQ_demo_text_only.jsonl")
        elif args.template == "TIQ_demo_image_only":
            prompt_file = os.path.join(data_dir, f"{args.dataset}_test_TIQ_demo_image_only.jsonl")
        elif args.template == "TIQ_demo_without_label":
            prompt_file = os.path.join(data_dir, f"{args.dataset}_test_TIQ_demo_without_label.jsonl")

        elif args.template == "ND":
            prompt_file = os.path.join(data_dir, f"{args.dataset}_test_ND.jsonl")
            graph_only = True
            use_neighbors = True
        elif args.template == "ND_sequence":
            prompt_file = os.path.join(data_dir,f"{args.dataset}_test_sequence.jsonl")
            use_neighbors = True
        elif args.template == "ND_all":
            prompt_file = os.path.join(data_dir,f"{args.dataset}_test_all.jsonl")
            use_neighbors = True
            use_all = True
        elif args.template == "ND_text_cls":
            prompt_file = os.path.join(data_dir,f"{args.dataset}_test_text.jsonl")
            use_text_cls = True
        elif args.template == "base":
            prompt_file = os.path.join(data_dir,f"{args.dataset}_test_base.jsonl")
        elif args.template == "base_sequence":
            prompt_file = os.path.join(data_dir,f"{args.dataset}_test_base_sequence.jsonl")
            use_neighbors = True
        elif args.template == "base_text_cls":
            prompt_file = os.path.join(data_dir,f"{args.dataset}_test_base.jsonl")
            use_text_cls = True

        data_path = os.path.join(data_dir, f"processed_data.pt")
    elif args.task in ["lp"]:
        if args.template == "TIQ":
            prompt_file = os.path.join(data_dir, f"{args.dataset}_test_TIQ_lp.jsonl")
        elif args.template == "TIQ_cat":
            prompt_file = os.path.join(data_dir, f"{args.dataset}_test_TIQ_lp_cat.jsonl")
        elif args.template == "TIQ_demo":
            prompt_file = os.path.join(data_dir, f"{args.dataset}_test_TIQ_lp_demo.jsonl")
        elif args.template == "TIQ_demo_1":
            prompt_file = os.path.join(data_dir, f"{args.dataset}_test_TIQ_lp_demo.jsonl")
        elif args.template == "TIQ_demo_2":
            prompt_file = os.path.join(data_dir, f"{args.dataset}_test_TIQ_lp_demo.jsonl")
        elif args.template == "TIQ_demo_bootstrap":
            prompt_file = os.path.join(data_dir, f"{args.dataset}_test_TIQ_lp_demo_bootstrap_{args.bootstrap}.jsonl")
        elif args.template == "TIQ_demo_text_only":
            prompt_file = os.path.join(data_dir, f"{args.dataset}_test_TIQ_lp_demo_text_only.jsonl")
        elif args.template == "TIQ_demo_image_only":
            prompt_file = os.path.join(data_dir, f"{args.dataset}_test_TIQ_lp_demo_image_only.jsonl")
        elif args.template == "TIQ_demo_without_label":
            prompt_file = os.path.join(data_dir, f"{args.dataset}_test_TIQ_lp_demo_without_label.jsonl")
        else:
            prompt_file = os.path.join(data_dir, f"edge_sampled_{args.use_hop}_{args.sample_neighbor_size}_only_test.jsonl")
        data_path = os.path.join(data_dir, f"processed_data.pt")
    else:
        raise ValueError

    # data = torch.load(data_path)
    # data = load_data(dataset_name=args.dataset, use_text=False)
    data = None
    print(f"Load from {prompt_file}\n")
    lines = open(prompt_file, "r").readlines()

    if args.start >= 0:
        if args.end < 0:
            args.end = len(lines)
        lines = lines[args.start:args.end]
    elif args.end > 0:
        lines = lines[:args.end]

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    if "tmp" not in args.answers_file and os.path.exists(answers_file):
        line_number = len(open(answers_file, 'r').readlines())
        print(f"{args.answers_file} already exists! it has {line_number} lines!!")
        if line_number >= len(lines):
            return
        lines = lines[line_number:]
        ans_file = open(answers_file, "a")
    else:
        ans_file = open(answers_file, "w")

    questions = [json.loads(q) for q in lines]

    index = None
    if "ND" in args.template or "base" in args.template or args.template == "TIQ" or args.template == "TIQ_semi" or args.template == "TIQ_v2" or args.template == "TIQ_demo" or args.template == "TIQ_cat" or args.template == "TIQ_demo_without_label" or args.template == "TIQ_demo_image_only" or args.template == "TIQ_demo_text_only" or args.template == "TIQ_demo_bootstrap" or args.template == "TIQ_demo_1" or args.template == "TIQ_demo_2":
        pretrained_graph_emb, pretrained_image_emb, pretrained_text_emb = load_pretrain_embedding_graph(data_dir, args.pretrained_embedding_type)
        if 'sequence' in args.template or 'all' in args.template or args.template == 'ND':
            if 'all' in args.template:
                pretrained_node_emb = pretrained_graph_emb
            structure_emb = torch.load(
                f"dataset/laplacian_{args.use_hop}_{args.sample_neighbor_size}.pt")
            if args.pretrained_embedding_type == 'clip_all' or args.pretrained_embedding_type == 'clip_qquery' or args.pretrained_embedding_type == 'all':
                pretrained_graph_emb = torch.mean(pretrained_graph_emb, dim=1)
            if args.pretrained_embedding_type == 'clip_all':
                pretrained_graph_emb = torch.cat([pretrained_graph_emb, pretrained_text_emb, pretrained_image_emb], dim=1)

    elif args.template == "HO":
        n = data.num_nodes
        if args.dataset in SMALL_DATASETS and args.task == "lp":
            pretrained_emb = load_pretrain_embedding_graph(data_dir, args.pretrained_embedding_type)
        elif args.task == "lp":
            # for small dataset, we remove test link during testing
            # for large dataset, remove test link and compute embedding may be more memory- and time-consuming , we precompute the embedding
            pretrained_emb, mask = load_pretrain_embedding_hop_lp(data_dir, args.pretrained_embedding_type,args.use_hop)
            index = torch.full([n], fill_value=n + 1, dtype=torch.long)
            test_index = torch.arange(mask.sum())
            index[mask] = test_index
        else:
            mask = torch.full([n], fill_value=False, dtype=torch.bool)
            for q in questions:
                idx = q["id"]
                if "lp" in  args.task:
                    assert len(idx) == 2
                    mask[idx[0]] = True
                    mask[idx[1]] = True
                elif args.task  in ["nc", "nd", "nctext"]:
                    assert isinstance(idx, int)
                    mask[idx] = True
            pretrained_emb = load_pretrain_embedding_hop(data_dir, args.pretrained_embedding_type, args.use_hop, mask)
            index = torch.full([n], fill_value=n + 1, dtype=torch.long)
            test_index = torch.arange(mask.sum())
            index[mask] = test_index
        structure_emb = None
    else:
        raise ValueError

    node_attention_stats = {
        "nc": {"scores": [], "count": 0},
        "lp": {"scores": [], "count": 0}
    }
    
    cross_task_flow_stats = {
        "nc": {"ratios": [], "count": 0},
        "lp": {"ratios": [], "count": 0}
    }
    for line in tqdm(questions):
        idx = line["id"]
        task_type = [line['task_type']]
        if args.task in ["nd", "nda"]:
            qs=f"Please briefly describe the center node of {DEFAULT_GRAPH_TOKEN}."
        elif args.task == "nc":
            if args.dataset == "products":
                qs = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent products sold in Amazon, and edges between products indicate they are purchased together. We need to classify the center node into 47 classes: Home & Kitchen, Health & Personal Care, Beauty, Sports & Outdoors, Books, Patio, Lawn & Garden, Toys & Games, CDs & Vinyl, Cell Phones & Accessories, Grocery & Gourmet Food, Arts, Crafts & Sewing, Clothing, Shoes & Jewelry, Electronics, Movies & TV, Software, Video Games, Automotive, Pet Supplies, Office Products, Industrial & Scientific, Musical Instruments, Tools & Home Improvement, Magazine Subscriptions, Baby Products, label 25, Appliances, Kitchen & Dining, Collectibles & Fine Art, All Beauty, Luxury Beauty, Amazon Fashion, Computers, All Electronics, Purchase Circles, MP3 Players & Accessories, Gift Cards, Office & School Supplies, Home Improvement, Camera & Photo, GPS & Navigation, Digital Music, Car Electronics, Baby, Kindle Store, Buy a Kindle, Furniture & D&#233;cor, #508510, please tell me which class the center node belongs to?"
            else:
                qs = line["conversations"][0]['value']
        elif args.task == "nctext":
            text = data.raw_texts[line['id']]
            text = text[:2000]
            if args.dataset == "arxiv":
                qs = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent papers and edges represent co-citations, the node feature of center node is {text}. We need to classify the center node into 40 classes: cs.NA(Numerical Analysis), cs.MM(Multimedia), cs.LO(Logic in Computer Science), cs.CY(Computers and Society), cs.CR(Cryptography and Security), cs.DC(Distributed, Parallel, and Cluster Computing), cs.HC(Human-Computer Interaction), cs.CE(Computational Engineering, Finance, and Science), cs.NI(Networking and Internet Architecture), cs.CC(Computational Complexity), cs.AI(Artificial Intelligence), cs.MA(Multiagent Systems), cs.GL(General Literature), cs.NE(Neural and Evolutionary Computing), cs.SC(Symbolic Computation), cs.AR(Hardware Architecture), cs.CV(Computer Vision and Pattern Recognition), cs.GR(Graphics), cs.ET(Emerging Technologies), cs.SY(Systems and Control), cs.CG(Computational Geometry), cs.OH(Other Computer Science), cs.PL(Programming Languages), cs.SE(Software Engineering), cs.LG(Machine Learning), cs.SD(Sound), cs.SI(Social and Information Networks), cs.RO(Robotics), cs.IT(Information Theory), cs.PF(Performance), cs.CL(Computational Complexity), cs.IR(Information Retrieval), cs.MS(Mathematical Software), cs.FL(Formal Languages and Automata Theory), cs.DS(Data Structures and Algorithms), cs.OS(Operating Systems), cs.GT(Computer Science and Game Theory), cs.DB(Databases), cs.DL(Digital Libraries), cs.DM(Discrete Mathematics), please tell me which class the center node belongs to? Direct tell me the class name."
            elif args.dataset == "products":
                qs = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent products sold in Amazon, and edges between products indicate they are purchased together, the node feature of center node is {text}. We need to classify the center node into 47 classes: Home & Kitchen, Health & Personal Care, Beauty, Sports & Outdoors, Books, Patio, Lawn & Garden, Toys & Games, CDs & Vinyl, Cell Phones & Accessories, Grocery & Gourmet Food, Arts, Crafts & Sewing, Clothing, Shoes & Jewelry, Electronics, Movies & TV, Software, Video Games, Automotive, Pet Supplies, Office Products, Industrial & Scientific, Musical Instruments, Tools & Home Improvement, Magazine Subscriptions, Baby Products, label 25, Appliances, Kitchen & Dining, Collectibles & Fine Art, All Beauty, Luxury Beauty, Amazon Fashion, Computers, All Electronics, Purchase Circles, MP3 Players & Accessories, Gift Cards, Office & School Supplies, Home Improvement, Camera & Photo, GPS & Navigation, Digital Music, Car Electronics, Baby, Kindle Store, Buy a Kindle, Furniture & D&#233;cor, #508510, please tell me which class the center node belongs to? Direct tell me the class name."
            elif args.dataset == "pubmed":
                qs = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent papers about Diabetes and edges represent co-citations, the node feature of center node is {text}. We need to classify the center node into 3 classes: Diabetes Mellitus Experimental, Diabetes Mellitus Type1, Diabetes Mellitus Type2, please tell me which class the center node belongs to? Direct tell me the class name."
            elif args.dataset == "cora":
                qs = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent papers and edges represent co-citations, the node feature of center node is {text}. We need to classify the center node into 7 classes: Case_Based, Genetic_Algorithms, Neural_Networks, Probabilistic_Methods, Reinforcement_Learning, Rule_Learning, Theory, please tell me which class the center node belongs to? Direct tell me the class name."
            else:
                raise ValueError
        elif args.task == "lp":
            qs=line["conversations"][0]['value']
        else:
            print(f"NOT SUPPORT {args.task}!!!")
            raise ValueError
        cur_prompt = qs
        if args.conv_mode == 'graphllava_llama_3':
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            input_ids = preprocess_llama3(line["conversations"][0], tokenizer, has_graph=True).cuda()
        else:
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_graph_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).cuda()

        if "ND" in args.template or "base" in args.template or args.template == "TIQ" or args.template == "TIQ_semi" or args.template == "TIQ_v2" or args.template == "TIQ_demo" or args.template == "TIQ_cat" or args.template == "TIQ_demo_without_label" or args.template == "TIQ_demo_image_only" or args.template == "TIQ_demo_text_only" or args.template == "TIQ_demo_bootstrap" or args.template == "TIQ_demo_1" or args.template == "TIQ_demo_2":
            if use_neighbors or use_all:
                if not isinstance(line['graph'][0], list):
                    line['graph'] = [line['graph']]
                graph = torch.LongTensor(line['graph'])
                mask = graph != DEFAULT_GRAPH_PAD_ID
                masked_graph_emb = pretrained_graph_emb[graph[mask]]
                s, n, d = graph.shape[0], graph.shape[1], masked_graph_emb.shape[1]
                graph_emb = torch.zeros((s, n, d))
                graph_emb[mask] = masked_graph_emb
                if structure_emb is not None:
                    graph_emb = torch.cat([graph_emb, structure_emb.unsqueeze(0).expand(s, -1, -1)], dim=-1)
                graph = graph.cuda()
            else:
                if args.template == "TIQ_demo" or args.template == "TIQ_demo_without_label" or args.template == "TIQ_demo_image_only" or args.template == "TIQ_demo_text_only" or args.template == "TIQ_demo_bootstrap" or args.template == "TIQ_demo_1" or args.template == "TIQ_demo_2":
                    demo_id = line["demo_id"]
                    id = [idx]
                    all_graph_id = id + demo_id
                    graph_emb_list = []
                    for node_id in all_graph_id:
                        if isinstance(node_id, list):
                            pair_emb_list = []
                            for pair_id in node_id:
                                pair_emb_list.append(pretrained_graph_emb[pair_id])
                            cat_pair_emb = torch.cat(pair_emb_list, dim=0)
                            graph_emb_list.append(cat_pair_emb)
                        else:
                            pair_emb_list = []
                            pair_emb_list.append(pretrained_graph_emb[node_id])
                            pair_emb_list.append(pretrained_graph_emb[node_id])
                            cat_pair_emb = torch.cat(pair_emb_list, dim=0)
                            graph_emb_list.append(cat_pair_emb)
                            # graph_emb_list.append(pretrained_graph_emb[node_id])
                    graph_emb = torch.stack(graph_emb_list, dim=0)

                else:
                    if isinstance(idx, list):
                        graph_emb_list = []
                        for node_id in idx:
                            graph_emb_list.append(pretrained_graph_emb[node_id])
                        if args.template == "TIQ_cat":
                            graph_emb = torch.cat(graph_emb_list, dim=0).unsqueeze(0)
                        else:
                            graph_emb = torch.stack(graph_emb_list, dim=0)
                    else:
                        graph_emb = pretrained_graph_emb[idx].unsqueeze(0)
                graph = None

            
            if use_all:
                node_emb = pretrained_node_emb[idx].unsqueeze(0)
                node_emb = node_emb.half().cuda()
            else:
                node_emb = None
            
            if use_text_cls:
                text_emb = pretrained_text_emb[idx].unsqueeze(0)
                text_emb = text_emb.half().cuda()
            else:
                text_emb =  None
            if not graph_only:
                if args.template == "TIQ_demo" or args.template == "TIQ_demo_without_label" or args.template == "TIQ_demo_image_only" or args.template == "TIQ_demo_text_only" or args.template == "TIQ_demo_bootstrap" or args.template == "TIQ_demo_1" or args.template == "TIQ_demo_2":
                    demo_id = line["demo_id"]
                    id = [idx]
                    all_image_id = id + demo_id
                    image_emb_list = []
                    for node_id in all_image_id:
                        if isinstance(node_id, list):
                            pair_emb_list = []
                            for pair_id in node_id:
                                pair_emb_list.append(pretrained_image_emb[pair_id])
                            cat_pair_emb = torch.stack(pair_emb_list, dim=0)
                            image_emb_list.append(cat_pair_emb)
                        else:
                            part_emb_list = []
                            part_emb_list.append(pretrained_image_emb[node_id])
                            part_emb_list.append(pretrained_image_emb[node_id])
                            cat_pair_emb = torch.stack(part_emb_list, dim=0)
                            image_emb_list.append(cat_pair_emb)
                            # image_emb_list.append(pretrained_image_emb[node_id])
                    image_emb = torch.stack(image_emb_list, dim=0)

                else:
                    if isinstance(idx, list):
                        image_emb_list = []
                        for node_id in idx:
                            image_emb_list.append(pretrained_image_emb[node_id])
                        if args.template == "TIQ_cat":
                            image_emb = torch.stack(image_emb_list, dim=0).unsqueeze(0)
                        else:
                            image_emb = torch.stack(image_emb_list, dim=0)
                    else:
                        image_emb = pretrained_image_emb[idx].unsqueeze(0)
                
                # image_emb = pretrained_image_emb[idx].unsqueeze(0)
                image_emb = image_emb.half().cuda()
            else:
                image_emb = None
        elif args.template == "HO":
            # for small dataset, we remove test link during testing
            # for large dataset, remove test link and compute embedding may be more memory- and time-consuming , we precompute the embedding

            if args.dataset in SMALL_DATASETS and args.task == "lp":
                mp = MP()
                center_nodes = []
                for g in range(len(line['graph'])):
                    center_id = line['graph'][g][0]
                    line['graph'][g] = [center_id] * (args.use_hop + 1)
                    center_nodes.append(center_id)
                graph = torch.LongTensor(line['graph'])
                center_id = graph[:, 0]
                graph_embs = [pretrained_emb[center_id].cuda()]
                subset, edge_index, mapping, edge_mask = k_hop_subgraph(center_nodes, args.use_hop, data.edge_index,
                                                                        relabel_nodes=True)
                local_edge_mask = ((edge_index[0] == mapping[0]) & (edge_index[1] == mapping[1])) | (
                            (edge_index[0] == mapping[1]) & (edge_index[1] == mapping[0]))
                edge_index = edge_index[:, ~local_edge_mask]
                local_x = pretrained_emb[subset].cuda()
                n = subset.shape[0]
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index)
                edge_index = edge_index.cuda()
                row, col = edge_index
                deg = degree(col, n, dtype=pretrained_emb.dtype)
                deg_inv_sqrt = deg.pow(-0.5)
                deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
                norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
                # local_x = pretrained_emb
                for _ in range(args.use_hop):
                    local_x = mp.propagate(edge_index, x=local_x, norm=norm)
                    graph_embs.append(local_x[mapping])
                graph_emb = torch.stack(graph_embs, dim=1)
            else:

                for g in range(len(line['graph'])):
                    center_id = line['graph'][g][0]
                    line['graph'][g] = [center_id]*(args.use_hop+1)
                graph = torch.LongTensor(line['graph'])
                center_id = graph[:, 0]
                graph_emb = torch.stack([emb[index[center_id]] for emb in pretrained_emb], dim=1)
        else:
            raise ValueError

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    graph_emb=graph_emb.half().cuda(),
                    image_emb=image_emb,
                    text_emb=text_emb,
                    node_emb=node_emb,
                    graph=graph,
                    task_type=task_type,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id)

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
        except Exception as e:
            print(f"!!!!!!Error!!!!! {e}")
            outputs=""

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "gt":line["conversations"][1]['value'],
                                   "answer_id": ans_id}) + "\n")
        ans_file.flush()
        # pdb.set_trace()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_base", type=str, default=None)
    # parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--pretrained_embedding_type", type=str, default="clip")
    parser.add_argument("--use_hop", type=int, default=2)
    parser.add_argument("--sample_neighbor_size", type=int, default=5)
    parser.add_argument("--answers_file", type=str, default="answer.jsonl")
    parser.add_argument("--conv_mode", type=str, default="v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--mm_use_graph_start_end",default=False, action="store_true")
    parser.add_argument("--task", type=str, default="nc")
    parser.add_argument("--dataset", type=str, default="arxiv")
    parser.add_argument("--cache_dir", type=str, default="./checkpoint")
    parser.add_argument("--template", type=str, default="ND")
    parser.add_argument("--pretrained_lora_path", type=str, default=None)
    parser.add_argument("--bootstrap", type=int)
    args = parser.parse_args()

    eval_model(args)
