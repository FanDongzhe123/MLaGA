import random

import torch
import json
import argparse
# from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import pdb


# def sbert(model_type, device):
#     model = SentenceTransformer(model_type, device=device)
#     return model

def get_sbert_embedding(model_type, texts, device):
    pass
    # if model_type == 'sbert':
    #     model_type = 'all-MiniLM-L6-v2'
    # sbert_model = sbert(model_type, f'cuda:{device}')
    # sbert_embeds = sbert_model.encode(texts, batch_size=8, show_progress_bar=True)
    # return torch.tensor(sbert_embeds)

# def eval_movies_nc(res_path, embedding_type, dataset, mode, epoch):
#     data=torch.load("./dataset/Movies/processed_data.pt")
#     labels=data.text_label
#     short_labels = [l.split('_')[0] for l in labels]
#     ys=data.y.numpy().tolist()


#     all_sample=0
#     overall_correct=0
#     strict_correct=0
#     errors=[]
#     with open(res_path, 'r') as f:
#         for line in f:
#             all_sample+=1
#             res = json.loads(line)
#             ans = res["text"]
#             y=ys[res["question_id"]]
#             short_label = short_labels[res["question_id"]]
#             label=labels[res["question_id"]]
#             if short_label.strip().lower() == ans.strip().lower():
#                 strict_correct+=1
#                 overall_correct+=1
#             elif short_label.lower() in ans.lower() and sum([la.lower() in ans.lower() for la in short_labels])==1:
#                 overall_correct+=1
#             else:
#                 error = {}
#                 error[res["question_id"]] = (label, ans)
#                 errors.append(error)
#             if args.sample > 0 and all_sample >= args.sample:
#                 break
#     overall_acc = overall_correct/all_sample
#     strict_acc = strict_correct / all_sample
#     print(f"Test samples: {all_sample}\nstrict_acc: {strict_acc:.4f}\noverall_acc: {overall_acc:.4f}")
#     with open("./Movies_result.txt", "a") as f:
#         f.write(f"{mode}_epoch_{epoch}_{embedding_type}: Test samples: {all_sample} strict_acc: {strict_acc:.4f} overall_acc: {overall_acc:.4f}\n")

def eval_movies_nc(res_path, embedding_type, dataset, mode, epoch):
    # old version
    # data=torch.load("/vast/df2362/Goodreads/processed_data.pt")
    # labels=data.text_label
    # short_labels = [l.split(' ')[0] for l in labels]
    # ys=data.y.numpy().tolist()
    #
    #
    # all_sample=0
    # overall_correct=0
    # strict_correct=0
    # errors=[]
    # with open(res_path, 'r') as f:
    #     for line in f:
    #         all_sample+=1
    #         res = json.loads(line)
    #         ans = res["text"]
    #         # y=ys[res["question_id"]]
    #         # short_label = short_labels[res["question_id"]]
    #         # label=labels[res["question_id"]]
    #         label = res["gt"]
    #         short_label = label.split('_')[0]
    #         if label.lower().strip() == ans.lower().strip():
    #             strict_correct+=1
    #             overall_correct+=1
    #         # elif short_label.lower() in ans.lower() and sum([la.lower() in ans.lower() for la in short_labels])==1:
    #         #     overall_correct+=1
    #         else:
    #             error = {}
    #             error[res["question_id"]] = (label, ans)
    #             errors.append(error)
    #         if args.sample > 0 and all_sample >= args.sample:
    #             break
    # print(overall_correct)
    # print(all_sample)
    # overall_acc = overall_correct/all_sample
    # strict_acc = strict_correct / all_sample
    # print(f"Test samples: {all_sample}\nstrict_acc: {strict_acc:.4f}\noverall_acc: {overall_acc:.4f}")
    # with open("./Movies_result.txt", "a") as f:
    #     f.write(f"{mode}_epoch_{epoch}_{embedding_type}: Test samples: {all_sample} strict_acc: {strict_acc:.4f} overall_acc: {overall_acc:.4f}\n")

    all_sample=0
    overall_correct=0
    strict_correct=0
    errors=[]
    with open(res_path, 'r') as f:
        for line in f:
            all_sample+=1
            res = json.loads(line)
            ans = res["text"]
            label = res["gt"]
            if label.lower().strip() == ans.lower().strip():
                strict_correct+=1
                overall_correct+=1
            else:
                error = {}
                error[res["question_id"]] = (label, ans)
                errors.append(error)
            if args.sample > 0 and all_sample >= args.sample:
                break
    print(overall_correct)
    print(all_sample)
    overall_acc = overall_correct/all_sample
    strict_acc = strict_correct / all_sample
    print(f"Test samples: {all_sample}\nstrict_acc: {strict_acc:.4f}\noverall_acc: {overall_acc:.4f}")
    with open("./Movies_result.txt", "a") as f:
        f.write(f"{mode}_epoch_{epoch}_{embedding_type}: Test samples: {all_sample} strict_acc: {strict_acc:.4f} overall_acc: {overall_acc:.4f}\n")

def eval_toys_nc(res_path, embedding_type, dataset, mode, epoch):
    data=torch.load("./dataset/Toys/processed_data.pt")
    labels=data.text_label
    short_labels = [l.split(' ')[0] for l in labels]
    ys=data.y.numpy().tolist()


    all_sample=0
    overall_correct=0
    strict_correct=0
    errors=[]
    with open(res_path, 'r') as f:
        for line in f:
            all_sample+=1
            res = json.loads(line)
            ans = res["text"]
            # y=ys[res["question_id"]]
            short_label = short_labels[res["question_id"]]
            label=labels[res["question_id"]]
            if label.lower().strip() == ans.lower().strip():
                strict_correct+=1
                overall_correct+=1
            elif short_label.lower() in ans.lower() and sum([la.lower() in ans.lower() for la in short_labels])==1:
                overall_correct+=1
            else:
                error = {}
                error[res["question_id"]] = (label, ans)
                errors.append(error)
            if args.sample > 0 and all_sample >= args.sample:
                break
    print(overall_correct)
    print(all_sample)
    overall_acc = overall_correct/all_sample
    strict_acc = strict_correct / all_sample
    print(f"Test samples: {all_sample}\nstrict_acc: {strict_acc:.4f}\noverall_acc: {overall_acc:.4f}")
    with open("./Goodreads_result.txt", "a") as f:
        f.write(f"{mode}_epoch_{epoch}_{embedding_type}: Test samples: {all_sample} strict_acc: {strict_acc:.4f} overall_acc: {overall_acc:.4f}\n")


def eval_grocery_nc(res_path, embedding_type):
    data=torch.load("./dataset/Grocery/processed_data.pt")
    labels=data.text_label
    short_labels = [l.split(' ')[0] for l in labels]
    ys=data.y.numpy().tolist()


    all_sample=0
    overall_correct=0
    strict_correct=0
    errors=[]
    with open(res_path, 'r') as f:
        for line in f:
            all_sample+=1
            res = json.loads(line)
            ans = res["text"]
            # y=ys[res["question_id"]]
            short_label = short_labels[res["question_id"]]
            label=labels[res["question_id"]]
            if label.lower().strip() == ans.lower().strip():
                strict_correct+=1
                overall_correct+=1
            elif short_label.lower() in ans.lower() and sum([la.lower() in ans.lower() for la in short_labels])==1:
                overall_correct+=1
            else:
                error = {}
                error[res["question_id"]] = (label, ans)
                errors.append(error)
            if args.sample > 0 and all_sample >= args.sample:
                break
    print(overall_correct)
    print(all_sample)
    overall_acc = overall_correct/all_sample
    strict_acc = strict_correct / all_sample
    print(f"Test samples: {all_sample}\nstrict_acc: {strict_acc:.4f}\noverall_acc: {overall_acc:.4f}")
    with open("./Grocery_result.txt", "a") as f:
        f.write(f"{embedding_type}: Test samples: {all_sample} strict_acc: {strict_acc:.4f} overall_acc: {overall_acc:.4f}\n")
    # with open("./Toys_error.json", "w") as f:
    #     for error in errors:
    #         json.dump(error, f, indent=None)
    #         f.write("\n")  

def eval_health_nc(res_path, embedding_type):
    data=torch.load("./dataset/Health/processed_data.pt")
    labels=data.text_label
    short_labels = [l.split(' ')[0] for l in labels]
    ys=data.y.numpy().tolist()


    all_sample=0
    overall_correct=0
    strict_correct=0
    errors=[]
    with open(res_path, 'r') as f:
        for line in f:
            all_sample+=1
            res = json.loads(line)
            ans = res["text"]
            # y=ys[res["question_id"]]
            short_label = short_labels[res["question_id"]]
            label=labels[res["question_id"]]
            if label.lower().strip() == ans.lower().strip():
                strict_correct+=1
                overall_correct+=1
            elif short_label.lower() in ans.lower() and sum([la.lower() in ans.lower() for la in short_labels])==1:
                overall_correct+=1
            else:
                error = {}
                error[res["question_id"]] = (label, ans)
                errors.append(error)
            if args.sample > 0 and all_sample >= args.sample:
                break
    print(overall_correct)
    print(all_sample)
    overall_acc = overall_correct/all_sample
    strict_acc = strict_correct / all_sample
    print(f"Test samples: {all_sample}\nstrict_acc: {strict_acc:.4f}\noverall_acc: {overall_acc:.4f}")
    with open("./Health_result.txt", "a") as f:
        f.write(f"{embedding_type}: Test samples: {all_sample} strict_acc: {strict_acc:.4f} overall_acc: {overall_acc:.4f}\n")

def eval_beauty_nc(res_path, embedding_type):
    data=torch.load("./dataset/Beauty/processed_data.pt")
    labels=data.text_label
    short_labels = [l.split(' ')[0] for l in labels]
    ys=data.y.numpy().tolist()


    all_sample=0
    overall_correct=0
    strict_correct=0
    errors=[]
    with open(res_path, 'r') as f:
        for line in f:
            all_sample+=1
            res = json.loads(line)
            ans = res["text"]
            # y=ys[res["question_id"]]
            short_label = short_labels[res["question_id"]]
            label=labels[res["question_id"]]
            if label.lower().strip() == ans.lower().strip():
                strict_correct+=1
                overall_correct+=1
            elif short_label.lower() in ans.lower() and sum([la.lower() in ans.lower() for la in short_labels])==1:
                overall_correct+=1
            else:
                error = {}
                error[res["question_id"]] = (label, ans)
                errors.append(error)
            if args.sample > 0 and all_sample >= args.sample:
                break
    print(overall_correct)
    print(all_sample)
    overall_acc = overall_correct/all_sample
    strict_acc = strict_correct / all_sample
    print(f"Test samples: {all_sample}\nstrict_acc: {strict_acc:.4f}\noverall_acc: {overall_acc:.4f}")
    with open("./Beauty_result.txt", "a") as f:
        f.write(f"{embedding_type}: Test samples: {all_sample} strict_acc: {strict_acc:.4f} overall_acc: {overall_acc:.4f}\n")

def eval_videogames_nc(res_path, embedding_type, dataset, mode, epoch):
    data=torch.load("./dataset/VideoGames/processed_data.pt")
    labels=data.text_label
    short_labels = [l.split(' ')[0] for l in labels]
    ys=data.y.numpy().tolist()


    all_sample=0
    overall_correct=0
    strict_correct=0
    errors=[]
    with open(res_path, 'r') as f:
        for line in f:
            all_sample+=1
            res = json.loads(line)
            ans = res["text"]
            # y=ys[res["question_id"]]
            short_label = short_labels[res["question_id"]]
            label=labels[res["question_id"]]
            if label.lower().strip() == ans.lower().strip():
                strict_correct+=1
                overall_correct+=1
            elif short_label.lower() in ans.lower() and sum([la.lower() in ans.lower() for la in short_labels])==1:
                overall_correct+=1
            else:
                error = {}
                error[res["question_id"]] = (label, ans)
                errors.append(error)
            if args.sample > 0 and all_sample >= args.sample:
                break
    print(overall_correct)
    print(all_sample)
    overall_acc = overall_correct/all_sample
    strict_acc = strict_correct / all_sample
    print(f"Test samples: {all_sample}\nstrict_acc: {strict_acc:.4f}\noverall_acc: {overall_acc:.4f}")
    with open("./VideoGames_result.txt", "a") as f:
        f.write(f"{mode}_epoch_{epoch}_{embedding_type}: Test samples: {all_sample} strict_acc: {strict_acc:.4f} overall_acc: {overall_acc:.4f}\n")

def eval_beauty_nc(res_path, embedding_type):
    data=torch.load("./dataset/Beauty/processed_data.pt")
    labels=data.text_label
    short_labels = [l.split(' ')[0] for l in labels]
    ys=data.y.numpy().tolist()


    all_sample=0
    overall_correct=0
    strict_correct=0
    errors=[]
    with open(res_path, 'r') as f:
        for line in f:
            all_sample+=1
            res = json.loads(line)
            ans = res["text"]
            # y=ys[res["question_id"]]
            short_label = short_labels[res["question_id"]]
            label=labels[res["question_id"]]
            if label.lower().strip() == ans.lower().strip():
                strict_correct+=1
                overall_correct+=1
            elif short_label.lower() in ans.lower() and sum([la.lower() in ans.lower() for la in short_labels])==1:
                overall_correct+=1
            else:
                error = {}
                error[res["question_id"]] = (label, ans)
                errors.append(error)
            if args.sample > 0 and all_sample >= args.sample:
                break
    print(overall_correct)
    print(all_sample)
    overall_acc = overall_correct/all_sample
    strict_acc = strict_correct / all_sample
    print(f"Test samples: {all_sample}\nstrict_acc: {strict_acc:.4f}\noverall_acc: {overall_acc:.4f}")
    with open("./Beauty_result.txt", "a") as f:
        f.write(f"{embedding_type}: Test samples: {all_sample} strict_acc: {strict_acc:.4f} overall_acc: {overall_acc:.4f}\n")

def eval_health_nc(res_path, embedding_type):
    data=torch.load("./dataset/Health/processed_data.pt")
    labels=data.text_label
    short_labels = [l.split(' ')[0] for l in labels]
    ys=data.y.numpy().tolist()


    all_sample=0
    overall_correct=0
    strict_correct=0
    errors=[]
    with open(res_path, 'r') as f:
        for line in f:
            all_sample+=1
            res = json.loads(line)
            ans = res["text"]
            # y=ys[res["question_id"]]
            short_label = short_labels[res["question_id"]]
            label=labels[res["question_id"]]
            if label.lower().strip() == ans.lower().strip():
                strict_correct+=1
                overall_correct+=1
            elif short_label.lower() in ans.lower() and sum([la.lower() in ans.lower() for la in short_labels])==1:
                overall_correct+=1
            else:
                error = {}
                error[res["question_id"]] = (label, ans)
                errors.append(error)
            if args.sample > 0 and all_sample >= args.sample:
                break
    print(overall_correct)
    print(all_sample)
    overall_acc = overall_correct/all_sample
    strict_acc = strict_correct / all_sample
    print(f"Test samples: {all_sample}\nstrict_acc: {strict_acc:.4f}\noverall_acc: {overall_acc:.4f}")
    with open("./Health_result.txt", "a") as f:
        f.write(f"{embedding_type}: Test samples: {all_sample} strict_acc: {strict_acc:.4f} overall_acc: {overall_acc:.4f}\n")

def eval_automotive_nc(res_path, embedding_type):
    data=torch.load("./dataset/Automotive/processed_data.pt")
    labels=data.text_label
    short_labels = [l.split(' ')[0] for l in labels]
    ys=data.y.numpy().tolist()


    all_sample=0
    overall_correct=0
    strict_correct=0
    errors=[]
    with open(res_path, 'r') as f:
        for line in f:
            all_sample+=1
            res = json.loads(line)
            ans = res["text"]
            # y=ys[res["question_id"]]
            short_label = short_labels[res["question_id"]]
            label=labels[res["question_id"]]
            if label.lower().strip() == ans.lower().strip():
                strict_correct+=1
                overall_correct+=1
            elif short_label.lower() in ans.lower() and sum([la.lower() in ans.lower() for la in short_labels])==1:
                overall_correct+=1
            else:
                error = {}
                error[res["question_id"]] = (label, ans)
                errors.append(error)
            if args.sample > 0 and all_sample >= args.sample:
                break
    print(overall_correct)
    print(all_sample)
    overall_acc = overall_correct/all_sample
    strict_acc = strict_correct / all_sample
    print(f"Test samples: {all_sample}\nstrict_acc: {strict_acc:.4f}\noverall_acc: {overall_acc:.4f}")
    with open("./Automotive_result.txt", "a") as f:
        f.write(f"{embedding_type}: Test samples: {all_sample} strict_acc: {strict_acc:.4f} overall_acc: {overall_acc:.4f}\n")

# def eval_arts_nc(res_path, embedding_type, dataset, mode, epoch):
#     data=torch.load("./dataset/Arts/processed_data.pt")
#     labels=data.text_label
#     short_labels = [l.split(' ')[0] for l in labels]
#     ys=data.y.numpy().tolist()


#     all_sample=0
#     overall_correct=0
#     strict_correct=0
#     errors=[]
#     with open(res_path, 'r') as f:
#         for line in f:
#             all_sample+=1
#             res = json.loads(line)
#             ans = res["text"]
#             # y=ys[res["question_id"]]
#             short_label = short_labels[res["question_id"]]
#             label=labels[res["question_id"]]
#             if label.lower().strip() == ans.lower().strip():
#                 strict_correct+=1
#                 overall_correct+=1
#             elif short_label.lower() in ans.lower() and sum([la.lower() in ans.lower() for la in short_labels])==1:
#                 overall_correct+=1
#             else:
#                 error = {}
#                 error[res["question_id"]] = (label, ans)
#                 errors.append(error)
#             if args.sample > 0 and all_sample >= args.sample:
#                 break
#     print(overall_correct)
#     print(all_sample)
#     overall_acc = overall_correct/all_sample
#     strict_acc = strict_correct / all_sample
#     print(f"Test samples: {all_sample}\nstrict_acc: {strict_acc:.4f}\noverall_acc: {overall_acc:.4f}")
#     with open("./Arts_result.txt", "a") as f:
#         f.write(f"{mode}_epoch_{epoch}_{embedding_type}: Test samples: {all_sample} strict_acc: {strict_acc:.4f} overall_acc: {overall_acc:.4f}\n")

def eval_arts_nc(res_path, embedding_type, dataset, mode, epoch):
    # old version
    # data=torch.load("/vast/df2362/Goodreads/processed_data.pt")
    # labels=data.text_label
    # short_labels = [l.split(' ')[0] for l in labels]
    # ys=data.y.numpy().tolist()
    #
    #
    # all_sample=0
    # overall_correct=0
    # strict_correct=0
    # errors=[]
    # with open(res_path, 'r') as f:
    #     for line in f:
    #         all_sample+=1
    #         res = json.loads(line)
    #         ans = res["text"]
    #         # y=ys[res["question_id"]]
    #         # short_label = short_labels[res["question_id"]]
    #         # label=labels[res["question_id"]]
    #         label = res["gt"]
    #         short_label = label.split('_')[0]
    #         if label.lower().strip() == ans.lower().strip():
    #             strict_correct+=1
    #             overall_correct+=1
    #         # elif short_label.lower() in ans.lower() and sum([la.lower() in ans.lower() for la in short_labels])==1:
    #         #     overall_correct+=1
    #         else:
    #             error = {}
    #             error[res["question_id"]] = (label, ans)
    #             errors.append(error)
    #         if args.sample > 0 and all_sample >= args.sample:
    #             break
    # print(overall_correct)
    # print(all_sample)
    # overall_acc = overall_correct/all_sample
    # strict_acc = strict_correct / all_sample
    # print(f"Test samples: {all_sample}\nstrict_acc: {strict_acc:.4f}\noverall_acc: {overall_acc:.4f}")
    # with open("./Arts_result.txt", "a") as f:
    #     f.write(f"{mode}_epoch_{epoch}_{embedding_type}: Test samples: {all_sample} strict_acc: {strict_acc:.4f} overall_acc: {overall_acc:.4f}\n")

    all_sample=0
    overall_correct=0
    strict_correct=0
    errors=[]
    with open(res_path, 'r') as f:
        for line in f:
            all_sample+=1
            res = json.loads(line)
            ans = res["text"]
            label = res["gt"]
            if label.lower().strip() == ans.lower().strip():
                strict_correct+=1
                overall_correct+=1
            else:
                error = {}
                error[res["question_id"]] = (label, ans)
                errors.append(error)
            if args.sample > 0 and all_sample >= args.sample:
                break
    print(overall_correct)
    print(all_sample)
    overall_acc = overall_correct/all_sample
    strict_acc = strict_correct / all_sample
    print(f"Test samples: {all_sample}\nstrict_acc: {strict_acc:.4f}\noverall_acc: {overall_acc:.4f}")
    with open("./Arts_result.txt", "a") as f:
        f.write(f"{mode}_epoch_{epoch}_{embedding_type}: Test samples: {all_sample} strict_acc: {strict_acc:.4f} overall_acc: {overall_acc:.4f}\n")

def eval_cd_nc(res_path, embedding_type):
    data=torch.load("./dataset/CD/processed_data.pt")
    labels=data.text_label
    short_labels = [l.split(' ')[0] for l in labels]
    ys=data.y.numpy().tolist()


    all_sample=0
    overall_correct=0
    strict_correct=0
    errors=[]
    with open(res_path, 'r') as f:
        for line in f:
            all_sample+=1
            res = json.loads(line)
            ans = res["text"]
            # y=ys[res["question_id"]]
            short_label = short_labels[res["question_id"]]
            label=labels[res["question_id"]]
            if label.lower().strip() == ans.lower().strip():
                strict_correct+=1
                overall_correct+=1
            elif short_label.lower() in ans.lower() and sum([la.lower() in ans.lower() for la in short_labels])==1:
                overall_correct+=1
            else:
                error = {}
                error[res["question_id"]] = (label, ans)
                errors.append(error)
            if args.sample > 0 and all_sample >= args.sample:
                break
    print(overall_correct)
    print(all_sample)
    overall_acc = overall_correct/all_sample
    strict_acc = strict_correct / all_sample
    print(f"Test samples: {all_sample}\nstrict_acc: {strict_acc:.4f}\noverall_acc: {overall_acc:.4f}")
    with open("./CD_result.txt", "a") as f:
        f.write(f"{embedding_type}: Test samples: {all_sample} strict_acc: {strict_acc:.4f} overall_acc: {overall_acc:.4f}\n")


def eval_reddits_nc(res_path, embedding_type, dataset, mode, epoch):
    data=torch.load("./dataset/RedditS/processed_data.pt")
    labels=data.text_label
    short_labels = [l.split(' ')[0] for l in labels]
    ys=data.y.numpy().tolist()


    all_sample=0
    overall_correct=0
    strict_correct=0
    errors=[]
    with open(res_path, 'r') as f:
        for line in f:
            all_sample+=1
            res = json.loads(line)
            ans = res["text"]
            # y=ys[res["question_id"]]
            short_label = short_labels[res["question_id"]]
            label=labels[res["question_id"]]
            if label.lower().strip() == ans.lower().strip():
                strict_correct+=1
                overall_correct+=1
            elif short_label.lower() in ans.lower() and sum([la.lower() in ans.lower() for la in short_labels])==1:
                overall_correct+=1
            else:
                error = {}
                error[res["question_id"]] = (label, ans)
                errors.append(error)
            if args.sample > 0 and all_sample >= args.sample:
                break
    print(overall_correct)
    print(all_sample)
    overall_acc = overall_correct/all_sample
    strict_acc = strict_correct / all_sample
    print(f"Test samples: {all_sample}\nstrict_acc: {strict_acc:.4f}\noverall_acc: {overall_acc:.4f}")
    with open("./RedditS_result.txt", "a") as f:
        f.write(f"{mode}_epoch_{epoch}_{embedding_type}: Test samples: {all_sample} strict_acc: {strict_acc:.4f} overall_acc: {overall_acc:.4f}\n")

def eval_lp(res_path, embedding_type, dataset, mode, epoch):
    all_sample=0
    correct = 0
    with open(res_path, 'r') as f:
        for line in f:
            res = json.loads(line)
            ans = res["text"].strip()
            label=res["gt"].strip()
            all_sample += 1
            if ("Yes" in ans and "Yes" in label) or ("Yes" not in ans and "No" in label):
                correct += 1
            if args.sample > 0 and all_sample >=  args.sample:
                break
    acc = correct / all_sample
    print(f"Test samples: {all_sample}\ncorrect: {correct}\n acc: {acc:.4f}")
    with open(f"./{dataset}_lp_result.txt", "a") as f:
        f.write(f"{mode}_epoch_{epoch}_{embedding_type}: Test samples: Test samples: {all_sample}\ncorrect: {correct}\n acc: {acc:.4f}\n")

def eval_goodreads_nc(res_path, embedding_type, dataset, mode, epoch):
    # data=torch.load("/vast/df2362/Goodreads/processed_data.pt")
    # labels=data.text_label
    # short_labels = [l.split(' ')[0] for l in labels]
    # ys=data.y.numpy().tolist()


    all_sample=0
    overall_correct=0
    strict_correct=0
    errors=[]
    with open(res_path, 'r') as f:
        for line in f:
            all_sample+=1
            res = json.loads(line)
            ans = res["text"]
            # y=ys[res["question_id"]]
            # short_label = short_labels[res["question_id"]]
            # label=labels[res["question_id"]]
            label = res["gt"]
            short_label = label.split('_')[0]
            if label.lower().strip() == ans.lower().strip():
                strict_correct+=1
                overall_correct+=1
            # elif short_label.lower() in ans.lower() and sum([la.lower() in ans.lower() for la in short_labels])==1:
            #     overall_correct+=1
            else:
                error = {}
                error[res["question_id"]] = (label, ans)
                errors.append(error)
            if args.sample > 0 and all_sample >= args.sample:
                break
    print(overall_correct)
    print(all_sample)
    overall_acc = overall_correct/all_sample
    strict_acc = strict_correct / all_sample
    print(f"Test samples: {all_sample}\nstrict_acc: {strict_acc:.4f}\noverall_acc: {overall_acc:.4f}")
    with open("./Goodreads_result.txt", "a") as f:
        f.write(f"{mode}_epoch_{epoch}_{embedding_type}: Test samples: {all_sample} strict_acc: {strict_acc:.4f} overall_acc: {overall_acc:.4f}\n")

def eval_reddit_nc(res_path, embedding_type, dataset, mode, epoch):
    # data=torch.load("./dataset/RedditS/processed_data.pt")
    # labels=data.text_label
    # short_labels = [l.split(' ')[0] for l in labels]
    # ys=data.y.numpy().tolist()


    all_sample=0
    overall_correct=0
    strict_correct=0
    errors=[]
    with open(res_path, 'r') as f:
        for line in f:
            all_sample+=1
            res = json.loads(line)
            ans = res["text"]
            # y=ys[res["question_id"]]
            # short_label = short_labels[res["question_id"]]
            # label=labels[res["question_id"]]
            label = res["gt"]
            short_label = label.split('_')[0]
            if label.lower().strip() == ans.lower().strip():
                strict_correct+=1
                overall_correct+=1
            # elif short_label.lower() in ans.lower() and sum([la.lower() in ans.lower() for la in short_labels])==1:
            #     overall_correct+=1
            else:
                error = {}
                error[res["question_id"]] = (label, ans)
                errors.append(error)
            if args.sample > 0 and all_sample >= args.sample:
                break
    print(overall_correct)
    print(all_sample)
    overall_acc = overall_correct/all_sample
    strict_acc = strict_correct / all_sample
    print(f"Test samples: {all_sample}\nstrict_acc: {strict_acc:.4f}\noverall_acc: {overall_acc:.4f}")
    with open("./Reddit_result.txt", "a") as f:
        f.write(f"{mode}_epoch_{epoch}_{embedding_type}: Test samples: {all_sample} strict_acc: {strict_acc:.4f} overall_acc: {overall_acc:.4f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_path", type=str, default="./results/llaga-opt-2.7b-v1-simteg_all_origin_tape_multihop-laplacian_-1-2-10-linear-only-train-pretrain_acc1_nc_test_nc.jsonl")
    parser.add_argument("--task", type=str, default="nc")
    parser.add_argument("--dataset", type=str, default="arxiv")
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--embedding_type", type=str, default="blip")
    parser.add_argument("--mode", type=str, default="data&task generalization")
    parser.add_argument("--epoch", type=int, default=1)
    args = parser.parse_args()

    func_dict = {
        "Movies":{
            "nc": eval_movies_nc,
            "lp": eval_lp

        },
        "Toys":{
            "nc": eval_toys_nc,
            "lp": eval_lp

        },
        "Grocery":{
            "nc": eval_grocery_nc,
            "lp": eval_lp

        },
        "VideoGames":{
            "nc": eval_videogames_nc,
            "lp": eval_lp

        },
        "Beauty":{
            "nc": eval_beauty_nc,
            "lp": eval_lp

        },
        "Health":{
            "nc": eval_health_nc,
            "lp": eval_lp

        },
        "CD":{
            "nc": eval_cd_nc,
            "lp": eval_lp

        },
        "Arts":{
            "nc": eval_arts_nc,
            "lp": eval_lp

        },
        "Automotive":{
            "nc": eval_automotive_nc,
            "lp": eval_lp

        },
        "RedditS":{
            "nc": eval_reddits_nc,
            "lp": eval_lp
        },
        "Goodreads":{
            "nc": eval_goodreads_nc,
            "lp": eval_lp
        },
        "Art500K":{
            "lp": eval_lp
        },
        "Reddit":{
            "nc": eval_reddit_nc,
            "lp": eval_lp
        }
    }

    func=func_dict[args.dataset][args.task]
    func(args.res_path, args.embedding_type, args.dataset, args.mode, args.epoch)

#python eval_res.py --res_path ../../res/Movies_base.jsonl --task nc --dataset Movies