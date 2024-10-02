import torch
import pandas as pd
import numpy as np
from utils import common_utils


def model_evaluate(model, args, eval_dataloder, embedding_set, device, mode='pretrain'):
    model.eval()
    if mode == 'pretrain' or mode == 'user':
        iter_list = np.arange(1, args.user_num)
    else:
        iter_list = np.arange(1, args.group_num)
    pred_list = None
    with torch.no_grad():
        for batch_idx, cur_tensors in enumerate(eval_dataloder):
            cur_tensors = tuple(t.to(device) for t in cur_tensors)
            if mode == 'pretrain':
                user_predicts, labels = model(cur_tensors, None, type_m='pretrain')
                user_scores = user_predicts[:, -1, :]
                user_scores = user_scores.gather(1, labels)
                user_scores = user_scores.cpu().data.numpy().copy()
                if batch_idx == 0:
                    pred_list = user_scores
                else:
                    pred_list = np.append(pred_list, user_scores, axis=0)
            elif mode == 'user':
                user_preferences, user_predicts, labels = model(cur_tensors, embedding_set, type_m='user')
                user_scores = user_predicts[:, -1, :]
                user_scores = user_scores.gather(1, labels)
                user_scores = user_scores.cpu().data.numpy().copy()
                if batch_idx == 0:
                    pred_list = user_scores
                else:
                    pred_list = np.append(pred_list, user_scores, axis=0)
            else:
                group_preferences, group_predicts, labels = model(cur_tensors, embedding_set, type_m='group')
                group_scores = group_predicts[:, -1, :]
                group_scores = group_scores.gather(1, labels)
                group_scores = group_scores.cpu().data.numpy().copy()
                if batch_idx == 0:
                    pred_list = group_scores
                else:
                    pred_list = np.append(pred_list, group_scores, axis=0)
        k_list = [5, 10, 20, 50]
        HT = [0.0 for k in k_list]
        NDCG = [0.0 for k in k_list]
        HT, NDCG = common_utils.calculate_evaluate_metric(args, pred_list, k_list, HT, NDCG, iter_list, mode)
    return HT, NDCG


def partition_group_bins(gu_path):
    df_gu = pd.read_csv(gu_path)
    gu_group = df_gu['group'].value_counts()
    group_bins = {}
    for g, counts in gu_group.items():
        if counts not in group_bins.keys():
            group_bins[counts] = []
            group_bins[counts].append(g)
        else:
            group_bins[counts].append(g)
    return group_bins


def calculate_ndcg_bins(model, args, eval_dataloder, embedding_set, device, group_bins, g_ndcg_bins):
    model.eval()
    iter_list = np.arange(1, args.group_num)
    pred_list = None
    with torch.no_grad():
        for batch_idx, cur_tensors in enumerate(eval_dataloder):
            cur_tensors = tuple(t.to(device) for t in cur_tensors)
            group_preferences, group_predicts, labels = model(cur_tensors, embedding_set, type_m='group')
            group_scores = group_predicts[:, -1, :]
            group_scores = group_scores.gather(1, labels)
            group_scores = group_scores.cpu().data.numpy().copy()
            if batch_idx == 0:
                pred_list = group_scores
            else:
                pred_list = np.append(pred_list, group_scores, axis=0)
        k_list = [5, 10, 20, 50]
        g_ndcg_bins = common_utils.set_ndcg_bins(pred_list, k_list, iter_list, group_bins, g_ndcg_bins)
    return g_ndcg_bins

