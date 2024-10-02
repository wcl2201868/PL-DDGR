import torch.nn as nn
import torch
import numpy as np
from HGCN import HGNN_conv

from EmbeddingLayer import LinearActivateLayer, AttentionLayer, PredictLayer, Embedding_Fusing_Bi, \
    Embedding_Fusing
from utils.common_utils import matrix_to_sp_tensor


class PL_DDGR(nn.Module):

    def __init__(self, args, ori_num_set, dy_num_set, subgraph_G, subgraph_g_G, reversed_subgraph_mapping_i,
                 reversed_subgraph_mapping_latest_i, reversed_subgraph_mapping_g_i, subgraph_sequence_u,
                 reversed_subgraph_mapping_latest_g_i, sorted_times, sorted_times_g, hyper_layer=2):
        super(PL_DDGR, self).__init__()
        self.args = args
        self.dy_user_num = dy_num_set[0]
        self.dy_item_num = dy_num_set[1]
        self.dy_g_item_num = dy_num_set[3]
        self.user_num = ori_num_set[0]
        self.item_num = ori_num_set[1]
        self.group_num = ori_num_set[2]
        self.g_item_num = ori_num_set[3]
        self.user_ori_embedding = nn.Embedding(args.user_num, args.embedding_size)
        self.item_ori_embedding = nn.Embedding(args.item_num, args.embedding_size)
        self.group_ori_embedding = nn.Embedding(args.group_num, args.embedding_size)
        self.g_item_ori_embedding = nn.Embedding(args.item_g_num, args.embedding_size)
        self.item_dy_embedding = nn.Embedding(args.item_dy_num, args.embedding_size)
        self.g_item_dy_embedding = nn.Embedding(args.g_item_dy_num, args.embedding_size)
        self.attribute_embedding = nn.Embedding(args.attributes_num, args.embedding_size)
        self.subgraph_G = {}
        self.subgraph_g_G = {}
        for t in sorted_times:
            self.subgraph_G[t] = {}
            self.subgraph_G[t]['G'] = matrix_to_sp_tensor(subgraph_G[t]['G']).to(args.device)
            self.subgraph_G[t]['E'] = matrix_to_sp_tensor(subgraph_G[t]['E']).to(args.device)
        for t in sorted_times_g:
            self.subgraph_g_G[t] = {}
            self.subgraph_g_G[t]['G'] = matrix_to_sp_tensor(subgraph_g_G[t]['G']).to(args.device)
            self.subgraph_g_G[t]['E'] = matrix_to_sp_tensor(subgraph_g_G[t]['E']).to(args.device)
        self.reversed_subgraph_mapping_i = reversed_subgraph_mapping_i
        self.reversed_subgraph_mapping_latest_i = reversed_subgraph_mapping_latest_i
        self.reversed_subgraph_mapping_g_i = reversed_subgraph_mapping_g_i
        self.reversed_subgraph_mapping_latest_g_i = reversed_subgraph_mapping_latest_g_i
        self.subgraph_sequence_u = subgraph_sequence_u
        self.sorted_times = sorted_times
        self.sorted_times_g = sorted_times_g
        self.hyper_layers = hyper_layer
        self.pre_embedding_fusing_layer = Embedding_Fusing_Bi(args.embedding_size, args.dropout_ratio)
        self.user_embedding_fusing_layer = Embedding_Fusing(args.embedding_size, args.dropout_ratio)
        self.member_embedding_fusing_layer = Embedding_Fusing_Bi(args.embedding_size, args.dropout_ratio)
        self.group_embedding_fusing_layer = Embedding_Fusing(args.embedding_size, args.dropout_ratio)
        self.aap_norm = nn.Linear(self.args.embedding_size, self.args.embedding_size)
        self.pretrain_predictor = nn.Linear(args.embedding_size, self.args.item_num, bias=False)
        nn.init.xavier_uniform_(self.pretrain_predictor.weight)
        self.user_predictor = nn.Linear(args.embedding_size, self.args.item_num, bias=False)
        nn.init.xavier_uniform_(self.user_predictor.weight)
        self.group_predictor = nn.Linear(args.embedding_size, self.args.item_num, bias=False)
        nn.init.xavier_uniform_(self.group_predictor.weight)
        self.rec_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.cl_loss = nn.BCELoss()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def _generate_ui_embedding(self, args, u_num, i_num, device):
        user_embeds = nn.Embedding(u_num, args.embedding_size).to(device)
        item_embeds = nn.Embedding(i_num, args.embedding_size).to(device)
        return user_embeds, item_embeds

    def _build_ul_hypergraph(self, user_embeds, item_embeds, device):
        all_item_embeds = self.item_dy_embedding.weight
        item_dy_emb_list = item_embeds.clone()
        user_dy_emb_list = user_embeds.clone()

        for t in self.sorted_times:
            item_ori_embeds = item_dy_emb_list[torch.LongTensor(self.reversed_subgraph_mapping_i[t])]
            item_dy_embeds = all_item_embeds[torch.LongTensor(self.reversed_subgraph_mapping_latest_i[t])]
            stack_feature = torch.stack([item_ori_embeds, item_dy_embeds], dim=0)
            # residual gating network
            linear_layer = LinearActivateLayer(self.args.embedding_size, self.args.dropout_ratio).to(device)
            feature_score = linear_layer(stack_feature)
            transformed_inputs = torch.sum(feature_score * stack_feature, dim=0)
            hgnn = HGNN_conv(self.args.embedding_size, self.args.embedding_size, self.args.dropout_ratio,
                             self.hyper_layers, device).to(device)
            nodes, edges = hgnn(transformed_inputs, self.subgraph_G[t])
            item_dy_emb_list = torch.cat([item_dy_emb_list, nodes], dim=0)
            user_dy_emb_list = torch.cat([user_dy_emb_list, edges], dim=0)
        return user_dy_emb_list, item_dy_emb_list

    def _build_gl_hypergraph(self, all_g_embs, all_i_embs, device):
        all_item_embeds = self.g_item_dy_embedding.weight
        item_dy_emb_list = all_i_embs.clone()
        group_dy_emb_list = all_g_embs.clone()
        for t in self.sorted_times_g:
            item_ori_embeds = all_i_embs[torch.LongTensor(self.reversed_subgraph_mapping_g_i[t])]
            item_dy_embeds = all_item_embeds[torch.LongTensor(self.reversed_subgraph_mapping_latest_g_i[t])]
            stack_feature = torch.stack([item_ori_embeds, item_dy_embeds], dim=0)
            # residual gating network
            linear_layer = LinearActivateLayer(self.args.embedding_size, self.args.dropout_ratio).to(device)
            feature_score = linear_layer(stack_feature)
            transformed_inputs = torch.sum(feature_score * stack_feature, dim=0)
            hgnn = HGNN_conv(self.args.embedding_size, self.args.embedding_size, self.args.dropout_ratio,
                             self.hyper_layers, device).to(device)
            nodes, edges = hgnn(transformed_inputs, self.subgraph_g_G[t])
            item_dy_emb_list = torch.cat([item_dy_emb_list, nodes], dim=0)
            group_dy_emb_list = torch.cat([group_dy_emb_list, edges], dim=0)

        return item_dy_emb_list, group_dy_emb_list

    def member_level_group_embed_generation(self, g_list, all_user_dy_embs):

        group_mem_dy_emb = torch.Tensor().to(self.args.device)
        for g in g_list:
            members_list = self.args.gu_dict[g]
            members_dy_list = []
            for m in members_list:
                members_dy_list.append(self.subgraph_sequence_u[m][-1])
            members_ori_embs = self.user_ori_embedding.weight[torch.LongTensor(members_list).to(self.args.device)]
            members_dy_embs = all_user_dy_embs[torch.LongTensor(members_dy_list).to(self.args.device)]
            stack_feature = torch.stack([members_dy_embs, members_ori_embs], dim=0)
            linear_layer = LinearActivateLayer(self.args.embedding_size, self.args.dropout_ratio).to(self.args.device)
            feature_score = linear_layer(stack_feature)
            mem_embs = torch.sum(feature_score * stack_feature, dim=0)
            attention_layer = AttentionLayer(self.args.embedding_size, self.args.dropout_ratio).to(self.args.device)
            at_wt = attention_layer(mem_embs)
            group_members_dy_emb = torch.matmul(at_wt, mem_embs)
            group_mem_dy_emb = torch.cat([group_mem_dy_emb, group_members_dy_emb], dim=0)
        return group_mem_dy_emb

    def contextual_contrastive_learning(self, sequence_output, item_attributes, masked_item_sequence):
        attribute_emb = self.attribute_embedding.weight
        sequence_output = self.aap_norm(sequence_output)
        sequence_output = sequence_output.view([-1, self.args.embedding_size, 1])
        score = torch.matmul(attribute_emb, sequence_output)
        score = torch.sigmoid(score.squeeze(-1))
        cl_loss = self.cl_loss(score, item_attributes.view([-1, self.args.attributes_num]).float())
        cl_loss = torch.sum(cl_loss * masked_item_sequence.flatten().unsqueeze(-1))
        return cl_loss

    def member_level_attention_visualization(self, all_user_dy_embs):
        all_user_dy_embs = all_user_dy_embs.detach()
        outputpath = f'./result/{self.args.dataset_name}_attention_matrix.txt'
        f = open(outputpath, 'w', encoding='utf-8')
        group_nums = 20
        g_list = []
        for _ in range(group_nums):
            g = np.random.randint(1, self.args.group_num)
            if g not in g_list:
                g_list.append(g)
        for g in g_list:
            members_list = self.args.gu_dict[g]
            tmp_str = 'group id: ' + str(g) + '\n'
            f.write(tmp_str)
            tmp_str = str(members_list) + '\n'
            f.write(tmp_str)
            for t in list(self.args.group_train_time)[:-1]:
                members_dy_list = []
                for m in members_list:
                    members_dy_list.append(self.subgraph_sequence_u[m][t])
                members_ori_embs = self.user_ori_embedding.weight[torch.LongTensor(members_list).to(self.args.device)]
                members_dy_embs = all_user_dy_embs[torch.LongTensor(members_dy_list).to(self.args.device)]
                stack_feature = torch.stack([members_dy_embs, members_ori_embs], dim=0)
                linear_layer = LinearActivateLayer(self.args.embedding_size, self.args.dropout_ratio).to(self.args.device)
                feature_score = linear_layer(stack_feature)
                mem_embs = torch.sum(feature_score * stack_feature, dim=0)
                attention_layer = AttentionLayer(self.args.embedding_size, self.args.dropout_ratio).to(self.args.device)
                at_wt = attention_layer(mem_embs)
                tmp_str = f'timestamp{t} attention weight:' + str(at_wt.squeeze().tolist()) + '\n'
                f.write(tmp_str)

    def forward(self, cur_tensors, embedding_set, type_m):
        assert type_m in {'pretrain', 'user', 'group'}
        if type_m == 'pretrain':
            all_u_embs, all_i_embs = self.user_ori_embedding, self.item_ori_embedding
            index, pre_train_user, pre_train_sequence, labels = cur_tensors
            user_embs = all_u_embs(pre_train_user)
            user_embs = user_embs.repeat(self.args.max_sequence_length, 1)
            user_embs = user_embs.reshape(-1, self.args.max_sequence_length, self.args.embedding_size)
            seq_embs = all_i_embs(pre_train_sequence)
            user_preferences = self.pre_embedding_fusing_layer(user_embs, seq_embs)
            user_predicts = self.pretrain_predictor(user_preferences)
            return user_predicts, labels
        elif type_m == 'user':
            all_user_dy_embs, all_item_dy_embs = embedding_set
            _, train_sequences_list, train_sequences_dy_list, train_uid_track, labels = cur_tensors
            train_user_dy_emb = all_user_dy_embs[train_uid_track]
            train_items_dy_emb = all_item_dy_embs[train_sequences_dy_list]
            items_ori_emb = all_item_dy_embs[train_sequences_list]
            user_preferences = self.user_embedding_fusing_layer(train_user_dy_emb, train_items_dy_emb, items_ori_emb)
            user_predicts = self.user_predictor(user_preferences)
            return user_preferences, user_predicts, labels
        else:
            all_user_dy_embs, item_dy_emb_list, group_dy_emb_list, all_gi_embs = embedding_set
            seq_idx, train_sequences_list, train_sequences_dy_list, train_gid_track, labels = cur_tensors
            train_group_dy_emb = group_dy_emb_list[train_gid_track]
            g_list = self.args.train_groups[seq_idx.cpu().data.numpy().copy()].tolist()
            g_mem_dy_emb = self.member_level_group_embed_generation(g_list, all_user_dy_embs)
            g_mem_dy_emb = g_mem_dy_emb.repeat(1, self.args.max_sequence_length, 1)
            g_mem_dy_emb = g_mem_dy_emb.reshape(-1, self.args.max_sequence_length, self.args.embedding_size)
            train_group_dy_emb = self.member_embedding_fusing_layer(train_group_dy_emb, g_mem_dy_emb)
            train_items_dy_emb = item_dy_emb_list[train_sequences_dy_list]
            items_ori_emb = all_gi_embs[train_sequences_list]
            group_preferences = self.group_embedding_fusing_layer(train_group_dy_emb, train_items_dy_emb, items_ori_emb)
            group_predicts = self.group_predictor(group_preferences)
            return group_preferences, group_predicts, labels
