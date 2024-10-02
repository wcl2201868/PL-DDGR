import csv
import random

import numpy as np
import pandas as pd
import gzip
import tqdm
import scipy.sparse as sp
import json
import datetime
import gc
from collections import defaultdict
import scipy.io as sio

temp_u = end_flag = 0
temp_group_list = []
indice_matrix = []
adj_matrix = []
gl_u_group_list = {}
temp_u2_list = []
refuse_count_list = []
refuse_max_len = 10000
max_group_size = 14

def parse(path): # for Amazon
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

def read_meta(dataset_name):
    datas = {}
    meta_flie = dataset_name + '.json.gz'
    i = 0
    for info in parse(meta_flie):
        datas[info['asin']] = info
        i += 1
        if i > 50:
            break
    return datas

# return (user item timestamp) sort in get_interaction
def Amazon(dataset_name, rating_score):
    '''
    reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
    asin - ID of the product, e.g. 0000013714
    reviewerName - name of the reviewer
    helpful - helpfulness rating of the review, e.g. 2/3
    --"helpful": [2, 3],
    reviewText - text of the review
    --"reviewText": "I bought this for my husband who plays the piano. ..."
    overall - rating of the product
    --"overall": 5.0,
    summary - summary of the review
    --"summary": "Heavenly Highway Hymns",
    unixReviewTime - time of the review (unix time)
    --"unixReviewTime": 1252800000,
    reviewTime - time of the review (raw)
    --"reviewTime": "09 13, 2009"
    '''
    datas = []
    # older Amazon
    data_flie = dataset_name + '.json.gz'
    # latest Amazon
    # data_flie = '/home/hui_wang/data/new_Amazon/' + dataset_name + '.json.gz'
    for inter in parse(data_flie):
        if float(inter['overall']) <= rating_score: # 小于一定分数去掉
            continue
        user = inter['reviewerID']
        item = inter['asin']
        time = inter['unixReviewTime']
        datas.append((user, item, int(time)))
    return datas


def Amazon_meta(dataset_name, df_ui):
    datas = {}
    meta_flie = dataset_name + '.json.gz'
    item_asins = df_ui['item_asin'].unique().tolist()
    for i in item_asins:
        datas[i] = {}
    for info in parse(meta_flie):
        if info['asin'] not in item_asins:
            continue
        datas[info['asin']] = info
    return datas


def add_comma(num):
    # 1000000 -> 1,000,000
    str_num = str(num)
    res_num = ''
    for i in range(len(str_num)):
        res_num += str_num[i]
        if (len(str_num)-i-1) % 3 == 0:
            res_num += ','
    return res_num[:-1]



def get_attribute_Amazon(meta_infos, item2id, attribute_core):
    attributes = defaultdict(int)
    for iid, info in meta_infos.items():
        if info:
            for cates in info['categories']:
                for cate in cates[1:]:
                    attributes[cate] += 1
            try:
                attributes[info['brand']] += 1
            except:
                pass
    print(f'before delete, attribute num:{len(attributes)}')
    new_meta = {}
    for iid, info in meta_infos.items():
        new_meta[iid] = []
        if info:
            try:
                if attributes[info['brand']] >= attribute_core:
                    new_meta[iid].append(info['brand'])
            except:
                pass
            for cates in info['categories']:
                for cate in cates[1:]:
                    if attributes[cate] >= attribute_core:
                        new_meta[iid].append(cate)
    attribute2id = {}
    id2attribute = {}
    attributeid2num = defaultdict(int)
    attribute_id = 1
    items2attributes = {}
    attribute_lens = []
    for item in item2id.keys():
        items2attributes[item2id[item]] = []
    for iid, attributes in new_meta.items():
        item_id = item2id[iid]
        for attribute in attributes:
            if attribute not in attribute2id:
                attribute2id[attribute] = attribute_id
                id2attribute[attribute_id] = attribute
                attribute_id += 1
            attributeid2num[attribute2id[attribute]] += 1
            items2attributes[item_id].append(attribute2id[attribute])
        attribute_lens.append(len(items2attributes[item_id]))
    print(f'before delete, attribute num:{len(attribute2id)}')
    print(f'attributes len, Min:{np.min(attribute_lens)}, Max:{np.max(attribute_lens)}, Avg.:{np.mean(attribute_lens):.4f}')
    return len(attribute2id), np.mean(attribute_lens), items2attributes


def get_interaction(datas):
    user_seq = {}
    for data in datas:
        user, item, time = data
        if user in user_seq:
            user_seq[user].append((item, time))
        else:
            user_seq[user] = []
            user_seq[user].append((item, time))

    for user, item_time in user_seq.items():
        item_time.sort(key=lambda x: x[1])  # 对各个数据集得单独排序
        user_seq[user] = item_time
    return user_seq

def check_Kcore(user_items, user_core, item_core):
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for user, items in user_items.items():
        for item in items:
            user_count[user] += 1
            item_count[item] += 1

    for user, num in user_count.items():
        if num < user_core:
            return user_count, item_count, False
    for item, num in item_count.items():
        if num < item_core:
            return user_count, item_count, False
    return user_count, item_count, True

def filter_Kcore(user_items, user_core, item_core):
    user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    while not isKcore:
        for user, num in user_count.items():
            if user_count[user] < user_core:
                user_items.pop(user)
            else:
                for item in user_items[user]:
                    if item_count[item] < item_core:
                        user_items[user].remove(item)
        user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    return user_items


def id_map(user_items):

    user2id = {}
    item2id = {}
    id2user = {}
    id2item = {}
    user_id = 1
    item_id = 1
    final_data = {}
    for user, items in user_items.items():
        if user not in user2id:
            user2id[user] = str(user_id)
            id2user[str(user_id)] = user
            user_id += 1
        iids = [] # item id lists
        for item in items:
            if item not in item2id:
                item2id[item] = str(item_id)
                id2item[str(item_id)] = item
                item_id += 1
            iids.append(item2id[item])
        uid = user2id[user]
        final_data[uid] = iids
    data_maps = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item
    }
    return final_data, user_id-1, item_id-1, data_maps


def main(dataset_name, data_type='Amazon'):
    assert data_type == 'Amazon'
    assert dataset_name in {'Toys_and_Games', 'Sports_and_Outdoors', 'Beauty'}
    np.random.seed(12345)
    user_core = 5
    item_core = 5
    rating_score = 3.0

    attribute_core = 0

    data_path = './'
    data_name = data_path + dataset_name
    datas = Amazon(data_name + '/reviews_' + dataset_name + '_5',  rating_score=rating_score)
    users_list = []
    items_list = []
    times_list = []
    user_item_time = get_interaction(datas)
    for u, item_time in user_item_time.items():
        for it in range(len(item_time)):
            users_list.append(u)
            items_list.append(item_time[it][0])
            times_list.append(item_time[it][1])
    assert len(users_list) == len(items_list) == len(times_list)
    df_ui = pd.DataFrame({'user_asin': users_list, 'item_asin': items_list, 'timestamp': times_list})
    df_ui = df_ui.drop_duplicates()
    df_ui['timestamp'] = pd.to_datetime(df_ui['timestamp'], unit='s')
    df_u_group = df_ui.groupby(['user_asin'])
    user_items = {}
    for u, u_group in df_u_group:
        user_items[u] = u_group['item_asin'].tolist()
    print(f'{data_name} Raw data has been processed! Lower than {rating_score} are deleted!')
    # raw_id user: [item1, item2, item3...]
    user_items = filter_Kcore(user_items, user_core=user_core, item_core=item_core)
    print(f'User {user_core}-core complete! Item {item_core}-core complete!')
    # change to dataframe type and merge
    users_list = []
    items_list = []
    for u, items in user_items.items():
        for item in items:
            users_list.append(u)
            items_list.append(item)
    df_ui_remain = pd.DataFrame({'user_asin': users_list, 'item_asin': items_list})
    df_ui = df_ui.merge(df_ui_remain, on=['user_asin', 'item_asin'])
    df_ui = df_ui.drop_duplicates()
    users_count = df_ui['user_asin'].value_counts()
    items_count = df_ui['item_asin'].value_counts()
    print("reorder the index!")
    # reorder the user and item index number from 1
    itemid = np.arange(1, items_count.shape[0] + 1)
    user_idx = pd.DataFrame({'user_asin': users_count.index, 'user': np.arange(1, users_count.shape[0] + 1)})
    item_idx = pd.DataFrame({'item_asin': items_count.index, 'item': itemid})
    item2id = {}
    for idx, item in enumerate(items_count.index):
        item2id[item] = int(itemid[idx])
    df_ui = df_ui.merge(user_idx).merge(item_idx)
    df_ui = df_ui.sort_values(by=['user', 'timestamp'])
    user_num = users_count.shape[0]
    item_num = items_count.shape[0]
    user_avg = np.mean(users_count.values)
    user_min = np.min(users_count.values)
    user_max = np.max(users_count.values)
    item_avg = np.mean(items_count.values)
    item_min = np.min(items_count.values)
    item_max = np.max(items_count.values)
    interact_num = np.sum([x for x in users_count.values])
    sparsity = (1 - interact_num / (user_num * item_num)) * 100
    show_info = f'Total User: {user_num}, Avg User: {user_avg:.4f}, Min Len: {user_min}, Max Len: {user_max}\n' + \
                f'Total Item: {item_num}, Avg Item: {item_avg:.4f}, Min Inter: {item_min}, Max Inter: {item_max}\n' + \
                f'Iteraction Num: {interact_num}, Sparsity: {sparsity:.2f}%'
    print(show_info)
    print('Begin extracting meta infos...')

    if data_type == 'Amazon':
        d_path = str(data_name + '/meta_' + dataset_name)
        meta_infos = Amazon_meta(d_path, df_ui)
        attribute_num, avg_attribute, item2attributes = get_attribute_Amazon(meta_infos, item2id, attribute_core)

    print(f'{data_name} & {add_comma(user_num)}& {add_comma(item_num)} & {user_avg:.1f}'
          f'& {item_avg:.1f}& {add_comma(interact_num)}& {sparsity:.2f}\%&{add_comma(attribute_num)}&'
          f'{avg_attribute:.1f} \\')

    # -------------- Save Data ---------------
    data_file = data_name + '_ui.csv'
    item2attributes_file = data_name + '_item2attributes.json'
    item2id_file = data_name + '_item2ids.json'
    df_ui.to_csv(data_file, index=False)
    json_str = json.dumps(item2attributes)
    with open(item2attributes_file, 'w') as out:
        out.write(json_str)
    json_str = json.dumps(item2id)
    with open(item2id_file, 'w') as out:
        out.write(json_str)


def fit_ui_data(ui_data_path, output_path, user_core, item_core):
    df_ui = pd.DataFrame(pd.read_csv(ui_data_path))
    users_count = df_ui['user'].value_counts()
    items_count = df_ui['item'].value_counts()
    users_count = users_count[users_count >= user_core]
    items_count = items_count[items_count >= item_core]
    df_ui = df_ui.merge(pd.DataFrame({'user': users_count.index})).merge(pd.DataFrame({'item': items_count.index}))
    user_items = {}
    for u, df in df_ui.groupby('user'):
        user_items[int(u)] = df['item'].tolist()
    user_items = filter_Kcore(user_items, user_core, item_core)
    # change to dataframe type and merge
    users_list = []
    items_list = []
    for u, items in user_items.items():
        for item in items:
            users_list.append(u)
            items_list.append(item)
    df_ui_remain = pd.DataFrame({'user': users_list, 'item': items_list})
    df_ui = df_ui.merge(df_ui_remain, on=['user', 'item'])
    df_ui = df_ui.drop_duplicates()
    users_count = df_ui['user'].value_counts()
    items_count = df_ui['item'].value_counts()
    itemid = np.arange(1, items_count.shape[0] + 1)
    item_newitem_dict = {}
    for idx, i in enumerate(items_count.index):
        item_newitem_dict[i] = int(itemid[idx])
    print("reorder the index!")
    # reorder the user and item index number from 1
    user_idx = pd.DataFrame({'user': users_count.index, 'user_new': np.arange(1, users_count.shape[0] + 1)})
    item_idx = pd.DataFrame({'item': items_count.index, 'item_new': itemid})
    df_ui = df_ui.merge(user_idx).merge(item_idx)
    df_ui = pd.DataFrame(df_ui, columns=['user_asin', 'item_asin', 'user_new', 'item_new', 'timestamp'])
    df_ui = df_ui.rename(columns={'user_new': 'user', 'item_new': 'item'})
    df_ui = df_ui.sort_values(by=['user', 'timestamp'])
    df_file = output_path + '_ui_new.csv'
    df_ui.to_csv(df_file, index=False)
    item_newitem_file = output_path + '_item_newitem.json'
    json_str = json.dumps(item_newitem_dict)
    with open(item_newitem_file, 'w') as out:
        out.write(json_str)
    print("has fit the csv data into target core!")


def user_similarity_score(inputpath, data_name):
    df = pd.DataFrame(pd.read_csv(inputpath))
    user_vec_dict = {}
    user_vec_dict[0] = []
    user_length = df['user'].max()
    for u in range(1, user_length+1):
        df_u1 = df[df['user'] == u]
        temp_list = df_u1['item'].tolist()
        user_vec_dict[u] = temp_list
    sim_score_mat = np.zeros((user_length+1, user_length+1))
    sim_score_mat[0][0] = 0.0
    for u1 in range(1, user_length+1):
        if u1 % 1000 == 0:
            print("1000 users has been processed!")
        for u2 in range(u1, user_length+1):
            if (len(user_vec_dict[u1]) < 5) | (len(user_vec_dict[u2]) < 5):
                continue
            else:
                # padding
                vec1 = user_vec_dict[u1].copy()
                vec2 = user_vec_dict[u2].copy()
                pad_length = max(len(vec1), len(vec2))
                if len(vec1) < pad_length:
                    vec1 += [0]*(pad_length - len(vec1))
                else:
                    vec2 += [0] * (pad_length - len(vec2))
                sim_score = np.corrcoef(vec1, vec2)[1, 0]
                sim_score = round(sim_score, 2)
                if sim_score >= 0.27:
                    sim_score_mat[u1][u2] = sim_score
    f = open(f"{data_name}_sim_score_table.csv", 'a', encoding='utf-8', newline='')
    writer = csv.writer(f)
    header = ('user1', 'user2', 'sim_score')
    writer.writerow(header)
    for u1 in range(1, user_length + 1):
        for u2 in range(u1, user_length + 1):
            if sim_score_mat[u1][u2] > 0.27:
                data = (u1, u2, sim_score_mat[u1][u2])
                writer.writerow(data)
    print("the score file has been writen!")


def check_member_interact(next_u_index):
    # check whether next_u has an edge with temp_group_list' members
    global temp_group_list, indice_matrix, adj_matrix, refuse_count_list, temp_u2_list, refuse_max_len
    flag1 = 0
    for member in temp_group_list:
        if indice_matrix[member - 1][next_u_index] == 0:
            refuse_count_list[member - 1] += 1
            if refuse_count_list[member - 1] > refuse_max_len:
                if member in temp_u2_list:
                    temp_u2_list.remove(member)
            flag1 = 1
            return flag1
    for member in temp_group_list:
        if adj_matrix[member - 1][next_u_index] == 1:
            flag1 = 1
            return flag1
    return flag1


def find_group_member_new(head_u, min_group_size=3, uu_sim_threshold=0.27):
    global gl_u_group_list
    global temp_u
    global end_flag
    global temp_group_list
    global indice_matrix
    global adj_matrix
    global temp_u2_list
    global refuse_count_list
    global max_group_size
    if len(temp_group_list) >= max_group_size:
        gl_u_group_list[head_u - 1].append(temp_group_list)
        temp_group_list = [head_u]
        temp_u = head_u
        max_group_size = random.randint(3, 16)
        return
    else:
        assert end_flag == 0
        res_flag = 0
        for i in temp_u2_list:
            if adj_matrix[head_u - 1][i - 1] != 1:
                res_flag = 1
                break
        if res_flag == 0:
            # head_u is an isolated node
            end_flag = 1
            return
        else:
            next_u_idx = 0
            next_flag = 0
            for i in temp_u2_list:
                if (indice_matrix[head_u - 1, i - 1].item() == 1) & (adj_matrix[head_u - 1, i - 1].item() == 0) & (
                        i > temp_u):
                    next_u_idx = i - 1
                    next_flag = 1
                    break
            if next_flag == 0:
                end_flag = 1
                if len(temp_group_list) >= min_group_size:
                    gl_u_group_list[head_u - 1].append(temp_group_list)
                    temp_group_list = []
                    return
                else:
                    temp_group_list = []
                    return
            else:
                res_flag2 = check_member_interact(next_u_idx)
                if res_flag2 == 1:
                    temp_u = next_u_idx + 1
                    return
                else:
                    # haven't been interacted
                    temp_group_list.append(next_u_idx + 1)
                    refuse_count_list[next_u_idx] = 0
                    for i in temp_group_list:
                        adj_matrix[i - 1][next_u_idx] = 1
                    temp_u = next_u_idx + 1
                    return


def generate_group_data(inputpath, outputpath, data_name, min_group_size=3, uu_sim_threshold=0.27):
    global temp_u, end_flag, temp_group_list, indice_matrix, adj_matrix, gl_u_group_list, temp_u2_list,\
        refuse_count_list, refuse_max_len, max_group_size
    df1 = pd.DataFrame(pd.read_csv(inputpath))
    df1.dropna(axis=0, how='any')
    user_num = df1['user1'].max()
    user1_list = df1['user1'].tolist()
    user2_list = df1['user2'].tolist()
    u_list = df1['user1'].unique().tolist()
    for i in range(user_num):
        gl_u_group_list[i] = []
    indice_matrix = np.zeros((user_num, user_num), dtype=int)
    adj_matrix = np.identity(user_num, dtype=int)
    for i in range(len(user1_list)):
        indice_matrix[user1_list[i] - 1, user2_list[i] - 1] = 1
    print("start proceeding the user list!")
    for u in u_list:
        temp_group_list = [u]
        temp_u = head_u = u
        temp_u2_list = [i + 1 for i, j in enumerate(indice_matrix[head_u - 1]) if j == 1]
        refuse_count_list = [0 for _ in range(user_num)]
        end_flag = 0
        while end_flag == 0:
            find_group_member_new(head_u)
        if u % 1000 == 0:
            print("has proceeded 1000 users' group list!")
        if u < 5:
            print(gl_u_group_list[u - 1])
    group_id = 1
    str1 = ','
    f = open(f"{outputpath}{data_name}_groupid_users.txt", 'w')
    for _, i in enumerate(gl_u_group_list.keys()):
        for j in range(len(gl_u_group_list[i])):
            f.write(str(group_id))
            f.write('\t')
            f.write(str1.join(str(k) for k in gl_u_group_list[i][j]))
            f.write('\n')
            group_id += 1
    del temp_u, end_flag, temp_group_list, indice_matrix, adj_matrix
    gc.collect()
    print("have generated the group_users txt!")


def check_gi_interaction(df_ui, member_list):
    """ generate gi interactions and timestamp """
    df_group = df_ui[df_ui['user'].isin(member_list)]
    gi_temp_list = []
    gi_list = []
    gi_time_list = []
    item_count = df_group['item'].value_counts()
    year_list = [2019,2020,2021,2022]
    for i, count in item_count.items():
        if count >= 2:
            gi_temp_list.append(i)
    for i in gi_temp_list:
        df_temp = df_group[df_group['item'] == i]
        timestamp_list = sorted(df_temp['timestamp'].tolist())
        year_dict = {}
        temp_year = timestamp_list[-1].year
        for y in year_list:
            year_dict[y] = 0
        max_year_count = 1
        for t in range(len(timestamp_list)):
            year_dict[timestamp_list[t].year] += 1
            if year_dict[timestamp_list[t].year] > max_year_count:
                temp_year = timestamp_list[t].year
                max_year_count = year_dict[timestamp_list[t].year]
        gi_time_list.append(temp_year)
        gi_list.append(i)
    return gi_list, gi_time_list


def generate_group_item_file(inputpath, ui_ipath, gu_outpath, gi_outpath):
    df_ui = pd.DataFrame(pd.read_csv(ui_ipath))
    df_ui['timestamp'] = pd.to_datetime(df_ui['timestamp'])
    df_ui.drop_duplicates(subset=['user', 'item'], keep='last', inplace=True)
    process_count = 0
    group_idx = 1
    group_list = []
    user_list = []
    group_i_list = []
    item_list = []
    time_list = []
    with open(inputpath, 'r') as f:
        for line in f:
            gid, member_list = line.rstrip().split('\t')
            member_list = member_list.split(',')
            member_list = [int(i) for i in member_list]
            gi_list, gi_time_list = check_gi_interaction(df_ui, member_list)
            if len(gi_list) >= 3:
                for u in member_list:
                    group_list.append(group_idx)
                    user_list.append(u)
                for i in range(len(gi_list)):
                    group_i_list.append(group_idx)
                    item_list.append(gi_list[i])
                    time_list.append(gi_time_list[i])
                group_idx += 1
            process_count += 1
            if process_count % 10000 == 0:
                print("has processed 10000 group")
    pd.DataFrame({'group': group_list, 'user': user_list}).to_csv(f"{gu_outpath}_gu.csv", index=False)
    pd.DataFrame({'group': group_i_list, 'item': item_list, 'timestamp': time_list}).to_csv(f"{gi_outpath}_gi.csv", index=False)
    print("has generated the gu and gi csv file")



def check_attr_len(data_type = 'Amazon'):
    print('Begin extracting meta infos...')
    if data_type == 'Amazon':
        d_path = './Toys_and_Games/meta_Toys_and_Games'
        df_ui = pd.DataFrame(pd.read_csv('./Toys_and_Games_ui.csv'))
        meta_infos = Amazon_meta(d_path, df_ui)
        attribute_core = 0
        item2idpath = './Toys_and_Games_item2ids_new.json'
        item2id = json.loads(open(item2idpath).readline())
        attribute_num, avg_attribute, item2attributes = get_attribute_Amazon(meta_infos, item2id, attribute_core)
        json_str = json.dumps(item2attributes)
        with open('./Toys_and_Games_item2attr_new.json', 'w') as out:
            out.write(json_str)
        print('attribute num', attribute_num)
        print('avag_attribute', avg_attribute)
        print("has check the attr")


amazon_datas2 = './temp_data/yelp/yelp'
inputpath = amazon_datas2 + '_groupid_users.txt'
ui_ipath = amazon_datas2 + '_ui.csv'
gu_outpath = amazon_datas2
gi_outpath = amazon_datas2
generate_group_item_file(inputpath, ui_ipath, gu_outpath, gi_outpath)
