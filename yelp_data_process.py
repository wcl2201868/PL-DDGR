import pandas as pd
import os
import json
import numpy as np
from collections import defaultdict
import datetime
def load_yelp_reviews(dataset_name, rating_score):
    datas = []
    data_flie = open(dataset_name, 'r', encoding='utf-8')
    for line in data_flie:
        content = json.loads(line)
        if float(content['stars']) <= rating_score:
            continue
        user = content['user_id']
        item = content['business_id']
        time = content['date']
        datas.append((user, item, time))
    return datas

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
        item_time.sort(key=lambda x: x[1])
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

def Yelp_meta(dataset_name, df_ui):
    datas = {}
    meta_file = open(dataset_name, 'r', encoding='utf-8')
    item_ids = df_ui['item_id'].unique().tolist()
    for i in item_ids:
        datas[i] = {}
    for info in meta_file:
        content = json.loads(info)
        if content['business_id'] not in item_ids:
            continue
        datas[content['business_id']] = content
    return datas

def add_comma(num):
    str_num = str(num)
    res_num = ''
    for i in range(len(str_num)):
        res_num += str_num[i]
        if (len(str_num)-i-1) % 3 == 0:
            res_num += ','
    return res_num[:-1]

def get_attribute_Yelp(meta_infos, item2id, attribute_core):
    attributes = defaultdict(int)
    for iid, info in meta_infos.items():
        if info:
            categories = list(info['categories'].replace(' ', '').split(','))
            for cate in categories:
                attributes[cate] += 1
            try:
                attributes[info['city']] += 1
            except:
                pass
    print(f'before delete, attribute num:{len(attributes)}')
    new_meta = {}
    for iid, info in meta_infos.items():
        new_meta[iid] = []
        if info:
            try:
                if attributes[info['city']] >= attribute_core:
                    new_meta[iid].append(info['city'])
            except:
                pass
            categories = list(info['categories'].replace(' ', '').split(','))
            for cate in categories:
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

def main(dataset_name):
    data_path = 'F:\dataset'
    np.random.seed(12345)
    user_core = 7
    item_core = 7
    attribute_core = 0
    rating_score = 4.0
    review_path = os.path.join(data_path, dataset_name, 'yelp_academic_dataset_review.json')
    datas = load_yelp_reviews(review_path, rating_score)
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
    df_ui = pd.DataFrame({'user_id': users_list, 'item_id': items_list, 'timestamp': times_list})
    df_ui = df_ui.drop_duplicates()
    df_ui['timestamp'] = pd.to_datetime(df_ui['timestamp'])
    timestamp_cutline = datetime.datetime.strptime('2019-01-01 00:00', "%Y-%m-%d %H:%M")
    df_ui = df_ui[df_ui['timestamp'] > timestamp_cutline]
    df_u_group = df_ui.groupby(['user_id'])
    user_items = {}
    for u, u_group in df_u_group:
        user_items[u] = u_group['item_id'].tolist()
    print(f'{dataset_name} Raw data has been processed! Lower than {rating_score} are deleted!')
    user_items = filter_Kcore(user_items, user_core=user_core, item_core=item_core)
    print(f'User {user_core}-core complete! Item {item_core}-core complete!')
    users_list = []
    items_list = []
    for u, items in user_items.items():
        for item in items:
            users_list.append(u)
            items_list.append(item)
    df_ui_remain = pd.DataFrame({'user_id': users_list, 'item_id': items_list})
    df_ui = df_ui.merge(df_ui_remain, on=['user_id', 'item_id'])
    df_ui = df_ui.drop_duplicates()
    users_count = df_ui['user_id'].value_counts()
    items_count = df_ui['item_id'].value_counts()
    print("reorder the index!")
    itemid = np.arange(1, items_count.shape[0] + 1)
    user_idx = pd.DataFrame({'user_id': users_count.index, 'user': np.arange(1, users_count.shape[0] + 1)})
    item_idx = pd.DataFrame({'item_id': items_count.index, 'item': itemid})
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
    output_log = open(dataset_name + '_output.log', 'w')
    show_info = f'Total User: {user_num}, Avg User: {user_avg:.4f}, Min Len: {user_min}, Max Len: {user_max}\n' + \
                f'Total Item: {item_num}, Avg Item: {item_avg:.4f}, Min Inter: {item_min}, Max Inter: {item_max}\n' + \
                f'Iteraction Num: {interact_num}, Sparsity: {sparsity:.2f}%'
    print(show_info)
    output_log.write(show_info)
    output_log.write('\nBegin extracting meta infos...\n')
    d_path = os.path.join(data_path, dataset_name, 'yelp_academic_dataset_business.json')
    meta_infos = Yelp_meta(d_path, df_ui)
    attribute_num, avg_attribute, item2attributes = get_attribute_Yelp(meta_infos, item2id, attribute_core)
    show_info = f'{dataset_name} & {add_comma(user_num)}& {add_comma(item_num)} & {user_avg:.1f}' + \
                f'& {item_avg:.1f}& {add_comma(interact_num)}& {sparsity:.2f}\%&{add_comma(attribute_num)}&' + \
                f'{avg_attribute:.1f}'
    output_log.write(show_info)
    print(show_info)
    data_file = dataset_name + '_ui.csv'
    item2attributes_file = dataset_name + '_item2attributes.json'
    item2id_file = dataset_name + '_item2ids.json'
    df_ui.to_csv(data_file, index=False)
    json_str = json.dumps(item2attributes)
    with open(item2attributes_file, 'w') as out:
        out.write(json_str)
    json_str = json.dumps(item2id)
    with open(item2id_file, 'w') as out:
        out.write(json_str)
main('yelp')