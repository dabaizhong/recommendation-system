import faiss
from collections import defaultdict
from tqdm import tqdm
import pickle
import torch
import pandas as pd
import numpy as np

save_path = ''

item_emb = torch.load(save_path + "item_embedding.pth")
item_df = pd.DataFrame(item_emb.numpy())
item_df['item_id'] = range(len(item_df))  # 假设物品ID是0,1,2,...
# 先重命名维度列
item_df.columns = [f'dim_{i}' for i in range(item_emb.shape[1])] + ['item_id']
cols = ['item_id'] + [f'dim_{i}' for i in range(item_emb.shape[1])]
item_df = item_df[cols]

def item_emb_sim(item_emb_df, save_path, topk):
    item_idx_2_rawid_dict = dict(zip(item_emb_df.index, item_emb_df['item_id']))
    item_cols = [x for x in item_emb_df.columns if 'dim' in x]
    # 拿到全部的embedding值
    item_emb_np = item_emb_df[item_cols].values.astype(np.float32)
    # 连续存储
    item_emb_np = np.ascontiguousarray(item_emb_np, dtype=np.float32)
    # 计算物品嵌入矩阵中每一行向量（即每个物品的向量）的 L2 范数（欧几里得长度）。
    norms = np.linalg.norm(item_emb_np, axis=1, keepdims=True)
    norms[norms==0] = 1e-10
    item_emb_np = item_emb_np / norms
    # 建立索引
    item_index = faiss.IndexFlatIP(item_emb_np.shape[1])
    # 加入嵌入向量
    item_index.add(item_emb_np)
    # 拿到搜搜索结果 结构是(sim, idx)
    sim, idx = item_index.search(item_index, topk)
    item_sim_dict = defaultdict(dict)
    for target_idx, sim_items, sim_values in tqdm(zip(range(len(item_emb_np)), sim, idx)):
        # 拿到目标索引 用目标索引拿到原始的目标id
        target_raw_id = item_idx_2_rawid_dict[target_idx]
        # 从一开始去掉自身
        for rele_idx, sim_value in zip(sim_items[1:],  sim_values[1:]): 
            # 拿到相似的id
            rele_raw_id = item_idx_2_rawid_dict[rele_idx]
            # 保存在字典当中
            item_sim_dict[target_raw_id][rele_raw_id] = item_sim_dict.get(target_raw_id, {}).get(rele_raw_id, 0) + sim_value
            
    return item_sim_dict


