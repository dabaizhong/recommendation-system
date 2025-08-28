from collections import defaultdict
from tqdm import tqdm
import numpy as np
import math

def i2i_sim(df, item_create_time_dict, user_click_time_dict):
    """
    这里的物品相似考虑了时间因素 包括用户的交互时间和 物品的创建时间
    Args:
        df: pd表格数据
        item_create_time_dict: 物品创建时间字典 格式为{item_id: time} 通过zip(df[item], df[time])得到
        user_click_time_dict: 用户对于物品交互时间的字典 结构是{user_id: [item_id, time]}
    """
    # 创建相似度的字典
    i2i_dict = {}
    item_count = defaultdict(int)
    # 遍历点击时间的字典
    for user, item_time_list in tqdm(user_click_time_dict.items()):
        # 遍历物品-时间列表
        for loci, item_i, click_time_i in enumerate(item_time_list):
            # 对枚举到的物品计数＋1
            item_count[item_i] += 1
            i2i_dict.setdefault(item_i, {})
            # 继续枚举同一个列表
            for locj, item_j, click_time_j in enumerate(item_time_list):
                # 相同的物品直接跳过
                if item_i == item_j:
                    continue
                # 位置相关参数
                loc_alpha = 1.0 if locj > loci else 0.7
                # 位置权重
                loc_weight = loc_alpha * (0.9 ** (np.abs(locj - loci) - 1))
                # 交互时间权重
                click_time_weight = np.exp(0.7 ** np.abs(click_time_i - click_time_j))
                # 创建时间权重
                created_time_weight = np.exp(0.8 ** np.abs(item_create_time_dict[item_i] - item_create_time_dict[item_j]))
                i2i_dict[item_i].setdefault(item_j, 0)
                # 计算最后的wij
                i2i_dict[item_i][item_j] += loc_weight * click_time_weight * created_time_weight / math.log(len(item_time_list) + 1)
    i2i_sim_ = i2i_sim.copy()
    for i, related_items in i2i_sim.items():
        for j, wij in related_items.items():
            i2i_sim_[i][j] = wij / math.sqrt(item_count[i] * item_count[j])
    return i2i_sim_

# 不考虑时间位置权重
def itemcf_sim(user_item_time_dict):
    """
    user_item_time_dict 结构是：
    {
        user1: [(itemA, timeA), (itemB, timeB), ...],
        user2: [(itemC, timeC), (itemD, timeD), ...],
        ...
    }
        文章与文章之间的相似性矩阵计算
        :param df: 数据表
        :item_created_time_dict:  文章创建时间的字典
        return : 文章与文章的相似性矩阵
        思路: 基于物品的协同过滤(详细请参考上一期推荐系统基础的组队学习)， 在多路召回部分会加上关联规则的召回策略
    """
    # i2i_sim[i][j]：表示文章 i 与文章 j 的相似度权重
    i2i_sim = {}
    # 文章 i 被多少用户点击过（点击次数）
    item_cnt = defaultdict(int)
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        for i, i_click_time in item_time_list:
            item_cnt[i] += 1
            # 如果 i2i_sim 这个字典里 还没有键 i，就创建一个键 i，并赋值为空字典 {}。
            i2i_sim.setdefault(i, {})
            for j, j_click_time in item_time_list:
                if i == j:
                    continue
                # 如果 i2i_sim[i] 里 没有键 j，就给它赋值 0。
                i2i_sim[i].setdefault(j, 0)
                # 对sim[i][j] += 并除一个惩罚因子，防止点击次数过多的用户对相似度的贡献过大
                i2i_sim[i][j] += 1 / math.log(1 + len(item_time_list))

    i2i_sim_ = i2i_sim.copy()
    for i, related_items in i2i_sim.items():
        # 物品 i 与 j 的共现次数
        for j, wij in related_items.items():
            # 相似度归一化（余弦相似度思想）避免热门物品主导相似度。
            i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])
    
    return i2i_sim_



