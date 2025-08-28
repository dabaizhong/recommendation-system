import numpy as np
import pandas as pd
import torch

from torch_rechub.basic.features import SparseFeature, DenseFeature, SequenceFeature
from torch_rechub.models.matching import YoutubeDNN
from torch_rechub.trainers import MatchTrainer
from torch_rechub.utils.data import MatchDataGenerator, df_to_dict
from torch_rechub.utils.match import gen_model_input, generate_seq_feature_match

from sklearn.preprocessing import LabelEncoder


def get_train_data(data_path):
    """
    Args:
        data_path: 文件加载路径
    这个函数是从df文件的背景下 构造YoutubeDNN所需要的数据格式 现在函数内的结构是天池新闻数据集
    其他数据集须按需要进行部分修改
    """
    data = pd.read_csv('save/data_on.csv')
    # 定义哪些列需要被用作稀疏特征， 后续会进行embedding
    sparse_fea = [
        'user_id', 
        'click_environment', 
        'click_deviceGroup', 
        'click_os', 
        'click_country', 
        'click_region',
        'click_referrer_type', 
        'category_id'
    ]
    # 这个是因为文章中的类别特征在另一个文件，需要拿过来
    art = pd.read_csv(data_path + 'articles.csv').rename(columns={'article_id': 'click_article_id'})
    # 编码器
    lbe = LabelEncoder()
    # 用户列和物品列的定义
    user_col, item_col = "user_id", "click_article_id"
    # 这里定义一个字典，因为后面不同的特征有不同的vocab_size 方便后面的使用
    feature_max_idx = {}
    # 对文章进行转换 +1是为了转换成 以1开始的，因为后续会有0进行填充
    art['click_article_id'] = lbe.fit_transform(art['click_article_id']) + 1
    # 这里是拿到vocab_size
    feature_max_idx['click_article_id'] = art['click_article_id'].max() + 1
    # 定义一个map 通过lbe编码器拿到 编码后的id 与 原始id的对应关系
    item_map = {encode_id + 1: raw_id for encode_id, raw_id in enumerate(lbe.classes_)}
    for fea in sparse_fea:
        data[fea] = lbe.fit_transform(data[fea]) + 1
        feature_max_idx[fea] = data[fea].max() + 1
        if fea == user_col:
            # 这里是拿到了编码后用户id 与 原始用户id的map
            user_map = {encode_id + 1: raw_id for encode_id, raw_id in enumerate(lbe.classes_)}
    np.save('save/raw_id_maps.npy', np.array((user_map, item_map), dtype=object))
    # 拿到唯一用户的画像
    user_profile = data[[
        'user_id', 
        'click_environment', 
        'click_deviceGroup', 
        'click_os', 
        'click_country', 
        'click_region', 
        'click_referrer_type' 
    ]].drop_duplicates('user_id')
    # 打印用户画像信息
    print("=" * 60)
    print("用户画像 (User Profile)")
    print("=" * 60)
    print(f"用户数量: {len(user_profile)}")
    # 拿到物品画像
    item_profile = art[['click_article_id', 'category_id']].drop_duplicates('click_article_id')
    # 联合起来 拿到完整的数据
    data = pd.merge(data, art, how='left', on='click_article_id')
    # 生成序列数据 这里生成负采样数据的方式是1 数据是采用list-wise
    df_train, df_test = generate_seq_feature_match(data, user_col, item_col, time_col="click_timestamp", item_attribute_cols=[], sample_method=1, mode=2, neg_ratio=3, min_item=0)
    # 获得模型输入的格式
    x_train = gen_model_input(df_train, user_profile, user_col, item_profile, item_col, seq_max_len=20)
    # label=0 means the first pred value is positive sample
    y_train = np.array([0] * df_train.shape[0])
    x_test = gen_model_input(df_test, user_profile, user_col, item_profile, item_col, seq_max_len=20)
    np.save('save/data_cache.npy', np.array((x_train, y_train, x_test), dtype=object))
    # 用户的特征列
    user_cols = [
        'user_id', 
        'click_environment',
        'click_deviceGroup',
        'click_os', 
        'click_country', 
        'click_region',
        'click_referrer_type'
    ]
    # 稀疏特征
    user_fea = [SparseFeature(name, vocab_size=feature_max_idx[name], embed_dim=16) for name in user_cols]
    # 序列特征
    user_fea += [SequenceFeature("hist_click_article_id", vocab_size=feature_max_idx["click_article_id"], embed_dim=16, pooling="mean", shared_with="click_article_id")]
    # 物品特征
    item_features = [SparseFeature('click_article_id', vocab_size=feature_max_idx['click_article_id'], embed_dim=16)]
    # 负采样物品特征
    neg_item_feature = [SequenceFeature('neg_items', vocab_size=feature_max_idx['click_article_id'], embed_dim=16, pooling="concat", shared_with="click_article_id")]
    # 转成字典
    all_item = df_to_dict(item_profile)
    test_user = x_test
    return user_fea, item_features, neg_item_feature, x_train, y_train, all_item, test_user

data_path = ''
save_path = ''

torch.manual_seed(2021)
user_fea, item_fea, neg_item_fea, x_train, y_train, all_item, test_user = get_train_data(data_path)
dg = MatchDataGenerator(x=x_train, y=y_train)
model = YoutubeDNN(user_fea, item_fea, neg_item_fea, user_params={"dims": [128, 64, 16]}, temperature=0.02)
trainer = MatchTrainer(model, mode=2, optimizer_params={"lr": 0.01, "weight_decay": 1e-6}, n_epoch=10, device='cuda:0', model_path=save_path)
train_dl, test_dl, item_dl = dg.generate_dataloader(test_user, all_item, batch_size=128)
trainer.fit(train_dl)
print("inference embedding")
user_embedding = trainer.inference_embedding(model=model, mode="user", data_loader=test_dl, model_path='save/')
item_embedding = trainer.inference_embedding(model=model, mode="item", data_loader=item_dl, model_path='save/')
print(user_embedding.shape, item_embedding.shape)
torch.save(user_embedding.data.cpu(), "user_embedding.pth")
torch.save(item_embedding.data.cpu(), "item_embedding.pth")
# match_evaluation(user_embedding, item_embedding, test_user, all_item, topk=10)