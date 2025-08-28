import numpy as np
import pandas as pd
import torch
import pickle

from tqdm import tqdm

from torch_rechub.basic.features import DenseFeature, SequenceFeature, SparseFeature
from torch_rechub.models.ranking import DIN
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator, df_to_dict, generate_seq_feature, pad_sequences

# 生成DIN模型训练所需要的数据格式
def get_data_dict(data_on_path, art_path):
    """
    这里的函数是根据torch_rechub中的示例修改而来
    Args:
        data_on_path: 这个是天池数据集将文章和点击文件merge之后的
        art_path: 这个是天池数据集文章信息的文件
    """
    # 载入联合之后的数据
    data = pd.read_csv(data_on_path)
    print('========== Read Data ==========')
    # 生成序列化训练数据
    train, val, test = generate_seq_feature(
        data=data, 
        user_col='user_id', 
        item_col='click_article_id', 
        time_col='click_timestamp', 
        item_attribute_cols=["category_id"],
        max_len=20
    )
    art = pd.read_csv(art_path)
    print('INFO: Now, the dataframe named: ', train.columns)
    # 拿到vocab_size
    n_users, n_items, n_cates = data["user_id"].max(), art["article_id"].max(), art["category_id"].max()
    # art的作用就是为了拿到vocab_size，因为有几个文章是在merge后的文件当中没出现的，存在冷启动的问题
    del art
    print(train.head())
    # 定义特征
    features = [
        SparseFeature("target_item_id", vocab_size=n_items + 1, embed_dim=8),
        SparseFeature("target_category_id", vocab_size=n_cates + 1, embed_dim=8),
        SparseFeature("user_id", vocab_size=n_users + 1, embed_dim=8)
    ]
    target_features = features
    history_features = [
        SequenceFeature("hist_item_id", vocab_size=n_items + 1, embed_dim=8, pooling="concat", shared_with="target_item_id"),
        SequenceFeature("hist_category_id", vocab_size=n_cates + 1, embed_dim=8, pooling="concat", shared_with="target_category_id")
    ]
    print('========== Generate input dict ==========')
    train = df_to_dict(train)
    val = df_to_dict(val)
    test = df_to_dict(test)
    train_y, val_y, test_y = train["label"], val["label"], test["label"]
    del train["label"]
    del val["label"]
    del test["label"]
    train_x, val_x, test_x = train, val, test
    return features, target_features, history_features, (train_x, train_y), (val_x, val_y), (test_x, test_y)


torch.manual_seed(2021)
data_on_path = 'data/data_on.csv'
art_path = 'data/articles.csv'
features, target_features, history_features, (train_x, train_y), (val_x, val_y), (test_x, test_y) = get_data_dict(data_on_path, art_path)
# 数据加载器
dg = DataGenerator(train_x, train_y)
train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(x_val=val_x, y_val=val_y, x_test=test_x, y_test=test_y, batch_size=1024)
# 定义model
model = DIN(
    features=features,
    history_features=history_features,
    target_features=target_features,
    mlp_params={"dims": [256, 128]}, 
    attention_mlp_params={"dims": [256, 128]}
)
# 开始训练
ctr_trainer = CTRTrainer(model, optimizer_params={"lr": 1e-3, "weight_decay": 1e-3}, n_epoch=3, earlystop_patience=4, device='cuda:0', model_path='save/')
ctr_trainer.fit(train_dataloader, val_dataloader)
auc = ctr_trainer.evaluate(ctr_trainer.model, test_dataloader)
print(f'test auc: {auc}')
# 训练完成之后的模型就可以用作排序，载入之前的召回字典
with open('save/recall_dict.pkl', 'rb') as f:
    recall_dict = pickle.load(f)

recall_df= []

for user_id, items in recall_dict.items():
    for item_id, _ in items:
        recall_df.append({'user_id': user_id, 'target_item_id': item_id})
df_recall = pd.DataFrame(recall_df)
# 这里载入文章文件是因为在训练din的时候用到了类别的历史序列，召回字典当中没有涉及，这里merge拿到文章的类别信息
art = pd.read_csv('data/articles.csv')
df_enriched = df_recall.merge(
    art[['article_id', 'category_id']].rename(columns={'article_id': 'target_item_id', 'category_id': 'target_category_id'}),
    on='target_item_id',
    how='left'
)

df_enriched.to_csv('save/enrich.csv')
# all_data就是全量的数据
data_on = pd.read_csv('data/data_on.csv')
# 按照时间戳进行排序
data_on.sort_values('click_timestamp', inplace=True)
# 生成序列数据
data = []
max_len = 20
for uid, hist in tqdm(data_on.groupby('user_id')):
    hist_list = hist['click_article_id'].tolist()
    len_hist_list = len(hist_list)
    len_hist_list = min(max_len, len_hist_list)
    hist_item = hist_list[:len_hist_list]
    hist_item = hist_item + [0] * (max_len - len_hist_list)

    hist_list_att = hist['category_id'].tolist()
    len_hist_list_att = len(hist_list_att)
    len_hist_list_att = min(max_len, len_hist_list_att)
    hist_item_att = hist_list_att[:len_hist_list_att]
    hist_item_att = hist_item_att + [0] * (max_len - len_hist_list_att)
    seq = [uid, hist_item, hist_item_att]
    data.append(seq)
col_name = ['user_id', 'hist_item_id', 'hist_category_id']
data = pd.DataFrame(data, columns=col_name)

enrich_df = pd.read_csv('save/enrich.csv')
# 将序列数据和召回的csv里面的target结合起来
data = data.merge(enrich_df, how='left', on='user_id').reset_index(drop=True)
# 按照训练时的数据格式
desired_columns = ['label', 'target_item_id', 'user_id', 'hist_item_id', 'hist_category_id', 'target_category_id']
data['label'] = 0
data = data.reindex(columns=desired_columns)

data_y = data['label']
del data['label']
data = df_to_dict(data)

# 构建数据加载器
dg = DataGenerator(data, data_y)
predict_dataloader, _, _ = dg.generate_dataloader(batch_size=1024)
# 预测
res = ctr_trainer.predict(model, predict_dataloader)
enrich_df['din_score'] = res
enrich_df.to_csv('save/din_rank.csv')