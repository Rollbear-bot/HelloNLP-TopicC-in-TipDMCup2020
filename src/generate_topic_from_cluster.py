# -*- coding: utf-8 -*-
# @Time: 2020/5/5 15:51
# @Author: Rollbear
# @Filename: generate_topic_from_cluster.py
# 利用LDA主题模型从聚类完毕的簇中抽取主题（若干关键词序列）

import time

import gensim
import pandas as pd  # 矩阵处理工具
from gensim.models import Word2Vec  # Word2Vec词嵌入模型
from sklearn.cluster import MeanShift

from util.dataset import fetch_data, fetch_default_stop_words
from util.evaluation import HotspotEvaluation
from util.topic_model import draw_cluster_key_word
from util.vec import doc_vec
from util.xl_read import read_xl_by_line

mean_shift_model_path = ""  # ms模型的保存路径


def main():
    # 加载附件
    comments_with_likes = read_xl_by_line("../resources/full_dataset/full_dataset_sheet_3.xlsx")

    stop_words = fetch_default_stop_words()
    # 分词、去停用词、并生成字典
    comm_dict_3 = fetch_data("full_dataset_sheet_3", stop_words=stop_words,
                             cut_all=True, remove_duplicates=False)
    print("data loading completed." + str(time.asctime(time.localtime(time.time()))))

    # 加载训练好的wv模型
    wv_model = gensim.models.KeyedVectors.load_word2vec_format(
        "../resources/wv_model_full_dataset_0425", binary=False)
    print("wv model loading completed." + str(time.asctime(time.localtime(time.time()))))

    # 计算基于Word2Vec模型的文档向量，并进行标准化
    docs_vec = [doc_vec(
        comm_dict_3[row[0]].seg_topic + comm_dict_3[row[0]].seg_detail,
        model=wv_model) for row in comments_with_likes]
    print("doc vec processing completed.", str(time.asctime(time.localtime(time.time()))))
    data = pd.DataFrame(docs_vec)
    data_zs = (data - data.mean()) / data.std()

    # 训练均值漂移模型
    ms_model = MeanShift(bandwidth=11)
    ms_model.fit(data_zs)  # 拟合模型

    # 加载训练好的Mean-Shift模型
    # ms_model = joblib.load(mean_shift_model_path)
    labels = ms_model.labels_
    print("model fit/load completed.", str(time.asctime(time.localtime(time.time()))))

    # 按标签分类存放在字典中
    comm_cluster = {}
    for index, row in enumerate(comments_with_likes):
        if labels[index] in list(comm_cluster.keys()):
            comm_cluster[labels[index]].append(row)
        else:
            comm_cluster[labels[index]] = [row]
    print("label picking completed." + str(time.asctime(time.localtime(time.time()))))

    # 仅考虑留言数量大于等于3的簇，剩下的视为“离群点”
    more_valuable_clusters = [cluster for cluster in comm_cluster.values() if len(cluster) >= 3]
    # 根据热度排序，选出热度前五的簇
    sorted_clusters = sorted(more_valuable_clusters, key=lambda c: HotspotEvaluation(c).score)
    print("ranking completed." + str(time.asctime(time.localtime(time.time()))))

    for cluster in sorted_clusters:
        print(cluster)

    # 对热度前五的簇进行LDA主题建模，抽取关键词
    top_5 = sorted_clusters[:5]
    for cluster in top_5:
        print(draw_cluster_key_word(cluster))


if __name__ == '__main__':
    main()
