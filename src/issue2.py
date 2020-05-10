# -*- coding: utf-8 -*-
# @Time: 2020/5/5 15:51
# @Author: Rollbear
# @Filename: issue2.py
# 利用LDA主题模型从聚类完毕的簇中抽取主题（若干关键词序列）

import time

import gensim
import pandas as pd  # 矩阵处理工具
from gensim.models import Word2Vec  # Word2Vec词嵌入模型
from sklearn.cluster import MeanShift

from util.dataset import fetch_data, fetch_default_stop_words
from util.evaluation import HotspotEvaluation
from util.path import *
from util.topic_model import draw_cluster_key_word
from util.vec import doc_vec
from util.xl import read_xl_by_line, write_rows


def main():
    # 加载附件三
    comments_with_likes = read_xl_by_line(sheet_3_input)
    # 加载停用词
    stop_words = fetch_default_stop_words()
    # 分词、去停用词、并生成字典
    comm_dict_3 = fetch_data("full_dataset_sheet_3", stop_words=stop_words,
                             cut_all=True, remove_duplicates=False)
    print("data loading completed." + str(time.asctime(time.localtime(time.time()))))

    # 加载训练好的wv模型
    wv_model = gensim.models.KeyedVectors.load_word2vec_format(
        word2vec_model_path, binary=False)
    print("wv model loading completed." + str(time.asctime(time.localtime(time.time()))))

    # 计算基于Word2Vec模型的文档向量，并进行标准化
    docs_vec = [doc_vec(
        comm_dict_3[row[0]].seg_topic + comm_dict_3[row[0]].seg_detail,
        model=wv_model) for row in comments_with_likes]
    print("doc vec processing completed.", str(time.asctime(time.localtime(time.time()))))
    data = pd.DataFrame(docs_vec)
    data_zs = (data - data.mean()) / data.std()

    # 训练均值漂移模型
    ms_model = MeanShift(bandwidth=4)
    ms_model.fit(data_zs)  # 拟合模型

    labels = ms_model.labels_
    print("mean-shift model fit/load completed.", str(time.asctime(time.localtime(time.time()))))

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
    sorted_clusters = sorted(more_valuable_clusters, key=lambda c: HotspotEvaluation(c).score, reverse=True)
    print("ranking completed." + str(time.asctime(time.localtime(time.time()))))

    # 对热度前五的簇进行LDA主题建模，抽取关键词
    top_5 = sorted_clusters[:5]
    for cluster in top_5:
        print(HotspotEvaluation(cluster).score)
        print(draw_cluster_key_word(cluster))
        print("----------------------")

    # 生成热点问题表
    cluster_sheet_title = ("热度排名", "问题ID", "热度指数",
                           "时间范围", "地点/人群", "问题描述")
    cluster_sheet_rows = [(index+1, index+1,
                           HotspotEvaluation(cluster).score,
                          HotspotEvaluation(cluster).date_range_str,
                           None,
                          "，".join(draw_cluster_key_word(cluster)))
                          for index, cluster in enumerate(top_5)]
    # 写入Excel表格
    write_rows(path=cluster_sheet_path, rows=cluster_sheet_rows, title=cluster_sheet_title)

    # 生成热点问题留言明细表
    detail_sheet_title = ("问题ID", "留言编号", "留言用户",
                          "留言主题", "留言时间", "留言详情", "点赞数", "反对数")
    detail_sheet_rows = [(index+1, *row[:5], row[6], row[5])  # 最后两列的顺序与附件四相反
                         for index, cluster in enumerate(top_5) for row in cluster]
    # 写入Excel表格
    write_rows(path=detail_sheet_path, rows=detail_sheet_rows, title=detail_sheet_title)


if __name__ == '__main__':
    main()
