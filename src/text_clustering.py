# -*- coding: utf-8 -*-
# @Time: 2020/4/17 22:23
# @Author: Rollbear
# @Filename: text_clustering.py
# 文本聚类

import pandas as pd  # 矩阵处理工具
from gensim.models import Word2Vec  # Word2Vec词嵌入模型
from sklearn.cluster import MeanShift

from util.dataset import fetch_data, fetch_default_stop_words
from util.vec import doc_vec
from util.xl_read import read_xl_by_line


def main():
    comments_with_likes = read_xl_by_line("../resources/xls/e3.xlsx")  # 加载附件三
    stop_words = fetch_default_stop_words()
    # 分词、去停用词、并生成字典
    comm_dict_3 = fetch_data("example_3", stop_words=stop_words, cut_all=True)

    # 方案一：从附件三建立Word2Vec模型
    line_sents = fetch_data("example_3", stop_words=stop_words, mode="lines")
    wv_model = Word2Vec(line_sents,
                        size=400, window=5, sg=1, min_count=5)

    # 方案二：从三个附件的语料建立Word2Vec模型
    # wv_model = Word2Vec(LineSentence("../resources/line_sents.txt"),
    #   size=400, window=5, sg=1, min_count=5)

    # 方案三：加载之前在云端训练的模型
    # todo::之前训练的模型好像有点问题
    # wv_model = gensim.models.KeyedVectors.load_word2vec_format(
    #     "../resources/word2vec_build_on_all_text", binary=False)

    # 计算基于Word2Vec模型的文档向量
    docs_vec = [doc_vec(
        comm_dict_3[row[0]].seg_topic + comm_dict_3[row[0]].seg_detail,
        model=wv_model) for row in comments_with_likes]

    # 使用pandas进行数据标准化（z-score）
    data = pd.DataFrame(docs_vec)
    data_zs = (data - data.mean()) / data.std()

    # k = 5  # 类数
    # iteration = 500  # 最大迭代次数
    # km_model = KMeans(n_clusters=k, n_jobs=4, max_iter=iteration)
    #
    # # 训练/拟合模型
    # km_model.fit(data_zs)
    #
    # # 输出每个类别样本个数
    # # print(pd.Series(km_model.labels_).value_counts())
    #
    # labels = km_model.labels_  # 获取样本的聚类标签

    # ------------------------------------------------
    # 尝试均值漂移（Mean Shift）模型
    ms_model = MeanShift()
    ms_model.fit(data_zs)  # 拟合模型
    labels = ms_model.labels_
    print(labels)

    # ------------------------------------------------
    # 尝试AP聚类
    # ap_model = AffinityPropagation()
    # ap_model.fit(data_zs)
    # labels = ap_model.labels_

    # 按标签分类存放在字典中
    comm_cluster = {}
    for index, row in enumerate(comments_with_likes):
        if labels[index] in list(comm_cluster.keys()):
            comm_cluster[labels[index]].append(row[4])
        else:
            comm_cluster[labels[index]] = [row[4]]

    # 展示聚类结果
    for label, cluster in comm_cluster.items():
        print(f"Label{label}: ")
        for count, text in enumerate(cluster):
            print(f"Text {count}: " + text)
        print("-------------------------------")
        print()


if __name__ == '__main__':
    main()
