# -*- coding: utf-8 -*-
# @Time: 2020/4/17 22:23
# @Author: Rollbear
# @Filename: text_clustering.py
# 文本聚类

import pandas as pd  # 矩阵处理工具
from gensim.models import Word2Vec  # Word2Vec词嵌入模型
from sklearn.cluster import KMeans  # k-means模型

from entity.comm import Comm
from util.txt_read import load_word_list
from util.vec import get_doc_vec
from util.xl_read import read_xl_by_line


def main():
    comments_with_likes = read_xl_by_line("../resources/xls/e3.xlsx")  # 加载附件三
    stop_words = load_word_list("../resources/special-words/stop_words.txt")
    comm_dict_3 = Comm.generate_comm_dict(comments_with_likes, True, stop_words)  # 分词、去停用词、并生成字典

    line_sents = [comm_dict_3[row[0]].seg_topic + comm_dict_3[row[0]].seg_detail for row in comments_with_likes]
    sents = list(map(lambda x: " ".join(x), line_sents))

    wv_model = Word2Vec(line_sents, size=400, window=5, sg=1, min_count=5)

    # 计算基于Word2Vec模型的文档向量
    doc_vec = [get_doc_vec(
        comm_dict_3[row[0]].seg_topic + comm_dict_3[row[0]].seg_detail,
        model=wv_model) for row in comments_with_likes]

    # 使用pandas进行数据标准化（z-score）
    data = pd.DataFrame(doc_vec)
    data_zs = (data - data.mean()) / data.std()

    k = 10  # 类数
    iteration = 500  # 最大迭代次数
    km_model = KMeans(n_clusters=k, n_jobs=4, max_iter=iteration)

    # 训练/拟合模型
    km_model.fit(data_zs)

    # 输出每个类别样本个数
    # print(pd.Series(km_model.labels_).value_counts())

    labels = km_model.labels_  # 获取样本的聚类标签

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
            print(f"Text {count}: "+ text)
        print()


if __name__ == '__main__':
    main()
