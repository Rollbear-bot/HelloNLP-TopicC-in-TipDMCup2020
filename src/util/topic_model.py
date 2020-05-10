# -*- coding: utf-8 -*-
# @Time: 2020/5/5 17:19
# @Author: Rollbear
# @Filename: topic_model.py
# 主题建模模块

import jieba
from gensim import corpora  # 语料/词频工具
from gensim.models.ldamodel import LdaModel  # LDA模型

from util.dataset import fetch_default_stop_words


def draw_cluster_key_word(cluster: list):
    """
    抽取一个聚类的关键词
    :param cluster: list of tuple(7)，问题二中聚类得到的簇
    :return: list of words，关键词列表
    """
    stop = fetch_default_stop_words()  # 停用词表
    stop.extend(["", " ", "\n", "\t", "*"])  # 附加几个停用词

    sents = [jieba.lcut(row[2] + "。" + row[4], cut_all=True) for row in cluster]  # 分词
    sents = [[word for word in sent if word not in stop] for sent in sents]  # 去停用词

    dictionary = corpora.Dictionary(sents)  # 建立词典
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in sents]  # 文档-词频矩阵

    # 训练LDA模型
    lda_model = LdaModel(doc_term_matrix, num_topics=1, id2word=dictionary, passes=1)

    # 解析出主题中概率最大的前6个词
    key_words = [word for index, word in
                 enumerate(lda_model.show_topics()[0][1].split("\""))
                 if index in [1, 3, 5, 7, 9, 11]]
    return key_words
