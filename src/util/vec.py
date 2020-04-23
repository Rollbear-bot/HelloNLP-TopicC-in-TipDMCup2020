# -*- coding: utf-8 -*-
# @Time: 2020/4/17 23:01
# @Author: Rollbear
# @Filename: vec.py
# 和向量有关的工具


def get_doc_vec(doc: list, model):
    """
    计算文档向量（各个词向量加权平均方式）
    :param doc: 词列表
    :param model: 词嵌入模型
    :return: 文档向量（一个n维向量，n是训练模型是指定的维数）
    """
    ignore = ["\t", " ", "\n"]  # 忽略词
    words = [word for word in doc if word not in ignore]

    # 所有词向量求和并除以词数量
    words_num = len(words)
    vec_sum = 0
    for word in words:
        try:
            vec_sum += model.wv[word]
        except KeyError:
            words_num -= 1
            continue
    return vec_sum / words_num
