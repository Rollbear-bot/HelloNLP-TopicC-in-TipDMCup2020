# -*- coding: utf-8 -*-
# @Time: 2020/4/17 23:01
# @Author: Rollbear
# @Filename: vec.py
# 和向量有关的工具

import numpy


class UnexpectedList(Exception):
    """传入了无效的列表"""
    def __str__(self):
        return "传入了无效的词列表"


def doc_vec(doc: list, model):
    """
    计算文档向量（各个词向量平均）
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
    if words_num == 0:  # 空白文本或无意义文本
        return numpy.array([0] * model.vector_size)
    else:
        return vec_sum / words_num


def doc_vec_with_weight(doc: list, model, weight: dict, default_weight=1):
    """
    计算文档向量（带权重）
    :param doc: 文档（词列表）
    :param model: 词嵌入模型
    :param weight: 词权重字典
    如果指定了作用的词，则在权重表中给出指定词的权重，未指定的词权重为default_weight
    :param default_weight: 默认权重
    :return: 文档向量（一个n维向量，n是训练模型是指定的维数）
    """
    # 检查格式是否合法
    # if action_range is None:
    #     if len(doc) != len(weight):
    #         raise UnexpectedList
    # elif len(action_range) != len(weight):
    #     raise UnexpectedList

    # 所有词向量求和并除以词数量
    words_num = len(doc)
    vec_sum = 0
    for word in doc:
        try:
            # 求向量和时乘上权重
            vec_sum += model.wv[word] * weight.get(word, default_weight)
        except KeyError:
            words_num -= 1
            continue
    return vec_sum / words_num
