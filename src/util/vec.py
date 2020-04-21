# -*- coding: utf-8 -*-
# @Time: 2020/4/17 23:01
# @Author: Rollbear
# @Filename: vec.py
# 和向量有关的工具


# 计算文档向量的方法
def get_doc_vec(doc: list, model):
    """计算文档向量"""
    ignore = ["\t", " ", "\n"]
    words = [word for word in doc if word not in ignore]
    # 所有词向量求和并除以词数量
    words_num = len(words)
    # print(words_num)
    vec_sum = 0
    for word in words:
        try:
            vec_sum += model.wv[word]
        except KeyError:
            words_num -= 1
            continue
    # print(words_num)
    return vec_sum / words_num