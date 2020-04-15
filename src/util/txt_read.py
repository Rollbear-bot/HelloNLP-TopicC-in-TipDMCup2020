# -*- coding: utf-8 -*-
# @Time: 2020/4/14 20:47
# @Author: Rollbear
# @Filename: txt_read.py


def load_word_list(filepath):
    """读取文本文档到列表"""
    stopwords = \
        [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords
