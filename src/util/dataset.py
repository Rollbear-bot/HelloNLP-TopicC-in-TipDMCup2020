# -*- coding: utf-8 -*-
# @Time: 2020/4/21 22:15
# @Author: Rollbear
# @Filename: dataset.py

from entity.comm import Comm
from util.txt_read import load_word_list
from util.xl_read import read_xl_by_line


def fetch_default_stop_words():
    """加载默认停用词库"""
    return load_word_list("../resources/special-words/stop_words.txt")


def fetch_data(ds_name: str, cut_all=True, mode='dict', stop_words=None):
    """
    加载数据集
    :param ds_name: 数据集名
    :param cut_all: 是否启用全模式
    :param mode: 字典或line sentences
    :param stop_words: 停用词表
    :return: list or dict
    """
    stop_words = [] if stop_words is None else stop_words
    comments = []

    if ds_name.startswith("example"):

        if ds_name == "example_3":
            comments = read_xl_by_line("../resources/xls/e3.xlsx")  # 加载附件三
        if ds_name == "example_2":
            comments = read_xl_by_line("../resources/xls/e2.xlsx")
        if ds_name == "example_4":
            comments = read_xl_by_line("../resources/xls/e4.xlsx")
        if ds_name == "example_all":
            # 三个附件语料的集合
            return fetch_data("example_2", cut_all, mode, stop_words) \
                   + fetch_data("example_3", cut_all, mode, stop_words) \
                   + fetch_data("example_4", cut_all, mode, stop_words)

    comm_dict = Comm.generate_comm_dict(comments, cut_all=cut_all, stop_words_lt=stop_words)

    if mode == 'dict':
        return comm_dict
    if mode == 'lines':
        return [comm_dict[row[0]].seg_topic + comm_dict[row[0]].seg_detail for row in comments]
