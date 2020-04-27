# -*- coding: utf-8 -*-
# @Time: 2020/4/21 22:15
# @Author: Rollbear
# @Filename: dataset.py

from entity.comm import Comm
from util.timer import timer
from util.txt_read import load_word_list
from util.xl_read import read_xl_by_line


class UnknownDataset(Exception):
    def __str__(self):
        return "未知的数据集名"


def fetch_default_stop_words():
    """加载默认停用词库"""
    return load_word_list("../resources/special-words/stop_words.txt")


@timer
def fetch_data(ds_name: str, cut_all=True, mode='dict', stop_words=None, remove_duplicates=True):
    """
    加载数据集
    :param ds_name: 数据集名
    :param cut_all: 是否启用全模式
    :param mode: 字典或line sentences
    :param stop_words: 停用词表
    :param remove_duplicates: 是否针对“留言详情”去重
    :return: list or dict
    """
    stop_words = [] if stop_words is None else stop_words
    comments = []
    comm_dict = {}

    if ds_name.startswith("example"):

        if ds_name == "example_3":
            comments = read_xl_by_line("../resources/xls/e3.xlsx")  # 加载附件三
        elif ds_name == "example_2":
            comments = read_xl_by_line("../resources/xls/e2.xlsx")
        elif ds_name == "example_4":
            comments = read_xl_by_line("../resources/xls/e4.xlsx")
        elif ds_name == "example_all":
            # 三个附件语料的集合
            return fetch_data("example_2", cut_all, mode, stop_words) \
                   + fetch_data("example_3", cut_all, mode, stop_words) \
                   + fetch_data("example_4", cut_all, mode, stop_words)
        comm_dict = Comm.generate_comm_dict(comments, cut_all=cut_all, stop_words_lt=stop_words)

    if ds_name.startswith("full_dataset"):
        if ds_name == "full_dataset_sheet_2":
            comments = read_xl_by_line("../resources/full_dataset/full_dataset_sheet_2.xlsx")
        elif ds_name == "full_dataset_sheet_3":
            comments = read_xl_by_line("../resources/full_dataset/full_dataset_sheet_3.xlsx")
        elif ds_name == "full_dataset_sheet_4":
            comments = read_xl_by_line("../resources/full_dataset/full_dataset_sheet_4.xlsx")
        else:
            raise UnknownDataset
        # 完整数据集的附件三的列排列顺序与示例数据不同，因此附加full_dataset参数进行适应
        comm_dict = Comm.generate_comm_dict(comments,
                                            cut_all=cut_all,
                                            stop_words_lt=stop_words,
                                            full_dataset=True)
    if remove_duplicates is True:
        # 针对“留言详情”去重（利用集合的特性实现去重）
        detail_key_dict = {elem.detail: elem for key, elem in comm_dict.items()}
        comm_dict = {elem.comm_id: elem for key, elem in detail_key_dict.items()}

    if mode == 'dict':
        return comm_dict
    if mode == 'lines':
        return [comm_dict[row[0]].seg_topic + comm_dict[row[0]].seg_detail for row in comments]
