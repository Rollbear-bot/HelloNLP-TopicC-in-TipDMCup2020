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
    1.以"example"开头的数据集为示例小数据集
    2.以"full_dataset"开头的是完整数据集
    3.其他

    :param cut_all: 是否启用全模式
    :param mode: 两种返回模式
    1.'dict': 存储Comm对象（留言对象）的字典，键为留言id，值为留言对象
    2.'lines': 已分词的词链表，list of list(words in a sentence)，原文本中的每句话为一个列表元素

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


def show_data_analysis(**kwargs):
    """
    数据集分析，输出到控制台
    :param kwargs: 与fetch_data一致的参数
    :return: None
    """
    data = fetch_data(mode='dict', remove_duplicates=False, **kwargs)  # 加载数据集
    dataset_name = kwargs['ds_name']  # 数据集名称
    raw_num_record = len(list(data.values()))  # 记录条数（去重前）

    data = fetch_data(mode='dict', remove_duplicates=True, **kwargs)  # 加载数据集（并去重）
    num_record = len(list(data.values()))  # 记录条数（去重后）
    max_comm_len = max([len(c.detail) for c in data.values()])  # 最大留言长度（留言详情一栏）
    min_comm_len = min([len(c.detail) for c in data.values()])  # 最短留言长度
    avg_comm_len = sum([len(c.detail) for c in data.values()]) / num_record  # 平均留言长度

    max_topic_len = max([len(c.topic) for c in data.values()])  # 最大留言主题长度（留言主题一栏）
    min_topic_len = min([len(c.topic) for c in data.values()])  # 最短主题长度
    avg_topic_len = sum([len(c.topic) for c in data.values()]) / num_record  # 平均主题长度

    max_num_like = min_num_like = avg_num_like \
        = max_num_tread = min_num_tread = avg_num_tread = None
    if list(data.values())[0].likes is not None:  # 只有附件三需要此分析
        max_num_like = max([int(c.likes) for c in data.values()])
        min_num_like = min([int(c.likes) for c in data.values()])
        avg_num_like = sum([int(c.likes) for c in data.values()]) / num_record
        max_num_tread = max([int(c.treads) for c in data.values()])
        min_num_tread = min([int(c.treads) for c in data.values()])
        avg_num_tread = sum([int(c.treads) for c in data.values()]) / num_record

    max_reply_len = min_reply_len = avg_reply_len = None
    if list(data.values())[0].reply is not None:  # 只有附件四需要此分析
        max_reply_len = max([len(c.reply) for c in data.values()])
        min_reply_len = min([len(c.reply) for c in data.values()])
        avg_reply_len = sum([len(c.reply) for c in data.values()]) / num_record

    # ------------------- 对预处理后的数据分析 -------------------
    max_num_seg_comm = max([len(c.seg_detail) for c in data.values()])  # （每条留言中的）最大词数
    min_num_seg_comm = min([len(c.seg_detail) for c in data.values()])  # 最小词数
    avg_num_seg_comm = sum([len(c.seg_detail) for c in data.values()]) / num_record  # 平均词数

    max_word_len = max([len(word) for c in data.values() for word in c.seg_detail])
    min_word_len = min([len(word) for c in data.values() for word in c.seg_detail])
    # 计算平均词长应该除以总词数
    avg_word_len = sum([len(word) for c in data.values() for word in c.seg_detail]) \
                   / sum([len(c.seg_detail) for c in data.values()])

    max_num_seg_reply = min_num_seg_reply = avg_num_seg_reply \
        = max_reply_word_len = min_reply_word_len = avg_reply_word_len = None
    if list(data.values())[0].reply is not None:  # 只有附件四需要此分析
        max_num_seg_reply = max([len(c.seg_reply) for c in data.values()])
        min_num_seg_reply = min([len(c.seg_reply) for c in data.values()])
        avg_num_seg_reply = sum([len(c.seg_reply) for c in data.values()]) / num_record

        max_reply_word_len = max([len(word) for c in data.values() for word in c.seg_reply])
        min_reply_word_len = min([len(word) for c in data.values() for word in c.seg_reply])
        # 计算平均词长应该除以总词数
        avg_reply_word_len = sum([len(word) for c in data.values() for word in c.seg_reply]) \
                             / sum([len(c.seg_detail) for c in data.values()])

    print(
        f'''
数据分析，cut_all={kwargs['cut_all']}
数据集名称：{dataset_name}
记录条数（去重前）：{raw_num_record}
记录条数（去重后）：{num_record}
----------------------------
**预处理前**
留言长度
最大：{max_comm_len}
最小：{min_comm_len}
平均：{avg_comm_len}
----------------------------
留言主题长度
最大：{max_topic_len}
最小：{min_topic_len}
平均：{avg_topic_len}
----------------------------
点赞数量
最大：{max_num_like}
最小：{min_num_like}
平均：{avg_num_like}
----------------------------
反对数量
最大：{max_num_tread}
最小：{min_num_tread}
平均：{avg_num_tread}
----------------------------
答复信息长度
最大：{max_reply_len}
最小：{min_reply_len}
平均：{avg_reply_len}
----------------------------
**预处理后**
留言中的词数
最大：{max_num_seg_comm}
最小：{min_num_seg_comm}
平均：{avg_num_seg_comm}
----------------------------
留言词长
最大：{max_word_len}
最小：{min_word_len}
平均：{avg_word_len}
----------------------------
答复词数
最大：{max_num_seg_reply}
最小：{min_num_seg_reply}
平均：{avg_num_seg_reply}
----------------------------
答复词长
最大：{max_reply_word_len}
最小：{min_reply_word_len}
平均：{avg_reply_word_len}
''')
