# -*- coding: utf-8 -*-
# @Time: 2020/4/24 7:56
# @Author: Rollbear
# @Filename: evaluation.py
# 评价模块

import time
from datetime import datetime

from entity.comm import Comm
from util.vec import *


def _process_date_str(date_str: str):
    """将字符格式的日期信息转化为可计算的日期对象"""
    if isinstance(date_str, datetime):
        return time.mktime(date_str.timetuple())
    if "/" in date_str:
        year, month = map(lambda x: int(x), str(date_str).split("/")[:2])
        day = int(str(date_str).split("/")[-1].split(" ")[0])
    else:
        year, month = map(lambda x: int(x), str(date_str).split("-")[:2])
        day = int(str(date_str).split("-")[-1].split(" ")[0])
    date = time.mktime((year, month, day, 0, 0, 0, 0, 0, 0))
    return date


class HotspotEvaluation:
    """文本热度评估"""
    # 热度参数权重（类字段）
    w_n_text = 100  # 簇中的文本数量所占权重
    w_n_like = 1  # 点赞数的权重
    w_n_tread = -30  # 反对的数量的权重（一般认为是负值）
    w_date_variance = 0  # 簇内文本发布日期的方差权重（一般为负值）

    def __init__(self, cluster: list):
        """
        在一个聚类完成的类别中给出热度评分
        :param cluster: list of tuple(7)
        """
        self.n_text = len(cluster)
        self.n_like = sum([row[5] for row in cluster])
        self.n_tread = sum([row[6] for row in cluster])

        # 发布日期的方差以“日”为最小计算单位
        date_avg = sum([_process_date_str(row[3]) for row in cluster]) / len(cluster)
        self.date_variance = sum(
            [(_process_date_str(row[3]) - date_avg) ** 2 for row in cluster]
        ) / len(cluster)

    @property
    def score(self):
        return HotspotEvaluation.w_n_text * self.n_text \
               + HotspotEvaluation.w_n_like * self.n_like \
               + HotspotEvaluation.w_n_tread * self.n_tread \
               + HotspotEvaluation.w_date_variance * self.date_variance


class ReplyEvaluation:
    """回复文本评价"""
    # 参数权重（类字段）
    w_similarity = 1
    w_integrity = 1
    w_interpretability = 1

    def __init__(self, comm: Comm, model):
        """
        构造方法，解析一个带有回复的留言对象
        :param comm:
        :param model:
        """
        # 抽取经过预处理的文本
        topic = comm.seg_topic  # 留言主题
        detail = comm.seg_detail  # 留言详情
        reply = comm.seg_reply  # 回复

        # 将“主题”和“留言详情”共同作为留言文本，计算文档向量
        text_vec = doc_vec(topic + detail, model)
        # 计算回复文本的文档向量
        reply_vec = doc_vec(reply, model)

        # 基于numpy矩阵，计算相似度（向量空间中的欧式距离）
        self.__similarity = numpy.linalg.norm(text_vec - reply_vec)

        self.__integrity = 0  # todo::完整性，尝试使用手工标注的方式训练
        self.__interpretability = 0  # todo::可解释性，同上

    @property
    def score(self):
        """回复评价得分"""
        return ReplyEvaluation.w_similarity * self.__similarity \
               + ReplyEvaluation.w_integrity * self.__integrity \
               + ReplyEvaluation.w_interpretability * self.__interpretability


def integrity_evaluation(reply_text: str, model):
    pass
