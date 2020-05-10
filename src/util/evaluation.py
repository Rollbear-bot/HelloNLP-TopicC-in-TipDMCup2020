# -*- coding: utf-8 -*-
# @Time: 2020/4/24 7:56
# @Author: Rollbear
# @Filename: evaluation.py
# 评价模块

import time
from datetime import datetime

from sklearn.externals import joblib  # 模型保存与加载

from entity.comm import Comm
from util.vec import *


class InitException(Exception):
    """初始化异常"""
    def __str__(self):
        return "资源未加载"


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
    w_n_text = 8  # 簇中的文本数量所占权重
    w_n_like = 1  # 点赞数的权重
    w_n_tread = -30  # 反对的数量的权重（一般认为是负值）
    w_date_variance = 1  # 簇内文本发布日期的方差倒数权重

    def __init__(self, cluster: list):
        """
        在一个聚类完成的类别中给出热度评分
        :param cluster: list of tuple(7)
        """
        self.cluster = cluster
        self.n_text = len(cluster)
        # 注意完整数据集“点赞数”属性的位置与示例数据集不同
        self.n_like = sum([row[6] for row in cluster])
        self.n_tread = sum([row[5] for row in cluster])

        # 发布日期的方差以“日”为最小计算单位
        date_avg = sum([_process_date_str(row[3]) for row in cluster]) / len(cluster)
        self.date_variance = sum(
            [(_process_date_str(row[3]) - date_avg) ** 2 for row in cluster]
        ) / len(cluster)

    @property
    def date_range_str(self):
        sorted_date = sorted([row[3] for row in self.cluster],
                             key=lambda date: _process_date_str(date))
        return f"{str(sorted_date[0]).split()[0]} 至 {str(sorted_date[-1]).split()[0]}"

    @property
    def score(self):
        sc = HotspotEvaluation.w_n_text * self.n_text \
               + HotspotEvaluation.w_n_like * self.n_like \
               + HotspotEvaluation.w_n_tread * self.n_tread \
               + HotspotEvaluation.w_date_variance * ((1/self.date_variance) if self.date_variance != 0 else 0)
        return round(sc, 2)


class ReplyEvaluation:
    """回复文本评价"""
    # 参数权重（类字段）
    w_similarity = 1
    w_integrity = 1
    w_interpretability = 1
    w_timeliness = 1

    # 指标评价器（分类器）
    integrity_clf = None  # 完整性指标分类器
    interpretability_clf = None  # 可解释性指标分类器

    def __init__(self, comm: Comm, model):
        """
        构造方法，解析一个带有回复的留言对象
        :param comm: 留言对象
        :param model: 词向量化模型
        """
        # 如果评价模型还未加载，则不能进行评价
        if ReplyEvaluation.interpretability_clf is None \
                or ReplyEvaluation.integrity_clf is None:
            raise InitException
        
        # 抽取经过预处理的文本
        topic = comm.seg_topic  # 留言主题
        detail = comm.seg_detail  # 留言详情
        reply = comm.seg_reply  # 回复

        # 时间信息
        comm_date = _process_date_str(comm.date)
        reply_date = _process_date_str(comm.reply_date)

        # 将“主题”和“留言详情”共同作为留言文本，计算文档向量
        text_vec = doc_vec(topic + detail, model)
        # 计算回复文本的文档向量
        reply_vec = doc_vec(reply, model)

        # 基于numpy矩阵，计算相似度（向量空间中的欧式距离）
        self.__similarity = -1/(numpy.linalg.norm(text_vec - reply_vec))
        # 计算时间差，其倒数作为时效性指标
        self.__timeliness = (1/(reply_date - comm_date) if reply_date != comm_date else 0)

        # 使用评价模型对后两个指标打分
        self.__integrity = ReplyEvaluation.integrity_clf.predict([reply_vec])[0]
        self.__interpretability = ReplyEvaluation.interpretability_clf.predict([reply_vec])[0]

    @classmethod
    def load_integrity_clf(cls, model_path: str):
        """加载完整性指标分类模型"""
        ReplyEvaluation.integrity_clf = joblib.load(model_path)

    @classmethod
    def load_interpretability_clf(cls, model_path: str):
        """加载可解释性指标分类模型"""
        ReplyEvaluation.interpretability_clf = joblib.load(model_path)

    @property
    def score(self):
        """回复评价得分"""
        score = ReplyEvaluation.w_similarity * self.__similarity \
               + ReplyEvaluation.w_integrity * self.__integrity \
               + ReplyEvaluation.w_interpretability * self.__interpretability \
               + ReplyEvaluation.w_timeliness * self.__timeliness
        return round(score, 2)

