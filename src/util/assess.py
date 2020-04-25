# -*- coding: utf-8 -*-
# @Time: 2020/4/24 7:56
# @Author: Rollbear
# @Filename: assess.py
# 评价模块

import time
from datetime import datetime


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


class HotspotAssess:
    """文本热度评估"""
    # 热度参数权重（类字段）
    w_n_text = 1  # 簇中的文本数量所占权重
    w_n_like = 1  # 点赞数的权重
    w_n_tread = -1  # “踩”的数量的权重（一般认为是负值）
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
        return HotspotAssess.w_n_text * self.n_text \
            + HotspotAssess.w_n_like * self.n_like \
            + HotspotAssess.w_n_tread * self.n_tread \
            + HotspotAssess.w_date_variance * self.date_variance


class ReplyAssess:
    """回复文本评价"""
    pass
