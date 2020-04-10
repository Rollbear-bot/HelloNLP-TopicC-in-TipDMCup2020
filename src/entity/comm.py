# -*- coding: utf-8 -*-
# @Time: 2020/4/10 23:27
# @Author: Rollbear
# @Filename: comm.py

import jieba


class Comm(object):
    """留言"""

    def __init__(self):
        """构造方法"""
        self.comm_id = None  # 留言编号
        self.user_id = None  # 用户编号
        self.topic = None  # 留言主题
        self.date = None  # 留言时间
        self.detail = None  # 留言详情
        self.fir_lev_label = None  # 一级标签
        self.likes = None  # 点赞数
        self.treads = None  # 反对数
        self.reply = None  # 答复
        self.reply_date = None  # 答复时间

        self.seg_topic = None  # 分词后的留言主题
        self.seg_detail = None  # 分词后的留言详情
        self.seg_reply = None  # 分词后的留言回复classmethod

    def load_from_row(self, row):
        """从行元组加载实例"""
        if len(row) == 6:
            # 六元组是附件二的格式
            self.comm_id, self.user_id, self.topic, \
                self.date, self.detail, self.fir_lev_label = row
        if len(row) == 7:
            # 附件三和附件四都是七元组
            if isinstance(row[5], int):
                # 附件三的第六列是点赞数，是整型数据
                self.comm_id, self.user_id, self.topic, \
                    self.date, self.detail, self.likes, self.treads = row
            else:
                # 附件四
                self.comm_id, self.user_id, self.topic, \
                    self.date, self.detail, self.reply, self.reply_date = row

    def cut(self, cut_all=False, stop_words_lt=None):
        """分词并去停用词"""
        # 分别对主题、详情、回复字段分词
        self.seg_topic = jieba.lcut(self.topic, cut_all=cut_all)
        self.seg_detail = jieba.lcut(self.detail, cut_all=cut_all)
        if self.reply is not None:
            self.seg_reply = jieba.lcut(self.reply, cut_all=cut_all)

        # 去停用词
        if stop_words_lt is not None:
            # 扫描主题
            for index, word in enumerate(self.seg_topic):
                if word in stop_words_lt:
                    # 若扫描到一个词在停用词表中，则剔除这个词
                    self.seg_topic.pop(index)
            # 扫描详情
            for index, word in enumerate(self.seg_detail):
                if word in stop_words_lt:
                    self.seg_detail.pop(index)
            # 扫描回复
            if self.seg_reply is not None:
                for index, word in enumerate(self.seg_reply):
                    if word in stop_words_lt:
                        self.seg_reply.pop(index)