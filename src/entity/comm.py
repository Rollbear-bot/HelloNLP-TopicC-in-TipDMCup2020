# -*- coding: utf-8 -*-
# @Time: 2020/4/10 23:27
# @Author: Rollbear
# @Filename: comm.py

import jieba
import numpy as np


# 建立“留言”对象，映射表中的每一行（留言）
# 同时储存分词、标签等处理结果
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
        self.seg_reply = None  # 分词后的留言回复
        self.integrity = None  # 答复完整性指标
        self.interpretability = None  # 答复可解释性指标

    def load_from_row(self, row, full_dataset=False):
        """从行元组加载实例"""
        if len(row) == 6:
            # 六元组是附件二的格式
            self.comm_id, self.user_id, self.topic, \
                self.date, self.detail, self.fir_lev_label = row

        if len(row) == 7:
            # 附件三和附件四都是七元组
            if isinstance(row[5], int):
                # 附件三的第六列是点赞数，是整型数据
                if full_dataset is False:
                    # 示例数据集
                    self.comm_id, self.user_id, self.topic, \
                        self.date, self.detail, self.likes, self.treads = row
                else:
                    # 完整数据集，“点赞”“反对”这两列的排列顺序和示例数据集不同
                    self.comm_id, self.user_id, self.topic, \
                        self.date, self.detail, self.treads, self.likes = row
            else:
                # 附件四
                self.comm_id, self.user_id, self.topic, \
                    self.date, self.detail, self.reply, self.reply_date = row

        elif len(row) == 9:  # 手工标注后的附件四，相比原始的附件四附加了完整性指标、可解释性指标两个字段
            self.comm_id, self.user_id, self.topic, \
                self.date, self.detail, self.reply, self.reply_date, \
                self.integrity, self.interpretability = row

        self.detail = self.detail.strip().rstrip()  # 抛弃“详情”一列的空白字符

    def cut(self, cut_all=False, stop_words_lt=None):
        """分词并去停用词"""
        # 跳过无意义的行
        if self.comm_id is None:
            return

        # 分别对主题、详情、回复字段分词
        self.seg_topic = jieba.lcut(self.topic, cut_all=cut_all)
        # “详情”一列，使用strip方法跳过空白字符
        self.seg_detail = jieba.lcut(self.detail, cut_all=cut_all)
        if self.reply is not None:
            self.seg_reply = jieba.lcut(self.reply, cut_all=cut_all)

        # 去停用词
        if stop_words_lt is not None:
            # 扫描主题
            self.seg_topic = [word for word in self.seg_topic if word not in stop_words_lt]
            # 扫描详情
            self.seg_detail = [word for word in self.seg_detail if word not in stop_words_lt]
            # 扫描回复
            if self.reply is not None:
                self.seg_reply = [word for word in self.seg_reply if word not in stop_words_lt]

    def get_vec(self, model):
        """计算词向量"""
        vec = []
        topic_vec = []
        detail_vec = []

        for word in self.seg_topic:
            try:
                word_vec = model[word]
            except KeyError:  # 跳过未登录词
                continue
            topic_vec.append(word_vec)
        for word in self.seg_detail:
            try:
                word_vec = model[word]
            except KeyError:  # 跳过未登录词
                continue
            detail_vec.append(word_vec)
        vec.append(topic_vec)
        vec.append(detail_vec)

        if self.seg_reply is not None:
            reply_vec = []
            for word in self.seg_reply:
                try:
                    word_vec = model[word]
                except KeyError:  # 跳过未登录词
                    continue
                reply_vec.append(word_vec)
            vec.append(reply_vec)
        return np.array(vec, dtype=np.float64)

    # 从行生成Comm字典的方法
    @classmethod
    def generate_comm_dict(cls, row_lt, cut_all=False, stop_words_lt=None, full_dataset=False):
        """从数据行生成key为留言编号，value为Comm对象的字典
        并对字典中的Comm对象执行分词、去停用词等预处理"""
        comm_dict = {}
        for row in row_lt:
            c = Comm()
            c.load_from_row(row, full_dataset=full_dataset)
            c.cut(cut_all=cut_all, stop_words_lt=stop_words_lt)
            comm_dict[row[0]] = c
        return comm_dict
