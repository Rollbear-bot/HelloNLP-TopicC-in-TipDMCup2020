# -*- coding: utf-8 -*-
# @Time: 2020/3/28 16:41
# @Author: Rollbear
# @Filename: label.py
# “标签”的定义


class LabelNode:
    """标签类"""
    def __init__(self, name: str):
        """构造方法"""
        self.sub_label = []  # 子标签
        self.label_name = name  # 标签名

    def __eq__(self, other):
        return self.label_name == other.label_name

    def __str__(self):
        return self.label_name

